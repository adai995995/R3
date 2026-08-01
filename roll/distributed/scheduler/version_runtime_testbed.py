"""Deterministic trace-driven testbed for version-aware AgenticRL runtimes.

The simulator deliberately models only system behavior: policy-version deadlines,
admission, inference service quanta, tool delay, and version-scoped KV placement.
It never models reward or changes a trajectory's action sequence.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from roll.distributed.scheduler.rollout_scheduler import (
    apply_dynamic_reserve_hysteresis,
    build_version_runtime_plan,
    compute_closed_loop_reserve,
)
from roll.distributed.scheduler.router import (
    TrajectoryRuntimeState,
    build_runtime_priority_key,
    select_rebuild_worker,
)


@dataclass(frozen=True)
class TraceTrajectory:
    trajectory_id: str
    total_actions: int
    prefix_id: int
    prefix_tokens: int
    tokens_per_action: int
    response_tokens_per_action: int
    tool_seconds_per_action: float = 0.0


@dataclass(frozen=True)
class TestbedConfig:
    versions: int = 20
    learner_demand: int = 8
    service_actions_per_version: int = 24
    staleness_tolerance: int = 2
    max_outstanding: int = 32
    safety_reserve: int = 4
    workers: int = 4
    rebuild_budget: int = 8
    phases: Tuple["TestbedPhase", ...] = ()
    adaptive_reserve: bool = False
    reserve_min: int = 0
    reserve_max: int = 16
    reserve_additive_step: int = 2
    reserve_decay: float = 0.5
    reserve_ewma_alpha: float = 0.5
    reserve_wait_high: float = 0.5
    reserve_overload_high: float = 0.25
    reserve_warmup_versions: int = 0
    reserve_signal_patience: int = 1
    reserve_cooldown_versions: int = 0


@dataclass(frozen=True)
class TestbedPhase:
    """Deterministic workload override starting at one policy version."""

    start_version: int
    learner_demand: Optional[int] = None
    service_actions_per_version: Optional[float] = None
    tool_delay_scale: float = 0.0
    prefill_cost_per_1k_tokens: float = 0.0
    worker_slowdowns: Tuple[float, ...] = ()


def _phase_for_version(config: TestbedConfig, version: int) -> TestbedPhase:
    active = TestbedPhase(start_version=0)
    for phase in sorted(config.phases, key=lambda item: item.start_version):
        if phase.start_version > version:
            break
        active = phase
    return active


def _phase_learner_demand(config: TestbedConfig, phase: TestbedPhase) -> int:
    return max(
        0,
        config.learner_demand
        if phase.learner_demand is None
        else int(phase.learner_demand),
    )


def _phase_service_budget(config: TestbedConfig, phase: TestbedPhase) -> float:
    return max(
        0.0,
        float(config.service_actions_per_version)
        if phase.service_actions_per_version is None
        else float(phase.service_actions_per_version),
    )


def _ewma(current: Optional[float], sample: float, alpha: float) -> float:
    bounded_alpha = min(1.0, max(0.0, float(alpha)))
    if current is None:
        return float(sample)
    return bounded_alpha * float(sample) + (1 - bounded_alpha) * current


@dataclass
class _TrajectoryState:
    spec: TraceTrajectory
    policy_version: int
    admission_order: int
    actions_completed: int = 0
    finish_version: Optional[int] = None
    worker: Optional[int] = None
    cached_prompt_tokens: int = 0
    invested_prompt_tokens: int = 0
    invested_response_tokens: int = 0

    @property
    def remaining_actions(self) -> int:
        return max(0, self.spec.total_actions - self.actions_completed)

    @property
    def complete(self) -> bool:
        return self.actions_completed >= self.spec.total_actions

    def prompt_tokens(self) -> int:
        return self.spec.prefix_tokens + self.actions_completed * self.spec.tokens_per_action


def generate_trace(
    count: int,
    seed: int,
    *,
    min_actions: int = 2,
    max_actions: int = 12,
    prefix_classes: int = 8,
) -> List[TraceTrajectory]:
    rng = random.Random(seed)
    trace = []
    for index in range(max(0, count)):
        trace.append(
            TraceTrajectory(
                trajectory_id=f"trace-{index}",
                total_actions=rng.randint(min_actions, max_actions),
                prefix_id=rng.randrange(max(1, prefix_classes)),
                prefix_tokens=rng.choice((256, 512, 1024, 2048)),
                tokens_per_action=rng.choice((64, 96, 128, 192)),
                response_tokens_per_action=rng.choice((32, 48, 64, 96)),
                tool_seconds_per_action=rng.choice((0.0, 0.05, 0.2, 0.5)),
            )
        )
    return trace


def dump_trace(trace: Iterable[TraceTrajectory], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(item) for item in trace], indent=2, sort_keys=True) + "\n"
    )


def load_trace(path: Path) -> List[TraceTrajectory]:
    return [TraceTrajectory(**item) for item in json.loads(path.read_text())]


def _priority_key(state: _TrajectoryState) -> Tuple[int, int, int, int]:
    runtime_state = TrajectoryRuntimeState(
        trajectory_id=state.spec.trajectory_id,
        policy_version=state.policy_version,
        current_version=state.policy_version,
        version_age=0,
        actions_completed=state.actions_completed,
        max_actions=state.spec.total_actions,
        group_id=state.admission_order,
        episode_id=0,
    )
    return (*build_runtime_priority_key(runtime_state, {}), state.admission_order)


def _predict_timely_completions(
    active: List[_TrajectoryState],
    service_budget: float,
    phase: TestbedPhase,
) -> int:
    """Predict completions using only remaining service and version urgency."""
    budget = max(0.0, float(service_budget))
    completed = 0
    for state in sorted((item for item in active if not item.complete), key=_priority_key):
        remaining_cost = state.remaining_actions * (
            1.0
            + max(0.0, state.spec.tool_seconds_per_action)
            * max(0.0, phase.tool_delay_scale)
        )
        if state.cached_prompt_tokens <= 0:
            remaining_cost += (
                state.prompt_tokens()
                * max(0.0, phase.prefill_cost_per_1k_tokens)
                / 1000.0
            )
        if remaining_cost > budget:
            continue
        budget -= remaining_cost
        completed += 1
    return completed


def _worker_slowdown(phase: TestbedPhase, worker: int) -> float:
    if worker < len(phase.worker_slowdowns):
        return max(1e-6, float(phase.worker_slowdowns[worker]))
    return 1.0


def _action_service_cost(
    state: _TrajectoryState,
    worker: int,
    prefill_tokens: int,
    phase: TestbedPhase,
) -> float:
    base = (
        1.0
        + max(0.0, state.spec.tool_seconds_per_action)
        * max(0.0, phase.tool_delay_scale)
        + max(0, int(prefill_tokens))
        * max(0.0, phase.prefill_cost_per_1k_tokens)
        / 1000.0
    )
    return base * _worker_slowdown(phase, worker)


def _select_worker(
    state: _TrajectoryState,
    policy: str,
    version: int,
    worker_prefixes: Dict[int, set],
    worker_assignments: Dict[int, int],
    rebuild_remaining: int,
    phase: TestbedPhase,
) -> Tuple[int, str]:
    if state.worker is not None:
        return state.worker, "affinity"
    workers = sorted(worker_assignments)
    least_loaded = min(
        workers,
        key=lambda rank: (
            worker_assignments[rank] * _worker_slowdown(phase, rank),
            rank,
        ),
    )
    if policy != "unified":
        return least_loaded, "least_loaded"
    if rebuild_remaining > 0 and state.policy_version < version:
        synthetic_prompt = [state.spec.prefix_id] * min(
            state.spec.prefix_tokens, 32
        )
        synthetic_worker_prompts = {
            rank: [
                [prefix_id] * min(state.spec.prefix_tokens, 32)
                for prefix_id in worker_prefixes[rank]
            ]
            for rank in workers
        }
        worker, _ = select_rebuild_worker(
            synthetic_prompt,
            synthetic_worker_prompts,
            workers,
            worker_assignments,
            32,
        )
        return worker, "rebuild"
    prefix_workers = [
        rank for rank in workers if state.spec.prefix_id in worker_prefixes[rank]
    ]
    if prefix_workers:
        return min(
            prefix_workers,
            key=lambda rank: (
                worker_assignments[rank] * _worker_slowdown(phase, rank),
                rank,
            ),
        ), "prefix"
    return least_loaded, "least_loaded"


def run_testbed(
    trace: List[TraceTrajectory], config: TestbedConfig, policy: str
) -> Dict[str, object]:
    if policy not in ("fixed_fifo", "unified"):
        raise ValueError(f"unsupported policy: {policy}")
    if not trace:
        raise ValueError("trace must contain at least one trajectory")

    active: List[_TrajectoryState] = []
    trace_cursor = 0
    admission_order = 0
    fifo_cursor = 0
    adaptive_reserve = max(0, int(config.safety_reserve))
    undersupply_ewma: Optional[float] = None
    overload_ewma: Optional[float] = None
    pending_direction = 0
    pending_count = 0
    cooldown_remaining = 0
    metrics: Dict[str, float] = {
        "admitted_trajectories": 0,
        "completed_trajectories": 0,
        "consumed_trajectories": 0,
        "consumed_version_age_sum": 0,
        "learner_shortfall_trajectories": 0,
        "stale_trajectories": 0,
        "stale_actions": 0,
        "stale_inference_tokens": 0,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "saved_prefill_tokens": 0,
        "prefill_tokens": 0,
        "tool_seconds": 0.0,
        "rebuild_requests": 0,
        "prefix_routes": 0,
    }
    boundaries = []

    for version in range(config.versions):
        phase = _phase_for_version(config, version)
        learner_demand = _phase_learner_demand(config, phase)
        service_budget = _phase_service_budget(config, phase)
        reserve_before = adaptive_reserve
        stale_tokens_before = metrics["stale_inference_tokens"]
        stale_trajectories_before = metrics["stale_trajectories"]
        survivors = []
        for state in active:
            age = version - state.policy_version
            if age <= config.staleness_tolerance:
                state.worker = None
                state.cached_prompt_tokens = 0
                survivors.append(state)
                continue
            metrics["stale_trajectories"] += 1
            metrics["stale_actions"] += state.actions_completed
            metrics["stale_inference_tokens"] += (
                state.invested_prompt_tokens + state.invested_response_tokens
            )
        active = survivors

        ready_supply = sum(1 for state in active if state.complete)
        if policy == "unified":
            predicted = _predict_timely_completions(
                active, service_budget, phase
            )
            invested_candidates = [
                (
                    state.admission_order,
                    0,
                    version - state.policy_version,
                    state.actions_completed,
                    int(state.actions_completed > 0),
                )
                for state in active
                if not state.complete and state.actions_completed > 0
            ]
            runtime_plan = build_version_runtime_plan(
                version=version,
                learner_demand=learner_demand,
                safety_reserve=adaptive_reserve,
                expected_existing_supply=ready_supply + predicted,
                outstanding_trajectories=len(active),
                max_outstanding_trajectories=config.max_outstanding,
                admission_width=1,
                group_size=1,
                staleness_tolerance=config.staleness_tolerance,
                invested_candidate_groups=invested_candidates,
                gpu_invested_candidate_groups=invested_candidates,
            )
            requested = runtime_plan.admission_budget
        else:
            predicted = 0
            requested = learner_demand + config.safety_reserve
        capacity = max(0, config.max_outstanding - len(active))
        admitted = min(requested, capacity, len(trace) - trace_cursor)
        for _ in range(admitted):
            active.append(
                _TrajectoryState(
                    spec=trace[trace_cursor],
                    policy_version=version,
                    admission_order=admission_order,
                )
            )
            trace_cursor += 1
            admission_order += 1
        metrics["admitted_trajectories"] += admitted

        worker_prefixes = {rank: set() for rank in range(max(1, config.workers))}
        worker_assignments = {rank: 0 for rank in worker_prefixes}
        rebuild_remaining = min(
            config.rebuild_budget,
            sum(1 for state in active if not state.complete and state.policy_version < version),
        ) if policy == "unified" else 0

        remaining_service_budget = service_budget
        service_actions = 0
        while remaining_service_budget > 0:
            runnable = [state for state in active if not state.complete]
            if not runnable:
                break
            if policy == "unified":
                state = min(runnable, key=_priority_key)
            else:
                runnable.sort(key=lambda item: item.admission_order)
                state = runnable[fifo_cursor % len(runnable)]
                fifo_cursor += 1

            prompt_tokens = state.prompt_tokens()
            worker, reason = _select_worker(
                state,
                policy,
                version,
                worker_prefixes,
                worker_assignments,
                rebuild_remaining,
                phase,
            )

            if state.worker == worker and state.cached_prompt_tokens > 0:
                saved = min(prompt_tokens, state.cached_prompt_tokens)
            elif state.spec.prefix_id in worker_prefixes[worker]:
                saved = min(prompt_tokens, state.spec.prefix_tokens)
            else:
                saved = 0
            prefill_tokens = prompt_tokens - saved
            action_cost = _action_service_cost(
                state, worker, prefill_tokens, phase
            )
            if action_cost > remaining_service_budget:
                break
            remaining_service_budget -= action_cost
            service_actions += 1

            if reason == "rebuild":
                rebuild_remaining -= 1
                metrics["rebuild_requests"] += 1
            elif reason == "prefix":
                metrics["prefix_routes"] += 1

            response_tokens = state.spec.response_tokens_per_action
            metrics["prompt_tokens"] += prompt_tokens
            metrics["response_tokens"] += response_tokens
            metrics["saved_prefill_tokens"] += saved
            metrics["prefill_tokens"] += prefill_tokens
            metrics["tool_seconds"] += state.spec.tool_seconds_per_action * (
                1.0 + max(0.0, phase.tool_delay_scale)
            )
            state.invested_prompt_tokens += prompt_tokens
            state.invested_response_tokens += response_tokens
            state.worker = worker
            state.cached_prompt_tokens = prompt_tokens
            state.actions_completed += 1
            worker_assignments[worker] += 1
            worker_prefixes[worker].add(state.spec.prefix_id)
            if state.complete:
                state.finish_version = version
                metrics["completed_trajectories"] += 1

        # The learner consumes the batch that closes this policy-version window.
        # Any excess remains ready and is visible to the next boundary plan.
        ready = sorted(
            (state for state in active if state.complete),
            key=lambda state: (state.finish_version, state.admission_order),
        )
        consumed = ready[:learner_demand]
        consumed_ids = {state.spec.trajectory_id for state in consumed}
        active = [state for state in active if state.spec.trajectory_id not in consumed_ids]
        metrics["consumed_trajectories"] += len(consumed)
        learner_shortfall = max(0, learner_demand - len(consumed))
        metrics["learner_shortfall_trajectories"] += learner_shortfall
        metrics["consumed_version_age_sum"] += sum(
            version - state.policy_version for state in consumed
        )

        stale_tokens = max(
            0.0, metrics["stale_inference_tokens"] - stale_tokens_before
        )
        stale_trajectories = max(
            0.0, metrics["stale_trajectories"] - stale_trajectories_before
        )
        consumed_tokens = sum(
            state.invested_prompt_tokens + state.invested_response_tokens
            for state in consumed
        )
        overload_signal = (
            stale_tokens / (stale_tokens + consumed_tokens)
            if stale_tokens + consumed_tokens > 0
            else 0.0
        )
        undersupply_signal = float(learner_shortfall)
        undersupply_ewma = _ewma(
            undersupply_ewma,
            undersupply_signal,
            config.reserve_ewma_alpha,
        )
        overload_ewma = _ewma(
            overload_ewma,
            overload_signal,
            config.reserve_ewma_alpha,
        )
        reserve_reason = 0
        if policy == "unified" and config.adaptive_reserve:
            candidate_reserve, candidate_reason = compute_closed_loop_reserve(
                adaptive_reserve,
                version,
                undersupply_ewma,
                overload_ewma,
                reserve_min=config.reserve_min,
                reserve_max=config.reserve_max,
                additive_step=config.reserve_additive_step,
                multiplicative_decay=config.reserve_decay,
                warmup_versions=config.reserve_warmup_versions,
                wait_high=config.reserve_wait_high,
                overload_high=config.reserve_overload_high,
            )
            (
                adaptive_reserve,
                reserve_reason,
                pending_direction,
                pending_count,
                cooldown_remaining,
            ) = apply_dynamic_reserve_hysteresis(
                adaptive_reserve,
                candidate_reserve,
                candidate_reason,
                pending_direction,
                pending_count,
                cooldown_remaining,
                signal_patience=config.reserve_signal_patience,
                cooldown_versions=config.reserve_cooldown_versions,
            )

        boundaries.append(
            {
                "version": version,
                "learner_demand": learner_demand,
                "service_budget": service_budget,
                "service_actions": service_actions,
                "ready_supply": ready_supply,
                "predicted_existing_completions": predicted,
                "admitted": admitted,
                "learner_shortfall": learner_shortfall,
                "stale_trajectories": stale_trajectories,
                "undersupply_signal": undersupply_signal,
                "undersupply_ewma": undersupply_ewma,
                "overload_signal": overload_signal,
                "overload_ewma": overload_ewma,
                "reserve_before": reserve_before,
                "reserve_after": adaptive_reserve,
                "reserve_update_reason": reserve_reason,
                "outstanding": len(active),
                "carryover": sum(
                    1 for state in active if state.policy_version < version
                ),
            }
        )

    consumed = metrics["consumed_trajectories"]
    prompt = metrics["prompt_tokens"]
    metrics["mean_consumed_version_age"] = (
        metrics["consumed_version_age_sum"] / consumed if consumed else 0.0
    )
    metrics["prefill_saved_ratio"] = (
        metrics["saved_prefill_tokens"] / prompt if prompt else 0.0
    )
    metrics["compute_conversion_ratio"] = (
        metrics["response_tokens"]
        / max(1.0, metrics["prefill_tokens"] + metrics["response_tokens"])
    )
    return {
        "policy": policy,
        "config": asdict(config),
        "metrics": metrics,
        "boundaries": boundaries,
        "trace_consumed": trace_cursor,
    }


def compare_policies(
    trace: List[TraceTrajectory], config: TestbedConfig
) -> Dict[str, object]:
    return {
        "fixed_fifo": run_testbed(trace, config, "fixed_fifo"),
        "unified": run_testbed(trace, config, "unified"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--write-trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectories", type=int, default=512)
    parser.add_argument("--versions", type=int, default=20)
    parser.add_argument("--learner-demand", type=int, default=8)
    parser.add_argument("--service-actions", type=int, default=24)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--max-outstanding", type=int, default=32)
    parser.add_argument("--reserve", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild-budget", type=int, default=8)
    args = parser.parse_args()

    trace = load_trace(args.trace) if args.trace else generate_trace(
        args.trajectories, args.seed
    )
    if args.write_trace:
        dump_trace(trace, args.write_trace)
    config = TestbedConfig(
        versions=args.versions,
        learner_demand=args.learner_demand,
        service_actions_per_version=args.service_actions,
        staleness_tolerance=args.tolerance,
        max_outstanding=args.max_outstanding,
        safety_reserve=args.reserve,
        workers=args.workers,
        rebuild_budget=args.rebuild_budget,
    )
    result = compare_policies(trace, config)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
