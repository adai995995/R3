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
    return (
        state.policy_version,
        int(state.actions_completed == 0),
        state.remaining_actions,
        state.admission_order,
    )


def _predict_timely_completions(
    active: List[_TrajectoryState], service_budget: int
) -> int:
    """Predict completions using only remaining service and version urgency."""
    budget = max(0, service_budget)
    completed = 0
    for state in sorted((item for item in active if not item.complete), key=_priority_key):
        if state.remaining_actions > budget:
            continue
        budget -= state.remaining_actions
        completed += 1
    return completed


def _select_worker(
    state: _TrajectoryState,
    policy: str,
    version: int,
    worker_prefixes: Dict[int, set],
    worker_assignments: Dict[int, int],
    rebuild_remaining: int,
) -> Tuple[int, str]:
    if state.worker is not None:
        return state.worker, "affinity"
    workers = sorted(worker_assignments)
    least_loaded = min(workers, key=lambda rank: (worker_assignments[rank], rank))
    if policy != "unified":
        return least_loaded, "least_loaded"
    if rebuild_remaining > 0 and state.policy_version < version:
        worker = min(
            workers,
            key=lambda rank: (
                int(state.spec.prefix_id in worker_prefixes[rank]),
                worker_assignments[rank],
                rank,
            ),
        )
        return worker, "rebuild"
    prefix_workers = [
        rank for rank in workers if state.spec.prefix_id in worker_prefixes[rank]
    ]
    if prefix_workers:
        return min(prefix_workers, key=lambda rank: (worker_assignments[rank], rank)), "prefix"
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
                active, config.service_actions_per_version
            )
            requested = max(
                0,
                config.learner_demand
                + config.safety_reserve
                - ready_supply
                - predicted,
            )
        else:
            predicted = 0
            requested = config.learner_demand + config.safety_reserve
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

        for _ in range(max(0, config.service_actions_per_version)):
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
            )
            if reason == "rebuild":
                rebuild_remaining -= 1
                metrics["rebuild_requests"] += 1
            elif reason == "prefix":
                metrics["prefix_routes"] += 1

            if state.worker == worker and state.cached_prompt_tokens > 0:
                saved = min(prompt_tokens, state.cached_prompt_tokens)
            elif state.spec.prefix_id in worker_prefixes[worker]:
                saved = min(prompt_tokens, state.spec.prefix_tokens)
            else:
                saved = 0
            response_tokens = state.spec.response_tokens_per_action
            metrics["prompt_tokens"] += prompt_tokens
            metrics["response_tokens"] += response_tokens
            metrics["saved_prefill_tokens"] += saved
            metrics["prefill_tokens"] += prompt_tokens - saved
            metrics["tool_seconds"] += state.spec.tool_seconds_per_action
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
        consumed = ready[: config.learner_demand]
        consumed_ids = {state.spec.trajectory_id for state in consumed}
        active = [state for state in active if state.spec.trajectory_id not in consumed_ids]
        metrics["consumed_trajectories"] += len(consumed)
        metrics["learner_shortfall_trajectories"] += max(
            0, config.learner_demand - len(consumed)
        )
        metrics["consumed_version_age_sum"] += sum(
            version - state.policy_version for state in consumed
        )

        boundaries.append(
            {
                "version": version,
                "ready_supply": ready_supply,
                "predicted_existing_completions": predicted,
                "admitted": admitted,
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
