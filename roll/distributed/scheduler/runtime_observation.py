from typing import Any, Dict, Mapping


RUNTIME_OBSERVATION_SCHEMA_VERSION = 4


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _router_components(config: Any) -> Dict[str, bool]:
    router_args = _get(config, "router_args", {})
    router_config = _get(router_args, "router_config", {})
    return {
        "kv_post_update_rebuild": bool(
            _get(router_config, "post_update_rebuild_enabled", False)
        ),
        "kv_working_set_routing": bool(
            _get(router_config, "working_set_routing_enabled", False)
        ),
        "kv_soft_locality": bool(
            _get(router_config, "soft_locality_enabled", False)
        ),
    }


def runtime_components(config: Any) -> Dict[str, bool]:
    components = {
        "version_admission": (
            _get(config, "trajectory_admission_policy", "step")
            == "version_adaptive"
        ),
        "version_priority": (
            _get(config, "trajectory_scheduling_policy", "fifo")
            == "version_priority"
        ),
    }
    components.update(_router_components(config))
    return components


def runtime_variant(components: Mapping[str, bool]) -> str:
    enabled = sum(bool(value) for value in components.values())
    if enabled == 0:
        return "baseline"
    if enabled == len(components):
        return "full"
    return "ablation"


def finalize_runtime_observation_report(
    report: Dict[str, Any], config: Any, completed_training_steps: int
) -> Dict[str, Any]:
    """Attach comparable experiment metadata and use actual learner consumption."""
    metrics = report.setdefault("metrics", {})
    rollout_batch_size = int(_get(config, "rollout_batch_size", 0))
    fallback_consumed = max(0, int(completed_training_steps)) * max(
        0, rollout_batch_size
    )
    consumed_trajectories = int(
        metrics.get("consumed/trajectories", fallback_consumed)
    )
    placeholder_trajectories = int(
        metrics.get("consumed/placeholder_trajectories", 0)
    )
    valid_trajectories = int(
        metrics.get(
            "consumed/valid_trajectories",
            max(0, consumed_trajectories - placeholder_trajectories),
        )
    )
    metrics["consumed/valid_trajectories"] = valid_trajectories
    metrics["consumed/placeholder_trajectories"] = placeholder_trajectories
    metrics["consumed/valid_fraction"] = (
        valid_trajectories / consumed_trajectories
        if consumed_trajectories
        else 0.0
    )
    terminal_waste = int(metrics.get("terminal_waste/trajectories", 0))
    async_waste = int(metrics.get("async_waste/trajectories", 0))
    metrics["terminal_waste/consumed_trajectories"] = consumed_trajectories
    metrics["terminal_waste/waste_to_consumed_ratio"] = (
        terminal_waste / consumed_trajectories if consumed_trajectories else 0.0
    )
    metrics["async_waste/consumed_trajectories"] = consumed_trajectories
    metrics["async_waste/waste_to_consumed_ratio"] = (
        async_waste / consumed_trajectories if consumed_trajectories else 0.0
    )

    valid_inference_tokens = int(
        metrics.get(
            "consumed/valid_inference_tokens",
            metrics.get("consumed/inference_tokens", 0),
        )
    )
    router_prompt_tokens = int(
        metrics.get(
            "router_lifetime/vllm/request_prompt_tokens",
            metrics.get("vllm/request_prompt_tokens", 0),
        )
    )
    zero_progress_anomaly = int(
        valid_trajectories > 0
        and valid_inference_tokens == 0
        and router_prompt_tokens > 0
    )
    report["data_quality"] = {
        "valid_consumed_zero_inference_tokens": bool(zero_progress_anomaly),
        "router_prompt_tokens_observed": router_prompt_tokens,
        "valid_consumed_inference_tokens": valid_inference_tokens,
    }
    metrics["data_quality/zero_progress_anomaly"] = zero_progress_anomaly

    components = runtime_components(config)
    tolerance = _get(config, "trajectory_staleness_tolerance")
    if tolerance is None:
        tolerance = int(_get(config, "async_generation_ratio", 0))
    report["schema_version"] = RUNTIME_OBSERVATION_SCHEMA_VERSION
    report["experiment"] = {
        "exp_name": str(_get(config, "exp_name", "unknown")),
        "completed_training_steps": int(completed_training_steps),
        "rollout_batch_size": rollout_batch_size,
        "async_generation_ratio": float(
            _get(config, "async_generation_ratio", 0)
        ),
        "trajectory_staleness_tolerance": int(tolerance),
        "trajectory_admission_policy": str(
            _get(config, "trajectory_admission_policy", "step")
        ),
        "trajectory_scheduling_policy": str(
            _get(config, "trajectory_scheduling_policy", "fifo")
        ),
        "max_outstanding_trajectories": _get(
            config, "max_outstanding_trajectories"
        ),
        "adaptive_admission_reserve_trajectories": int(
            _get(config, "adaptive_admission_reserve_trajectories", 0)
        ),
        "adaptive_admission_bucketed_finish_enabled": bool(
            _get(config, "adaptive_admission_bucketed_finish_enabled", False)
        ),
        "version_adaptive_progress_floor_enabled": bool(
            _get(config, "version_adaptive_progress_floor_enabled", False)
        ),
        "dynamic_admission_reserve_enabled": bool(
            _get(config, "dynamic_admission_reserve_enabled", False)
        ),
        "dynamic_admission_reserve_controller": str(
            _get(
                config,
                "dynamic_admission_reserve_controller",
                "closed_loop_aimd",
            )
        ),
        "runtime_variant": runtime_variant(components),
        "runtime_components": components,
        "checkpointing_enabled": bool(
            _get(config, "enable_checkpointing", True)
        ),
        "save_final_checkpoint": bool(
            _get(config, "save_final_checkpoint", True)
        ),
    }
    return report
