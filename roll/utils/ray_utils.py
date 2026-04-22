import os

import ray


@ray.remote
def get_visible_gpus(device_control_env_var: str):
    # Prefer the env var set by the platform (e.g. CUDA_VISIBLE_DEVICES).
    raw = (os.environ.get(device_control_env_var, "") or "").strip()
    if raw:
        # Filter empty segments to avoid returning [''].
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]

    # Fallback: when running under Ray placement groups, the env var may not be set
    # as expected depending on runtime/platform detection. Ray still tracks assigned
    # accelerators for the task/actor; use that view to derive GPU ids.
    try:
        gpu_ids = ray.get_gpu_ids()
        # Ray may return float ids; normalize to int strings for downstream code.
        return [str(int(x)) for x in gpu_ids]
    except Exception:
        return []


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
