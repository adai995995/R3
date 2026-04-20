from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple, Literal


@dataclass(frozen=True)
class ResumeFallbackContext:
    enable_resume_priority: bool
    enable_resume_aware_routing: bool
    active_dp_ranks: Set[int]
    # Backend status pulled from gateway/router (Path A). When unavailable, fields may be None.
    hint_present: bool
    hint_ready: Optional[bool]
    hint_inflight: Optional[int]
    hint_queue_depth: Optional[int]
    overloaded_inflight_threshold: int
    overloaded_queue_depth_threshold: int
    # Fallback local signal when pulled state is unavailable.
    max_running_requests_per_worker: int
    hint_worker_load: int


def parse_ratio(value: str) -> Tuple[int, int]:
    """Parse a quota ratio string like '3:1' -> (3, 1)."""
    raw = (value or "").strip()
    if not raw:
        return 1, 1
    parts = raw.split(":")
    if len(parts) != 2:
        return 1, 1
    try:
        a = max(0, int(parts[0]))
        b = max(0, int(parts[1]))
    except ValueError:
        return 1, 1
    if a == 0 and b == 0:
        return 1, 1
    return a, b


def bucketize_queue_wait_s(queue_wait_s: float) -> str:
    s = max(0.0, float(queue_wait_s))
    if s < 0.01:
        return "lt_10ms"
    if s < 0.1:
        return "lt_100ms"
    if s < 0.5:
        return "lt_500ms"
    if s < 1.0:
        return "lt_1s"
    if s < 3.0:
        return "lt_3s"
    if s < 10.0:
        return "lt_10s"
    return "ge_10s"


def bucketize_score(score: float) -> str:
    v = float(score)
    if v < -5.0:
        return "lt_-5"
    if v < -1.0:
        return "lt_-1"
    if v < 0.0:
        return "lt_0"
    if v < 1.0:
        return "lt_1"
    if v < 5.0:
        return "lt_5"
    return "ge_5"


def resume_fallback_reason(
    *,
    route_meta: Dict[str, Any],
    selected_dp_rank: int,
    previous_rank: Optional[int],
    ctx: ResumeFallbackContext,
    force_migrate_age_s: float,
) -> str:
    """
    Best-effort fallback reason for resume affinity, using only scheduler-visible signals.

    Returns one of:
    - hit
    - forced_migrate
    - disabled
    - no_hint
    - not_found
    - inactive
    - not_ready
    - overloaded
    - selected_other
    """
    if previous_rank is not None and selected_dp_rank == previous_rank:
        return "hit"
    pause_age = float(route_meta.get("pause_age_s", 0.0) or 0.0)
    if pause_age >= force_migrate_age_s:
        return "forced_migrate"
    if not ctx.enable_resume_priority or not ctx.enable_resume_aware_routing:
        return "disabled"
    last_backend_id = route_meta.get("last_backend_id")
    if last_backend_id is None:
        return "no_hint"
    if not isinstance(last_backend_id, int):
        return "not_found"
    if last_backend_id < 0:
        return "not_found"
    if last_backend_id not in ctx.active_dp_ranks:
        return "inactive"

    # Prefer pulled gateway status if available.
    if not ctx.hint_present:
        return "inactive"
    if ctx.hint_ready is False:
        return "not_ready"

    if ctx.hint_inflight is not None and ctx.overloaded_inflight_threshold > 0:
        if ctx.hint_inflight >= ctx.overloaded_inflight_threshold:
            return "overloaded"
    if ctx.hint_queue_depth is not None and ctx.overloaded_queue_depth_threshold > 0:
        if ctx.hint_queue_depth >= ctx.overloaded_queue_depth_threshold:
            return "overloaded"

    # Fallback: local per-worker capacity limit when pulled state is missing.
    if (
        (ctx.hint_inflight is None and ctx.hint_queue_depth is None)
        and ctx.max_running_requests_per_worker > 0
        and ctx.hint_worker_load >= ctx.max_running_requests_per_worker
    ):
        return "overloaded"
    if selected_dp_rank != last_backend_id:
        return "selected_other"
    return "hit"


def choose_queue_for_soft_quota(
    *,
    resume_empty: bool,
    normal_empty: bool,
    quota_resume_target: int,
    quota_normal_target: int,
    quota_cursor: int,
    oldest_resume_wait_s: float,
    oldest_normal_wait_s: float,
    resume_max_queue_wait_s: float,
    normal_max_queue_wait_s: float,
) -> Literal["resume", "normal", "none"]:
    """
    Pure decision helper for soft-quota scheduling.

    Implements:
    - timeout escape hatch (normal/resume)
    - skip-on-empty
    - quota cycle preference
    """
    if resume_empty and normal_empty:
        return "none"

    # Timeout escape hatch (prefer the starved side).
    if normal_max_queue_wait_s > 0 and (not normal_empty) and oldest_normal_wait_s >= normal_max_queue_wait_s:
        return "normal"
    if resume_max_queue_wait_s > 0 and (not resume_empty) and oldest_resume_wait_s >= resume_max_queue_wait_s:
        return "resume"

    # Skip-on-empty.
    if normal_empty:
        return "resume"
    if resume_empty:
        return "normal"

    cycle_len = max(1, int(quota_resume_target) + int(quota_normal_target))
    cursor = int(quota_cursor) % cycle_len
    prefer_resume = cursor < max(0, int(quota_resume_target))
    return "resume" if prefer_resume else "normal"

