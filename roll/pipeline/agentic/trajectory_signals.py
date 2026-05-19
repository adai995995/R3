"""Runtime-side trajectory penalty signals for value-based scheduling."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def _is_invalid_step(entry: Dict[str, Any]) -> bool:
    if entry.get("invalid") is True:
        return True
    if entry.get("is_valid_action") is False:
        return True
    if entry.get("parse_error") is True:
        return True
    err = entry.get("error")
    if isinstance(err, str) and err.strip():
        return True
    return False


def _response_fingerprint(entry: Dict[str, Any]) -> Optional[str]:
    text = entry.get("llm_response")
    if not isinstance(text, str):
        return None
    normalized = text.strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def compute_trajectory_signals(
    *,
    history: List[Dict[str, Any]],
    step: int,
    max_steps: int,
    terminated: bool,
    truncated: bool,
    loop_window: int = 3,
    stall_reward_window: int = 4,
) -> Dict[str, float]:
    """
    Derive scheduling signals from rollout history (no env reward as positive boost).

    Returns floats in {0.0, 1.0} for penalty indicators plus remaining_steps_ratio.
    """
    completed = [h for h in history if h.get("llm_response") is not None]
    invalid = 1.0 if any(_is_invalid_step(h) for h in completed) else 0.0

    fingerprints = [_response_fingerprint(h) for h in completed[-loop_window:]]
    fingerprints = [fp for fp in fingerprints if fp is not None]
    loop = 0.0
    if len(fingerprints) >= loop_window and len(set(fingerprints)) == 1:
        loop = 1.0

    stall = 0.0
    if len(completed) >= stall_reward_window:
        tail = completed[-stall_reward_window:]
        rewards = [float(h.get("reward", 0.0) or 0.0) for h in tail]
        if all(r == 0.0 for r in rewards) and len(tail) >= stall_reward_window:
            stall = 1.0

    term = 1.0 if (terminated or truncated) else 0.0
    remaining = max(0, int(max_steps) - int(step))
    max_s = max(1, int(max_steps))
    remaining_ratio = float(remaining) / float(max_s)

    return {
        "trajectory_invalid": invalid,
        "trajectory_loop": loop,
        "trajectory_stall": stall,
        "trajectory_terminated": term,
        "remaining_steps": float(remaining),
        "max_steps": float(max_s),
        "remaining_steps_ratio": remaining_ratio,
    }
