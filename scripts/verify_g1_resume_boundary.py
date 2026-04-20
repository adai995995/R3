"""
Minimal verification for G1 (strict resume boundary).

This script does NOT require running a full agentic pipeline or importing ROLL/GEM dependencies.
It validates the pure boundary logic (kept identical to TrajEnvManager):
only tool-return steps are treated as resume boundaries.

Usage:
  python scripts/verify_g1_resume_boundary.py
"""

from __future__ import annotations

from typing import Any, Optional

def is_tool_return_resume_boundary(info: Optional[dict[str, Any]], prev_tool_use_counter: Optional[int]) -> bool:
    """Keep identical logic to TrajEnvManager._is_tool_return_resume_boundary."""
    if not info:
        return False
    if info.get("use_tool") is True:
        return True
    metrics = info.get("metrics")
    if not isinstance(metrics, dict):
        return False
    raw = metrics.get("tool_use_counter")
    if raw is None:
        return False
    try:
        cur = int(raw)
    except (TypeError, ValueError):
        return False
    prev = prev_tool_use_counter if prev_tool_use_counter is not None else 0
    return cur > prev


def update_tool_use_counter_baseline(info: Optional[dict[str, Any]], prev_tool_use_counter: Optional[int]) -> Optional[int]:
    """Keep identical logic to TrajEnvManager._update_tool_use_counter_baseline."""
    if not info:
        return prev_tool_use_counter
    metrics = info.get("metrics")
    if not isinstance(metrics, dict):
        return prev_tool_use_counter
    raw = metrics.get("tool_use_counter")
    if raw is None:
        return prev_tool_use_counter
    try:
        return int(raw)
    except (TypeError, ValueError):
        return prev_tool_use_counter


def main() -> None:
    prev_tool_use_counter: Optional[int] = None

    cases = [
        ("explicit_use_tool_true", {"use_tool": True}, True),
        ("explicit_use_tool_false", {"use_tool": False}, False),
        ("no_info", None, False),
        ("no_metrics", {"metrics": {}}, False),
        ("tool_counter_first_step_0", {"metrics": {"tool_use_counter": 0}}, False),
        ("tool_counter_increase", {"metrics": {"tool_use_counter": 1}}, True),
        ("tool_counter_same", {"metrics": {"tool_use_counter": 1}}, False),
        ("tool_counter_increase_again", {"metrics": {"tool_use_counter": 2}}, True),
    ]

    passed = 0
    for name, info, expected in cases:
        got = is_tool_return_resume_boundary(info, prev_tool_use_counter=prev_tool_use_counter)
        prev_tool_use_counter = update_tool_use_counter_baseline(info, prev_tool_use_counter=prev_tool_use_counter)
        ok = got == expected
        if not ok:
            raise AssertionError(f"case={name} expected={expected} got={got} info={info}")
        passed += 1

    print(f"OK: {passed} cases passed (G1 boundary logic).")


if __name__ == "__main__":
    main()

