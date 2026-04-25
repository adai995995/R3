from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _normalize_answer(x: str) -> str:
    return " ".join(str(x).strip().split())


@dataclass
class MathEnv:
    """Compatibility shim for `gem.envs.math_env.MathEnv`.

    The ROLL wrapper (`roll.pipeline.agentic.env.gem.math_env.MathEnv`) inherits this
    class but overrides most runtime behaviors. The one method it relies on is
    `check_correct(model_answer, gold_answer)`.
    """

    extract_boxed: bool = True

    def check_correct(self, model_answer: Optional[str], gold_answer: Optional[str]) -> bool:
        if model_answer is None or gold_answer is None:
            return False

        # Try `math_verify` if available; fall back to normalized string match.
        try:
            from math_verify import verify

            # `verify` is tolerant to common LaTeX formatting differences.
            return bool(verify(_normalize_answer(model_answer), _normalize_answer(gold_answer)))
        except Exception:
            return _normalize_answer(model_answer) == _normalize_answer(gold_answer)


__all__ = ["MathEnv"]

