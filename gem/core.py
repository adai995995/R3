from __future__ import annotations

from typing import Any, Optional


class Env:
    """Minimal Env base class for agentic environments.

    ROLL's agentic pipeline expects env objects to expose `reset()` / `step()`.
    This module exists for backward compatibility with code that imports
    `gem.core.Env` (historically from an external GEM package).
    """

    def reset(self, seed: Optional[int] = None) -> Any:  # pragma: no cover
        return None

    def step(self, action: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


__all__ = ["Env"]

