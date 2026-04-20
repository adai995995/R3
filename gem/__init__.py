from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


_REGISTRY: Dict[str, str] = {}


class Env:
    """Minimal Env base class for ROLL agentic envs.

    This is a lightweight shim to satisfy imports in `roll/pipeline/agentic/env/*`.
    """

    def reset(self, seed: Optional[int] = None) -> Any:  # pragma: no cover
        return None

    def step(self, action: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def register(env_id: str, entry_point: str) -> None:
    """Register an environment entry point.

    Args:
        env_id: Environment id used by `make`.
        entry_point: "module.submodule:ClassName"
    """

    _REGISTRY[env_id] = entry_point


def _load_entry_point(entry_point: str) -> Callable[..., Any]:
    module_path, _, attr = entry_point.partition(":")
    if not module_path or not attr:
        raise ValueError(f"Invalid entry_point: {entry_point}")
    module = importlib.import_module(module_path)
    cls = getattr(module, attr)
    return cls


def make(env_id: str, **kwargs: Any) -> Any:
    """Instantiate a registered env."""

    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown env_id={env_id!r}. Registered: {sorted(_REGISTRY.keys())}")
    cls = _load_entry_point(_REGISTRY[env_id])
    return cls(**kwargs)


__all__ = [
    "Env",
    "make",
    "register",
]

