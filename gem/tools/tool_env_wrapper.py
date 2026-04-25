from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple


@dataclass
class ToolEnvWrapper:
    """A lightweight tool wrapper compatible with ROLL's agentic pipeline.

    ROLL configs may pass wrapper args like `tool_reward`, `tool_success_reward`,
    `max_tool_uses`, etc. Older repos sometimes relied on an external GEM package.
    In this repo we provide a small in-tree implementation that:
    - accepts the wrapper args (and ignores unknown ones)
    - forwards `reset/step` to the wrapped env
    - exposes `tool_use_counter` / `tool_success_counter` in info for metrics

    Note: This implementation is intentionally minimal. The tool execution logic
    (turning a tool-call action into an intermediate observation) is handled in
    ROLL's higher-level wrappers/tools when available.
    """

    env: Any
    tools: List[Any] = field(default_factory=list)

    tool_reward: float = 0.0
    tool_success_reward: float = 0.0
    max_tool_uses: int = 0

    # internal counters
    tool_use_counter: int = 0
    tool_success_counter: int = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        if hasattr(self.env, "reset"):
            observation, info = self.env.reset(seed=seed)
        else:
            observation, info = None, {}
        if info is None:
            info = {}
        info.setdefault("tool_use_counter", self.tool_use_counter)
        info.setdefault("tool_success_counter", self.tool_success_counter)
        return observation, info

    def step(
        self, action: str, verbose: bool = False
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        # Minimal passthrough: rely on env to compute reward/termination.
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            info = {}
        info.setdefault("tool_use_counter", self.tool_use_counter)
        info.setdefault("tool_success_counter", self.tool_success_counter)
        return observation, reward, terminated, truncated, info

