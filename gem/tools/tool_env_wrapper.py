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
        self.tool_use_counter = 0
        self.tool_success_counter = 0
        if hasattr(self.env, "reset"):
            observation, info = self.env.reset(seed=seed)
        else:
            observation, info = None, {}
        if info is None:
            info = {}
        tool_instructions = [
            tool.instruction_string()
            for tool in self.tools
            if hasattr(tool, "instruction_string")
        ]
        if tool_instructions:
            info["env_instruction"] = "\n".join(
                [info.get("env_instruction", ""), *tool_instructions]
            ).strip()
        info.setdefault("tool_use_counter", self.tool_use_counter)
        info.setdefault("tool_success_counter", self.tool_success_counter)
        return observation, info

    def step(
        self, action: str, verbose: bool = False
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        for tool in self.tools:
            if not hasattr(tool, "execute_action"):
                continue
            is_valid, has_error, observation, parsed_action = tool.execute_action(action)
            if not is_valid:
                continue

            self.tool_use_counter += 1
            if not has_error:
                self.tool_success_counter += 1

            reward = self.tool_success_reward if not has_error else self.tool_reward
            if self.max_tool_uses and self.tool_use_counter > self.max_tool_uses:
                observation = "Maximum tool uses reached. Please provide the final answer.\n"
                reward = self.tool_reward

            info = {
                "use_tool": True,
                "parsed_action": parsed_action,
                "tool_name": getattr(tool, "name", tool.__class__.__name__),
                "tool_use_counter": self.tool_use_counter,
                "tool_success_counter": self.tool_success_counter,
            }
            return observation, reward, False, False, info

        # No tool call was parsed; rely on env to compute final reward/termination.
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            info = {}
        info.setdefault("tool_use_counter", self.tool_use_counter)
        info.setdefault("tool_success_counter", self.tool_success_counter)
        return observation, reward, terminated, truncated, info

