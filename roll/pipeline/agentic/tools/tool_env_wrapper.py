from typing import Dict, List, Optional, Tuple, Any, SupportsFloat

from gem import Env
from gem.tools.tool_env_wrapper import ToolEnvWrapper as GEMToolEnvWrapper

from roll.pipeline.agentic.tools.registration import make_tool

class ToolEnvWrapper(GEMToolEnvWrapper):
    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        self._last_tool_success_counter = 0
        observation, info = super().reset(seed=seed)
        metrics = {
            "tool_use_counter": info.pop("tool_use_counter"),
            "tool_success_counter": info.pop("tool_success_counter"),
            "tool_reward": 0.0,
            "tool_success_reward": 0.0,
        }
        metrics_agg_mode = {
            "tool_use_counter": "last",
            "tool_success_counter": "last",
            "tool_reward": "sum",
            "tool_success_reward": "sum",
        }
        metrics.update(info.pop("metrics", {}))
        metrics_agg_mode.update(info.pop("metrics_agg_mode", {}))
        info["metrics"] = metrics
        info["metrics_agg_mode"] = metrics_agg_mode
        return observation, info

    def step(
            self,
            action: str,
            verbose: bool = False,
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action, verbose)
        tool_use_counter = info.pop("tool_use_counter")
        tool_success_counter = info.pop("tool_success_counter")
        is_tool_step = bool(info.get("use_tool", False))
        last_tool_success_counter = getattr(self, "_last_tool_success_counter", 0)
        is_tool_success = is_tool_step and tool_success_counter > last_tool_success_counter
        self._last_tool_success_counter = tool_success_counter

        metrics = {
            "tool_use_counter": tool_use_counter,
            "tool_success_counter": tool_success_counter,
            "tool_reward": float(reward) if is_tool_step and not is_tool_success else 0.0,
            "tool_success_reward": float(reward) if is_tool_success else 0.0,
        }
        metrics_agg_mode = {
            "tool_use_counter": "last",
            "tool_success_counter": "last",
            "tool_reward": "sum",
            "tool_success_reward": "sum",
        }
        metrics.update(info.pop("metrics", {}))
        metrics_agg_mode.update(info.pop("metrics_agg_mode", {}))
        info["metrics"] = metrics
        info["metrics_agg_mode"] = metrics_agg_mode
        return observation, reward, terminated, truncated, info


def tool_wrapper(env: Env, wrapper_args: Dict, tool_configs: List[Dict]):
    tools = []

    for tool_config in tool_configs:
        tool_id = tool_config["tool_id"]
        tool_args = tool_config["tool_args"]
        tools.append(make_tool(tool_id=tool_id, **tool_args))

    tool_env_wrapper = ToolEnvWrapper(env, tools, **wrapper_args)
    return tool_env_wrapper

