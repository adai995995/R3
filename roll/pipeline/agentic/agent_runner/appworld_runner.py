import json
import time
import uuid
from typing import Any, Dict, List

import httpx
from omegaconf import DictConfig

from roll.pipeline.agentic.agent_runner.base import AgentRunner, EpisodeResult
from roll.utils.logging import get_logger


APPWORLD_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_appworld_code",
        "description": (
            "Execute Python code in the persistent AppWorld shell. The shell "
            "provides `apis`; make exactly one AppWorld API call, pass every "
            "argument by keyword, and print the result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in AppWorld.",
                }
            },
            "required": ["code"],
        },
    },
}


def _system_prompt(instruction: str) -> str:
    return f"""You are an autonomous agent operating inside AppWorld.

Task:
{instruction}

Use the execute_appworld_code tool on every turn. The Python shell is stateful and
already contains `apis`.

Strict API rules:
1. Make exactly one `apis.<app>.<api>(...)` call per tool execution.
2. AppWorld accepts keyword arguments only. Positional arguments always fail.
3. Wrap the API call in `print(...)` so you can inspect its result.
4. `apis` contains application names such as `api_docs`, `supervisor`, `spotify`,
   and `file_system`; do not invent generic apps such as `search` or `account`.

Correct discovery examples:
`print(apis.api_docs.search_api_docs(query="spotify playlists", page_limit=5))`
`print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))`
Incorrect: `apis.api_docs.search_api_docs("spotify")`

Use the documentation response to copy the exact app name, API name, and keyword
parameters. When the task is complete, make this its own tool execution:
`print(apis.supervisor.complete_task(answer=<answer or None>, status="success"))`
Do not merely describe an action or answer outside the tool."""


class AppWorldRunner(AgentRunner):
    """Run one AppWorld task through an isolated HTTP environment server."""

    def __init__(
        self,
        base_url: str,
        env_id: int,
        env_config: DictConfig,
        **kwargs,
    ):
        super().__init__(base_url, env_id, env_config, **kwargs)
        self.logger = get_logger()
        server_host = str(
            self.env_params.get("server_host", "http://127.0.0.1")
        ).rstrip("/")
        server_base_port = int(self.env_params.get("server_base_port", 18200))
        self.server_url = f"{server_host}:{server_base_port + env_id}"
        self.task_split = str(self.env_params.get("task_split", "train"))
        configured_task_ids = self.env_params.get("task_ids")
        if not configured_task_ids:
            raise ValueError("AppWorldRunner requires a non-empty config.task_ids")
        self.task_ids = [str(task_id) for task_id in configured_task_ids]
        self.max_steps = int(env_config.get("max_steps", 30))
        self.http_timeout = float(env_config.get("http_timeout", 3600.0))
        self.runtime_metrics: Dict[str, float] = {}

    def _task_id(self, seed: int) -> str:
        return self.task_ids[int(seed) % len(self.task_ids)]

    @staticmethod
    def _output(response: httpx.Response) -> Any:
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return payload.get("output", payload)

    def _post(
        self,
        client: httpx.Client,
        path: str,
        payload: Dict[str, Any],
    ) -> Any:
        return self._output(client.post(f"{self.server_url}{path}", json=payload))

    @staticmethod
    def _tool_code(tool_call: Dict[str, Any]) -> str:
        arguments = tool_call.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not isinstance(
            arguments.get("code"), str
        ):
            raise ValueError("execute_appworld_code requires a string `code` argument")
        return arguments["code"]

    def run_job(self, seed: int) -> EpisodeResult:
        task_id = self._task_id(seed)
        experiment_name = (
            f"r3_env{self.env_id}_seed{seed}_{uuid.uuid4().hex[:10]}"
        )
        start_time = time.monotonic()
        tool_seconds = 0.0
        tool_calls = 0
        self.runtime_metrics = {
            "tool_calls": 0,
            "tool_wall_seconds": 0.0,
        }
        model_turns = 0
        completed = False
        initialized = False
        evaluation: Dict[str, Any] = {}
        failure = ""

        with httpx.Client(timeout=self.http_timeout) as client:
            try:
                task = self._post(
                    client,
                    "/initialize",
                    {
                        "task_id": task_id,
                        "experiment_name": experiment_name,
                        "random_seed": int(seed),
                        "max_interactions": self.max_steps,
                        "max_api_calls_per_interaction": 1,
                    },
                )
                initialized = True
                instruction = str(task["instruction"])
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": _system_prompt(instruction)},
                    {
                        "role": "user",
                        "content": (
                            "Begin solving the task. Use execute_appworld_code now."
                        ),
                    },
                ]

                for _ in range(self.max_steps):
                    response = self._llm_request(
                        client,
                        messages,
                        tools=[APPWORLD_TOOL],
                        tool_choice="required",
                        request_role="agent",
                    )
                    if "error" in response:
                        raise RuntimeError(str(response["error"]))
                    model_turns += 1

                    message = dict(response["choices"][0]["message"])
                    message.setdefault("role", "assistant")
                    calls = message.get("tool_calls") or []
                    if not calls:
                        messages.extend(
                            [
                                message,
                                {
                                    "role": "user",
                                    "content": (
                                        "Call execute_appworld_code; do not answer in text."
                                    ),
                                },
                            ]
                        )
                        continue

                    tool_call = dict(calls[0])
                    message["tool_calls"] = [tool_call]
                    messages.append(message)
                    tool_calls += 1
                    tool_started = time.monotonic()
                    try:
                        code = self._tool_code(tool_call)
                        tool_output = self._post(
                            client,
                            "/execute",
                            {"task_id": task_id, "code": code},
                        )
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        tool_output = f"Tool argument error: {exc}"
                    tool_seconds += time.monotonic() - tool_started
                    self.runtime_metrics = {
                        "tool_calls": tool_calls,
                        "tool_wall_seconds": tool_seconds,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get(
                                "id", f"appworld_{tool_calls}"
                            ),
                            "name": "execute_appworld_code",
                            "content": str(tool_output),
                        }
                    )
                    completed = bool(
                        self._post(client, "/task_completed", {"task_id": task_id})
                    )
                    if completed:
                        break

                evaluation_output = self._post(
                    client,
                    "/evaluate",
                    {"task_id": task_id, "suppress_errors": True},
                )
                if isinstance(evaluation_output, dict):
                    evaluation = evaluation_output
            except (
                httpx.HTTPError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                self.logger.error(
                    f"[AppWorldRunner] task={task_id} failed: {exc}",
                    exc_info=True,
                )
            finally:
                if initialized:
                    try:
                        self._post(client, "/close", {"task_id": task_id})
                    except (httpx.HTTPError, RuntimeError):
                        self.logger.warning(
                            f"[AppWorldRunner] failed to close task={task_id}",
                            exc_info=True,
                        )

        success = bool(evaluation.get("success", False))
        metrics = {
            "task_split": self.task_split,
            "task_id": task_id,
            "completed": completed,
            "truncated": not completed,
            "success": success,
            "tool_calls": tool_calls,
            "model_turns": model_turns,
            "tool_wall_seconds": tool_seconds,
            "runner_wall_seconds": time.monotonic() - start_time,
            "difficulty": evaluation.get("difficulty", 0),
        }
        if failure:
            return EpisodeResult(
                status="Failed",
                score=float(success),
                agent_exit_reason=failure,
                metrics=metrics,
            )
        return EpisodeResult(
            status="Finished",
            score=float(success),
            step_scores=[float(success)],
            agent_exit_reason="completed" if completed else "max_steps",
            metrics=metrics,
        )
