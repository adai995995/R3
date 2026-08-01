import json
from types import MethodType

from omegaconf import DictConfig

from roll.pipeline.agentic.agent_runner import appworld_runner
from roll.pipeline.agentic.agent_runner.appworld_runner import AppWorldRunner


class _FakeResponse:
    def __init__(self, output):
        self._output = output

    def raise_for_status(self):
        return None

    def json(self):
        return {"output": self._output}


class _FakeClient:
    posts = []
    completed_checks = 0

    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, json):
        self.posts.append((url, json))
        if url.endswith("/initialize"):
            return _FakeResponse({"instruction": "Find and report an item."})
        if url.endswith("/execute"):
            return _FakeResponse("ok")
        if url.endswith("/task_completed"):
            type(self).completed_checks += 1
            return _FakeResponse(type(self).completed_checks >= 2)
        if url.endswith("/evaluate"):
            return _FakeResponse({"success": True, "difficulty": 2})
        if url.endswith("/close"):
            return _FakeResponse(None)
        raise AssertionError(f"unexpected URL: {url}")


def test_appworld_runner_executes_multi_turn_task_on_env_specific_port(monkeypatch):
    _FakeClient.posts = []
    _FakeClient.completed_checks = 0
    monkeypatch.setattr(appworld_runner.httpx, "Client", _FakeClient)
    config = DictConfig(
        {
            "max_steps": 4,
            "http_timeout": 10,
            "config": {
                "server_host": "http://127.0.0.1",
                "server_base_port": 18200,
                "task_split": "train",
                "task_ids": ["task_a", "task_b"],
            },
        }
    )
    runner = AppWorldRunner(
        base_url="http://unused",
        env_id=3,
        env_config=config,
    )
    model_calls = 0

    def fake_request(
        self,
        client,
        messages,
        tools=None,
        tool_choice="auto",
        track_trajectory=True,
        request_role="agent",
    ):
        nonlocal model_calls
        model_calls += 1
        code = (
            "print(apis.api_docs.show_app_descriptions())"
            if model_calls == 1
            else "print(apis.supervisor.complete_task(answer='done'))"
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_{model_calls}",
                                "type": "function",
                                "function": {
                                    "name": "execute_appworld_code",
                                    "arguments": json.dumps({"code": code}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    runner._llm_request = MethodType(fake_request, runner)
    result = runner.run_job(seed=1)

    assert result.status == "Finished"
    assert result.score == 1.0
    assert result.metrics["task_id"] == "task_b"
    assert result.metrics["tool_calls"] == 2
    assert result.metrics["tool_wall_seconds"] >= 0
    assert runner.runtime_metrics["tool_calls"] == 2
    assert runner.runtime_metrics["tool_wall_seconds"] >= 0
    assert result.metrics["completed"] is True
    assert all(url.startswith("http://127.0.0.1:18203") for url, _ in _FakeClient.posts)
    initialize_payload = next(
        payload for url, payload in _FakeClient.posts if url.endswith("/initialize")
    )
    assert initialize_payload["max_api_calls_per_interaction"] == 1
