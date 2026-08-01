from threading import Lock
from types import MethodType, SimpleNamespace

from omegaconf import DictConfig

from roll.pipeline.agentic.agent_runner.tau_bench_runner import TauBenchRunner
from roll.pipeline.agentic.env_manager.version_aware_proxy_env_manager import (
    VersionAwareProxyEnvManager,
)


class _Info:
    def model_dump(self):
        return {"source": "test"}


class _FakeTauEnv:
    wiki = "Follow the service policy."
    tools_info = []
    tasks = [object()]

    def reset(self, task_index):
        return SimpleNamespace(
            observation=self.user.reset("Ask the agent for help."),
            info=_Info(),
        )

    def step(self, action):
        observation = self.user.step(action.kwargs["content"])
        return SimpleNamespace(
            observation=observation,
            reward=1.0,
            done="###STOP###" in observation,
            info=_Info(),
        )


def test_tau_bench_user_tokens_are_not_added_to_policy_trajectory():
    config = DictConfig(
        {
            "max_steps": 4,
            "http_timeout": 10,
            "config": {"domain": "retail", "task_split": "train"},
        }
    )
    runner = TauBenchRunner(
        base_url="http://unused",
        env_id=3,
        env_config=config,
    )
    runner.env = _FakeTauEnv()
    calls = []
    environment_calls = 0

    def fake_request(
        self,
        client,
        messages,
        tools=None,
        tool_choice="auto",
        track_trajectory=True,
        request_role="agent",
    ):
        nonlocal environment_calls
        calls.append((request_role, track_trajectory))
        if request_role == "environment_user":
            environment_calls += 1
            content = "I need some help." if environment_calls == 1 else "###STOP###"
        else:
            content = "Your request has been handled."
        return {
            "choices": [
                {"message": {"role": "assistant", "content": content}}
            ]
        }

    runner._llm_request = MethodType(fake_request, runner)
    result = runner.run_job(seed=0)

    assert result.status == "Finished"
    assert result.score == 1.0
    assert result.metrics["completed"] is True
    assert result.metrics["tool_calls"] == 0
    assert result.metrics["tool_wall_seconds"] == 0
    assert calls == [
        ("environment_user", False),
        ("agent", True),
        ("environment_user", False),
    ]


def test_submitted_proxy_snapshot_is_not_reported_as_terminal_inflight():
    manager = VersionAwareProxyEnvManager.__new__(VersionAwareProxyEnvManager)
    manager._progress_snapshot_lock = Lock()
    manager._latest_progress_snapshot = {
        "trajectory_id": "already-submitted",
        "runtime_phase": "submitted",
    }

    assert manager.get_progress_snapshot() is None

    manager._latest_progress_snapshot = {
        "trajectory_id": "still-running",
        "runtime_phase": "policy_inference",
    }
    assert manager.get_progress_snapshot()["trajectory_id"] == "still-running"


def test_proxy_request_metrics_accumulate_router_scheduling_wait():
    manager = VersionAwareProxyEnvManager.__new__(VersionAwareProxyEnvManager)
    manager.history = [
        {
            "request_metrics": {
                "router/scheduling_wait_seconds": 0.25,
            }
        },
        {
            "request_metrics": {
                "router/scheduling_wait_seconds": 0.75,
            }
        },
    ]

    assert manager._request_metric_total(
        "router/scheduling_wait_seconds"
    ) == 1.0
