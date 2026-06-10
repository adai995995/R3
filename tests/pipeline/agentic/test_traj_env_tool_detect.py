"""Tool-invocation heuristics for TrajEnvManager lease suspend."""

from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager


def test_llm_response_may_invoke_tool_python_code_tags():
    assert TrajEnvManager._llm_response_may_invoke_tool(
        "<code>\nprint(1+1)\n</code>"
    )
    assert TrajEnvManager._llm_response_may_invoke_tool("```python\nprint(2)\n```")
    assert not TrajEnvManager._llm_response_may_invoke_tool("The answer is 42.")
    assert TrajEnvManager._llm_response_may_invoke_tool('<tool_call>{"name": "python"}</tool_call>')
    assert TrajEnvManager._llm_response_may_invoke_tool("<search>who founded Apple</search>")
    assert not TrajEnvManager._llm_response_may_invoke_tool("FINAL: Apple Inc.")
