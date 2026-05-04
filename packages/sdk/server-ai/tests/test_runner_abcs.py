import pytest

from ldai.providers import AgentGraphResult, AgentGraphRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics, RunnerResult


# --- Concrete test doubles ---

class ConcreteAgentGraphRunner:
    async def run(self, input):
        return AgentGraphResult(
            output=f"graph response to: {input}",
            raw={"raw": input},
            metrics=LDAIMetrics(success=True),
        )


class MissingRunMethod:
    pass


@pytest.mark.asyncio
async def test_runner_result_fields():
    metrics = LDAIMetrics(success=True)
    result = RunnerResult(content="done", metrics=metrics, raw={"key": "val"})
    assert result.content == "done"
    assert result.raw == {"key": "val"}
    assert result.metrics is metrics


# --- AgentGraphRunner ---

def test_agent_graph_runner_structural_check_passes():
    assert isinstance(ConcreteAgentGraphRunner(), AgentGraphRunner)


def test_agent_graph_runner_structural_check_fails_when_run_missing():
    assert not isinstance(MissingRunMethod(), AgentGraphRunner)


@pytest.mark.asyncio
async def test_agent_graph_runner_run_returns_agent_graph_result():
    runner = ConcreteAgentGraphRunner()
    result = await runner.run("hello graph")
    assert isinstance(result, AgentGraphResult)
    assert result.output == "graph response to: hello graph"
    assert result.raw == {"raw": "hello graph"}
    assert result.metrics.success is True


@pytest.mark.asyncio
async def test_agent_graph_result_fields():
    metrics = LDAIMetrics(success=False)
    result = AgentGraphResult(output="", raw=None, metrics=metrics)
    assert result.output == ""
    assert result.raw is None
    assert result.metrics.success is False


# --- ToolRegistry ---

def test_tool_registry_is_dict_of_callables():
    tools: ToolRegistry = {
        "search": lambda q: f"results for {q}",
        "calculator": lambda x: x * 2,
    }
    assert tools["search"]("python") == "results for python"
    assert tools["calculator"](21) == 42


# --- Top-level exports ---

def test_top_level_exports():
    import ldai
    assert hasattr(ldai, 'AgentGraphRunner')
    assert hasattr(ldai, 'AgentGraphResult')
    assert hasattr(ldai, 'RunnerResult')
    assert hasattr(ldai, 'ToolRegistry')
