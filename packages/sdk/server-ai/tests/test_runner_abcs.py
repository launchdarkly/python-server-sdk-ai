import pytest

from ldai.providers.types import LDAIMetrics
from ldai.runners.agent_graph_runner import AgentGraphRunner
from ldai.runners.agent_runner import AgentRunner
from ldai.runners.types import AgentGraphResult, AgentResult, ToolRegistry


# --- Concrete test doubles ---

class ConcreteAgentRunner(AgentRunner):
    async def run(self, input):
        return AgentResult(
            output=f"agent response to: {input}",
            raw={"raw": input},
            metrics=LDAIMetrics(success=True),
        )


class ConcreteAgentGraphRunner(AgentGraphRunner):
    async def run(self, input):
        return AgentGraphResult(
            output=f"graph response to: {input}",
            raw={"raw": input},
            metrics=LDAIMetrics(success=True),
        )


# --- AgentRunner ---

def test_agent_runner_is_abstract():
    with pytest.raises(TypeError):
        AgentRunner()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_agent_runner_run_returns_agent_result():
    runner = ConcreteAgentRunner()
    result = await runner.run("hello")
    assert isinstance(result, AgentResult)
    assert result.output == "agent response to: hello"
    assert result.raw == {"raw": "hello"}
    assert result.metrics.success is True


@pytest.mark.asyncio
async def test_agent_result_fields():
    metrics = LDAIMetrics(success=True)
    result = AgentResult(output="done", raw={"key": "val"}, metrics=metrics)
    assert result.output == "done"
    assert result.raw == {"key": "val"}
    assert result.metrics is metrics


# --- AgentGraphRunner ---

def test_agent_graph_runner_is_abstract():
    with pytest.raises(TypeError):
        AgentGraphRunner()  # type: ignore[abstract]


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
    assert hasattr(ldai, 'AgentRunner')
    assert hasattr(ldai, 'AgentGraphRunner')
    assert hasattr(ldai, 'AgentResult')
    assert hasattr(ldai, 'AgentGraphResult')
    assert hasattr(ldai, 'ToolRegistry')
