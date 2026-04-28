"""Tests for ManagedAgentGraph and LDAIClient.create_agent_graph()."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, ManagedAgentGraph, ManagedGraphResult
from ldai.providers.types import AgentGraphRunnerResult, GraphMetrics, LDAIMetrics
from ldai.providers import AgentGraphResult, AgentGraphRunner, ToolRegistry
from ldai.tracker import TokenUsage


# --- Test doubles ---

class StubAgentGraphRunner(AgentGraphRunner):
    """Legacy runner that returns AgentGraphResult (old shape)."""
    def __init__(self, output: str = "stub output"):
        self._output = output

    async def run(self, input) -> AgentGraphResult:
        return AgentGraphResult(
            output=self._output,
            raw={"input": input},
            metrics=LDAIMetrics(success=True),
        )


class StubNewShapeRunner(AgentGraphRunner):
    """New-shape runner that returns AgentGraphRunnerResult."""
    def __init__(self, content: str = "new shape output"):
        self._content = content

    async def run(self, input) -> AgentGraphRunnerResult:
        return AgentGraphRunnerResult(
            content=self._content,
            metrics=GraphMetrics(
                success=True,
                path=["root", "specialist"],
                duration_ms=42,
                usage=TokenUsage(total=10, input=5, output=5),
                node_metrics={},
            ),
            raw={"input": input},
        )


# --- ManagedAgentGraph unit tests (legacy shape) ---

@pytest.mark.asyncio
async def test_managed_agent_graph_run_delegates_to_runner():
    """Legacy AgentGraphResult shape: content comes from output field."""
    runner = StubAgentGraphRunner("hello world")
    managed = ManagedAgentGraph(runner)
    result = await managed.run("test input")
    assert isinstance(result, ManagedGraphResult)
    assert result.content == "hello world"
    assert result.metrics.success is True


def test_managed_agent_graph_get_runner():
    runner = StubAgentGraphRunner()
    managed = ManagedAgentGraph(runner)
    assert managed.get_agent_graph_runner() is runner


# --- ManagedAgentGraph unit tests (new AgentGraphRunnerResult shape) ---

@pytest.mark.asyncio
async def test_managed_agent_graph_run_handles_new_shape():
    """New AgentGraphRunnerResult shape: content and GraphMetrics are surfaced."""
    runner = StubNewShapeRunner("final answer")
    mock_graph = MagicMock()
    mock_tracker = MagicMock()
    mock_graph.create_tracker = MagicMock(return_value=mock_tracker)

    managed = ManagedAgentGraph(runner, graph=mock_graph)
    result = await managed.run("test input")

    assert isinstance(result, ManagedGraphResult)
    assert result.content == "final answer"
    assert result.metrics.success is True
    assert result.metrics.path == ["root", "specialist"]
    assert result.metrics.duration_ms == 42
    assert result.metrics.usage is not None
    assert result.metrics.usage.total == 10


@pytest.mark.asyncio
async def test_managed_agent_graph_new_shape_drives_tracking():
    """New shape: managed layer calls tracker methods from result.metrics."""
    runner = StubNewShapeRunner()
    mock_graph = MagicMock()
    mock_tracker = MagicMock()
    mock_graph.create_tracker = MagicMock(return_value=mock_tracker)

    managed = ManagedAgentGraph(runner, graph=mock_graph)
    await managed.run("test input")

    mock_tracker.track_path.assert_called_once_with(["root", "specialist"])
    mock_tracker.track_duration.assert_called_once_with(42)
    mock_tracker.track_invocation_success.assert_called_once()
    mock_tracker.track_total_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_managed_agent_graph_new_shape_no_graph_skips_tracking():
    """New shape without graph: no tracking called (graph not available)."""
    runner = StubNewShapeRunner()
    managed = ManagedAgentGraph(runner, graph=None)
    # Should not raise even without a graph reference
    result = await managed.run("test input")
    assert result.content == "new shape output"
    assert result.metrics.success is True




# --- LDAIClient.create_agent_graph() integration tests ---

@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()

    td.update(
        td.flag('travel-graph')
        .variations({
            'root': 'triage-agent',
            'edges': {
                'triage-agent': [{'key': 'specialist-agent'}],
            },
            '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
        })
        .variation_for_all(0)
    )

    td.update(
        td.flag('triage-agent')
        .variations({
            'model': {'name': 'gpt-4'},
            'provider': {'name': 'openai'},
            'instructions': 'You are a triage agent.',
            '_ldMeta': {'enabled': True, 'variationKey': 'triage-v1', 'version': 1},
        })
        .variation_for_all(0)
    )

    td.update(
        td.flag('specialist-agent')
        .variations({
            'model': {'name': 'gpt-4'},
            'provider': {'name': 'openai'},
            'instructions': 'You are a specialist.',
            '_ldMeta': {'enabled': True, 'variationKey': 'specialist-v1', 'version': 1},
        })
        .variation_for_all(0)
    )

    td.update(
        td.flag('disabled-graph')
        .variations({
            '_ldMeta': {'enabled': False, 'variationKey': 'disabled-v1', 'version': 1},
        })
        .variation_for_all(0)
    )

    return td


@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config('sdk-key', update_processor_class=td, send_events=False)
    return LDClient(config=config)


@pytest.fixture
def ldai_client(client: LDClient) -> LDAIClient:
    return LDAIClient(client)


@pytest.mark.asyncio
async def test_create_agent_graph_returns_managed_agent_graph(ldai_client: LDAIClient):
    context = Context.create('user-key')
    stub_runner = StubAgentGraphRunner("result")

    with patch(
        'ldai.providers.runner_factory.RunnerFactory.create_agent_graph',
        new=MagicMock(return_value=stub_runner),
    ):
        managed = await ldai_client.create_agent_graph('travel-graph', context)

    assert managed is not None
    assert isinstance(managed, ManagedAgentGraph)
    assert managed.get_agent_graph_runner() is stub_runner


@pytest.mark.asyncio
async def test_create_agent_graph_returns_none_when_disabled(ldai_client: LDAIClient):
    context = Context.create('user-key')
    managed = await ldai_client.create_agent_graph('disabled-graph', context)
    assert managed is None


@pytest.mark.asyncio
async def test_create_agent_graph_returns_none_when_runner_factory_fails(ldai_client: LDAIClient):
    context = Context.create('user-key')

    with patch(
        'ldai.providers.runner_factory.RunnerFactory.create_agent_graph',
        new=MagicMock(return_value=None),
    ):
        managed = await ldai_client.create_agent_graph('travel-graph', context)

    assert managed is None


@pytest.mark.asyncio
async def test_create_agent_graph_passes_tools_to_factory(ldai_client: LDAIClient):
    context = Context.create('user-key')
    tools: ToolRegistry = {'search': lambda q: f'results for {q}'}
    captured = {}

    def fake_create_agent_graph(graph_def, tools_arg, default_ai_provider=None):
        captured['tools'] = tools_arg
        return StubAgentGraphRunner()

    with patch(
        'ldai.providers.runner_factory.RunnerFactory.create_agent_graph',
        new=fake_create_agent_graph,
    ):
        await ldai_client.create_agent_graph('travel-graph', context, tools=tools)

    assert captured['tools'] is tools


@pytest.mark.asyncio
async def test_create_agent_graph_run_produces_result(ldai_client: LDAIClient):
    context = Context.create('user-key')

    with patch(
        'ldai.providers.runner_factory.RunnerFactory.create_agent_graph',
        new=MagicMock(return_value=StubAgentGraphRunner("final answer")),
    ):
        managed = await ldai_client.create_agent_graph('travel-graph', context)

    assert managed is not None
    result = await managed.run("find restaurants")
    assert isinstance(result, ManagedGraphResult)
    assert result.content == "final answer"
    assert result.metrics.success is True


# --- Top-level export ---

def test_managed_agent_graph_exported():
    import ldai
    assert hasattr(ldai, 'ManagedAgentGraph')
