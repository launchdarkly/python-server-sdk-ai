"""Tests for ManagedAgentGraph and LDAIClient.create_agent_graph()."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, LDAIMetricSummary, ManagedAgentGraph, ManagedGraphResult
from ldai.providers import AgentGraphRunner, ToolRegistry
from ldai.providers.types import (
    AgentGraphRunnerResult,
    GraphMetrics,
    GraphMetricSummary,
    LDAIMetrics,
)
from ldai.tracker import TokenUsage

# --- Test doubles ---


class StubAgentGraphRunner(AgentGraphRunner):
    """Runner that returns AgentGraphRunnerResult (new shape)."""
    def __init__(self, content: str = "stub output"):
        self._content = content

    async def run(self, input) -> AgentGraphRunnerResult:
        return AgentGraphRunnerResult(
            content=self._content,
            raw={"input": input},
            metrics=GraphMetrics(success=True),
        )


class StubRunnerWithMetrics(AgentGraphRunner):
    """Runner that returns AgentGraphRunnerResult with full GraphMetrics."""
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
                node_metrics={
                    "root": LDAIMetrics(
                        success=True,
                        usage=TokenUsage(total=5, input=3, output=2),
                        duration_ms=20,
                    ),
                    "specialist": LDAIMetrics(
                        success=True,
                        usage=TokenUsage(total=5, input=2, output=3),
                        duration_ms=22,
                    ),
                },
            ),
            raw={"input": input},
        )


# --- ManagedAgentGraph unit tests ---

def _make_graph_tracker_mock(runner_result):
    """Create a mock graph tracker whose track_graph_metrics_of_async returns runner_result."""
    m = runner_result.metrics
    summary = GraphMetricSummary(
        success=m.success,
        path=list(m.path),
        duration_ms=m.duration_ms,
        usage=m.usage,
    )
    mock_tracker = MagicMock()
    mock_tracker.track_graph_metrics_of_async = AsyncMock(return_value=runner_result)
    mock_tracker.get_summary = MagicMock(return_value=summary)
    return mock_tracker


@pytest.mark.asyncio
async def test_managed_agent_graph_run_delegates_to_runner():
    """Runner result content is surfaced correctly."""
    runner = StubAgentGraphRunner("hello world")
    runner_result = await runner.run("test input")

    mock_graph = MagicMock()
    mock_graph.create_tracker = MagicMock(return_value=_make_graph_tracker_mock(runner_result))
    mock_graph.get_node = MagicMock(return_value=None)
    managed = ManagedAgentGraph(mock_graph, runner)
    result = await managed.run("test input")
    assert isinstance(result, ManagedGraphResult)
    assert result.content == "hello world"
    assert result.metrics.success is True


def test_managed_agent_graph_get_runner():
    runner = StubAgentGraphRunner()
    mock_graph = MagicMock()
    managed = ManagedAgentGraph(mock_graph, runner)
    assert managed.get_agent_graph_runner() is runner


@pytest.mark.asyncio
async def test_managed_agent_graph_run_surfaces_graph_metrics():
    """GraphMetrics fields are reflected in GraphMetricSummary."""
    runner = StubRunnerWithMetrics("final answer")
    runner_result = await runner.run("test input")

    mock_graph = MagicMock()
    mock_graph.create_tracker = MagicMock(return_value=_make_graph_tracker_mock(runner_result))
    mock_graph.get_node = MagicMock(return_value=None)

    managed = ManagedAgentGraph(mock_graph, runner)
    result = await managed.run("test input")

    assert isinstance(result, ManagedGraphResult)
    assert result.content == "final answer"
    assert result.metrics.success is True
    assert result.metrics.path == ["root", "specialist"]
    assert result.metrics.duration_ms == 42
    assert result.metrics.usage is not None
    assert result.metrics.usage.total == 10


@pytest.mark.asyncio
async def test_managed_agent_graph_drives_graph_level_tracking():
    """Managed layer delegates graph-level tracking to track_graph_metrics_of_async."""
    runner = StubRunnerWithMetrics()
    runner_result = await runner.run("test input")

    mock_graph = MagicMock()
    mock_tracker = _make_graph_tracker_mock(runner_result)
    mock_graph.create_tracker = MagicMock(return_value=mock_tracker)
    mock_graph.get_node = MagicMock(return_value=None)

    managed = ManagedAgentGraph(mock_graph, runner)
    await managed.run("test input")

    mock_tracker.track_graph_metrics_of_async.assert_called_once()


def _make_node_tracker_mock():
    """Create a mock node tracker with a real LDAIMetricSummary from get_summary."""
    tracker = MagicMock()
    tracker.get_summary = MagicMock(return_value=LDAIMetricSummary())
    return tracker


@pytest.mark.asyncio
async def test_managed_agent_graph_drives_per_node_tracking():
    """Managed layer creates per-node trackers and fires node-level events."""
    runner = StubRunnerWithMetrics()
    runner_result = await runner.run("test input")

    mock_graph = MagicMock()
    mock_graph.create_tracker = MagicMock(return_value=_make_graph_tracker_mock(runner_result))

    root_tracker = _make_node_tracker_mock()
    specialist_tracker = _make_node_tracker_mock()

    root_node = MagicMock()
    root_node.get_config.return_value.create_tracker = MagicMock(return_value=root_tracker)
    specialist_node = MagicMock()
    specialist_node.get_config.return_value.create_tracker = MagicMock(return_value=specialist_tracker)

    def get_node(key):
        return {"root": root_node, "specialist": specialist_node}.get(key)

    mock_graph.get_node = get_node

    managed = ManagedAgentGraph(mock_graph, runner)
    result = await managed.run("test input")

    # root node tracking
    root_tracker.track_tokens.assert_called_once()
    root_tracker.track_duration.assert_called_once_with(20)
    root_tracker.track_success.assert_called_once()

    # specialist node tracking
    specialist_tracker.track_tokens.assert_called_once()
    specialist_tracker.track_duration.assert_called_once_with(22)
    specialist_tracker.track_success.assert_called_once()

    # node_metrics on the summary are LDAIMetricSummary instances
    assert "root" in result.metrics.node_metrics
    assert "specialist" in result.metrics.node_metrics
    assert isinstance(result.metrics.node_metrics["root"], LDAIMetricSummary)
    assert isinstance(result.metrics.node_metrics["specialist"], LDAIMetricSummary)


@pytest.mark.asyncio
async def test_managed_agent_graph_run_succeeds_with_graph():
    """Run succeeds and returns correct content when graph is provided."""
    runner = StubRunnerWithMetrics()
    runner_result = await runner.run("test input")

    mock_graph = MagicMock()
    mock_graph.create_tracker = MagicMock(return_value=_make_graph_tracker_mock(runner_result))
    mock_graph.get_node = MagicMock(return_value=None)
    managed = ManagedAgentGraph(mock_graph, runner)
    result = await managed.run("test input")
    assert result.content == "new shape output"
    assert result.metrics.success is True


@pytest.mark.asyncio
async def test_managed_agent_graph_failure_calls_track_graph_metrics():
    """On a failed run, track_graph_metrics_of_async is still called."""

    class FailingRunner(AgentGraphRunner):
        async def run(self, input) -> AgentGraphRunnerResult:
            return AgentGraphRunnerResult(
                content='',
                raw=None,
                metrics=GraphMetrics(success=False, duration_ms=5),
            )

    failing_runner = FailingRunner()
    runner_result = await failing_runner.run("test input")

    mock_graph = MagicMock()
    mock_tracker = _make_graph_tracker_mock(runner_result)
    mock_graph.create_tracker = MagicMock(return_value=mock_tracker)
    mock_graph.get_node = MagicMock(return_value=None)

    managed = ManagedAgentGraph(mock_graph, failing_runner)
    result = await managed.run("test input")

    assert result.metrics.success is False
    mock_tracker.track_graph_metrics_of_async.assert_called_once()


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
