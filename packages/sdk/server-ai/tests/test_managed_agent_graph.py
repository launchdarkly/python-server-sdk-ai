"""Tests for ManagedAgentGraph and LDAIClient.create_agent_graph()."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, ManagedAgentGraph
from ldai.providers.types import LDAIMetrics
from ldai.providers import AgentGraphResult, AgentGraphRunner, ToolRegistry
from ldai.tracker import AIGraphTracker


# --- Test double ---

class StubAgentGraphRunner(AgentGraphRunner):
    def __init__(self, output: str = "stub output"):
        self._output = output

    async def run(self, input) -> AgentGraphResult:
        return AgentGraphResult(
            output=self._output,
            raw={"input": input},
            metrics=LDAIMetrics(success=True),
        )


# --- ManagedAgentGraph unit tests ---

@pytest.mark.asyncio
async def test_managed_agent_graph_run_delegates_to_runner():
    runner = StubAgentGraphRunner("hello world")
    managed = ManagedAgentGraph(runner)
    result = await managed.run("test input")
    assert result.output == "hello world"
    assert result.metrics.success is True


def test_managed_agent_graph_get_runner():
    runner = StubAgentGraphRunner()
    managed = ManagedAgentGraph(runner)
    assert managed.get_agent_graph_runner() is runner


def test_managed_agent_graph_get_tracker_none_by_default():
    runner = StubAgentGraphRunner()
    managed = ManagedAgentGraph(runner)
    assert managed.get_tracker() is None


def test_managed_agent_graph_get_tracker_returns_tracker():
    runner = StubAgentGraphRunner()
    tracker = MagicMock(spec=AIGraphTracker)
    managed = ManagedAgentGraph(runner, tracker)
    assert managed.get_tracker() is tracker


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
    assert result.output == "final answer"
    assert result.metrics.success is True


# --- Top-level export ---

def test_managed_agent_graph_exported():
    import ldai
    assert hasattr(ldai, 'ManagedAgentGraph')
