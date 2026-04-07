"""Tests for ManagedAgentGraph and LDAIClient.create_agent_graph()."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, ManagedAgentGraph
from ldai.providers.types import LDAIMetrics
from ldai.providers import AgentGraphResult, AgentGraphRunner, ToolRegistry
from ldai.tracker import AIGraphTracker


# ---------------------------------------------------------------------------
# OTel patch fixture (mirrors test_observe.py)
# ---------------------------------------------------------------------------

def _make_span(recording: bool = True) -> MagicMock:
    span = MagicMock()
    span.is_recording.return_value = recording
    return span


@pytest.fixture()
def patch_otel(monkeypatch):
    span = _make_span()
    trace_mod = MagicMock()
    trace_mod.get_current_span.return_value = span
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=span)
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    trace_mod.get_tracer.return_value = tracer

    baggage_mod = MagicMock()
    baggage_mod.set_baggage.side_effect = lambda key, val, context=None: context or {}
    baggage_mod.get_baggage.return_value = None

    context_mod = MagicMock()
    context_mod.get_current.return_value = {}
    context_mod.attach.return_value = object()
    context_mod.detach = MagicMock()

    import ldai.observe as obs
    monkeypatch.setattr(obs, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(obs, "_otel_trace", trace_mod)
    monkeypatch.setattr(obs, "_otel_baggage", baggage_mod)
    monkeypatch.setattr(obs, "_otel_context", context_mod)

    yield {"span": span, "trace": trace_mod}


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


@pytest.mark.asyncio
async def test_managed_agent_graph_run_creates_span_and_annotates(patch_otel):
    """run() creates an ld.ai.agent_graph span and annotates it with tracker metadata."""
    span = patch_otel["span"]
    trace_mod = patch_otel["trace"]

    tracker = MagicMock()
    tracker._graph_key = "my-graph"
    tracker._variation_key = "gvar-1"
    tracker._version = 2
    tracker._context.key = "user-456"

    runner = StubAgentGraphRunner("result")
    managed = ManagedAgentGraph(runner, tracker)
    await managed.run("input")

    trace_mod.get_tracer.assert_called_with("launchdarkly.ai")
    tracer = trace_mod.get_tracer.return_value
    tracer.start_as_current_span.assert_called_with("ld_ai.agent_graph", context=None)

    span.set_attribute.assert_any_call("ld_ai.ai_config.key", "my-graph")
    span.set_attribute.assert_any_call("ld_ai.ai_config.variation_key", "gvar-1")
    span.set_attribute.assert_any_call("ld_ai.ai_config.version", 2)
    span.set_attribute.assert_any_call("ld_ai.ai_config.context_key", "user-456")


@pytest.mark.asyncio
async def test_managed_agent_graph_run_no_annotation_without_tracker(patch_otel):
    """run() creates span but skips annotation when no tracker is present."""
    span = patch_otel["span"]

    runner = StubAgentGraphRunner("result")
    managed = ManagedAgentGraph(runner)
    await managed.run("input")

    attr_keys = [call.args[0] for call in span.set_attribute.call_args_list]
    assert "ld_ai.ai_config.key" not in attr_keys


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
