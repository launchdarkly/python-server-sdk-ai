"""Tests for the optional OpenTelemetry span wrapping in ldai._otel."""

from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

import pytest

from ldai import _otel
from ldai._otel import _CONFIG_KEY_ATTR, run_span
from ldai.evaluator import Evaluator
from ldai.managed_agent import ManagedAgent
from ldai.managed_agent_graph import ManagedAgentGraph
from ldai.models import AIAgentConfig
from ldai.providers.types import AIGraphMetrics, LDAIMetrics, RunnerResult
from ldai.tracker import LDAIConfigTracker, LDAIMetricSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _otel_available() -> bool:
    return _otel._HAS_OTEL


@pytest.fixture
def in_memory_exporter():
    """Install a fresh in-memory tracer provider and return its exporter."""
    if not _otel_available():
        pytest.skip("opentelemetry-api/sdk not installed")

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # The _otel module already cached the tracer at import time using the
    # global (proxy) provider. Reach into the module and replace it with
    # a tracer from the new provider so emitted spans are recorded by the
    # exporter we just installed.
    original_tracer = _otel._tracer
    _otel._tracer = provider.get_tracer("ld.ai")
    try:
        yield exporter
    finally:
        _otel._tracer = original_tracer
        exporter.clear()


def _make_summary(success: bool = True) -> LDAIMetricSummary:
    summary = LDAIMetricSummary()
    summary._success = success
    return summary


def _make_agent_config(
    key: str = "my-agent-config",
    tracker_result: RunnerResult = None,
) -> MagicMock:
    if tracker_result is None:
        tracker_result = RunnerResult(
            content="hello",
            raw=None,
            metrics=LDAIMetrics(success=True, tokens=None),
        )
    mock_config = MagicMock(spec=AIAgentConfig)
    mock_config.key = key
    mock_tracker = MagicMock(spec=LDAIConfigTracker)
    mock_tracker.track_metrics_of_async = AsyncMock(return_value=tracker_result)
    mock_tracker.get_summary = MagicMock(return_value=_make_summary(True))
    mock_config.create_tracker = MagicMock(return_value=mock_tracker)
    mock_config.evaluator = Evaluator.noop()
    return mock_config


# ---------------------------------------------------------------------------
# Module-level behavior
# ---------------------------------------------------------------------------


def test_attribute_constant_is_exact_string():
    """_CONFIG_KEY_ATTR must match the documented attribute name."""
    assert _CONFIG_KEY_ATTR == "ld.ai.config_key"


def test_run_span_returns_nullcontext_when_otel_missing(monkeypatch):
    """When OTel is not present, run_span must yield a no-op context."""
    monkeypatch.setattr(_otel, "_HAS_OTEL", False)
    cm = run_span("anything", "config-key")
    assert isinstance(cm, type(nullcontext()))
    with cm as result:
        assert result is None


def test_run_span_returns_real_context_when_otel_present():
    """When OTel is present, run_span should return a span-producing CM."""
    if not _otel_available():
        pytest.skip("opentelemetry-api/sdk not installed")
    cm = run_span("test.span", "ck")
    # Should not be a nullcontext when OTel is available.
    assert not isinstance(cm, type(nullcontext()))


# ---------------------------------------------------------------------------
# ManagedAgent.run wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_managed_agent_run_emits_span(in_memory_exporter):
    """ManagedAgent.run must emit a span with the agent span name + config key."""
    mock_config = _make_agent_config("agent-foo")
    mock_runner = MagicMock()

    agent = ManagedAgent(mock_config, mock_runner)
    result = await agent.run("hi")
    if result.evaluations is not None:
        await result.evaluations

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ld.ai.agent.run"
    assert span.attributes[_CONFIG_KEY_ATTR] == "agent-foo"


@pytest.mark.asyncio
async def test_managed_agent_run_returns_value_unchanged(in_memory_exporter):
    """The wrapper must not alter the run() return value."""
    mock_config = _make_agent_config(
        "agent-foo",
        tracker_result=RunnerResult(
            content="payload-text",
            metrics=LDAIMetrics(success=True, tokens=None),
            raw={"raw": True},
        ),
    )
    mock_runner = MagicMock()

    agent = ManagedAgent(mock_config, mock_runner)
    result = await agent.run("hi")
    if result.evaluations is not None:
        await result.evaluations

    assert result.content == "payload-text"
    assert result.raw == {"raw": True}


@pytest.mark.asyncio
async def test_managed_agent_run_exception_propagates(in_memory_exporter):
    """Exceptions raised inside run() must propagate through the wrapper."""
    mock_config = MagicMock(spec=AIAgentConfig)
    mock_config.key = "agent-foo"
    mock_tracker = MagicMock(spec=LDAIConfigTracker)
    mock_tracker.track_metrics_of_async = AsyncMock(side_effect=RuntimeError("boom"))
    mock_config.create_tracker = MagicMock(return_value=mock_tracker)
    mock_config.evaluator = Evaluator.noop()

    agent = ManagedAgent(mock_config, MagicMock())

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("hi")

    # The span should still have been started and ended (with error status).
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "ld.ai.agent.run"
    assert spans[0].attributes[_CONFIG_KEY_ATTR] == "agent-foo"


# ---------------------------------------------------------------------------
# ManagedAgentGraph.run wrapping
# ---------------------------------------------------------------------------


def _make_graph(graph_key: str = "graph-bar"):
    """Build a mock AgentGraphDefinition whose ._agent_graph.key == graph_key."""
    inner = MagicMock()
    inner.key = graph_key
    graph = MagicMock()
    graph._agent_graph = inner

    summary = MagicMock()
    summary.node_metrics = {}

    graph_tracker = MagicMock()
    from ldai.providers.types import AgentGraphRunnerResult
    graph_tracker.track_graph_metrics_of_async = AsyncMock(
        return_value=AgentGraphRunnerResult(
            content="graph-out",
            raw={"raw": True},
            metrics=AIGraphMetrics(success=True, node_metrics={}),
        )
    )
    graph_tracker.get_summary = MagicMock(return_value=summary)
    graph.create_tracker = MagicMock(return_value=graph_tracker)
    return graph


@pytest.mark.asyncio
async def test_managed_agent_graph_run_emits_span(in_memory_exporter):
    """ManagedAgentGraph.run must emit a span with the graph span name + config key."""
    graph = _make_graph("graph-bar")
    runner = MagicMock()

    managed = ManagedAgentGraph(graph, runner)
    result = await managed.run("hi")

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ld.ai.agent_graph.run"
    assert span.attributes[_CONFIG_KEY_ATTR] == "graph-bar"
    assert result.content == "graph-out"


@pytest.mark.asyncio
async def test_managed_agent_graph_run_exception_propagates(in_memory_exporter):
    """Exceptions raised inside run() must propagate through the wrapper."""
    graph = MagicMock()
    graph._agent_graph = MagicMock()
    graph._agent_graph.key = "graph-bar"
    graph_tracker = MagicMock()
    graph_tracker.track_graph_metrics_of_async = AsyncMock(side_effect=ValueError("nope"))
    graph.create_tracker = MagicMock(return_value=graph_tracker)

    managed = ManagedAgentGraph(graph, MagicMock())

    with pytest.raises(ValueError, match="nope"):
        await managed.run("hi")

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "ld.ai.agent_graph.run"


# ---------------------------------------------------------------------------
# No-op path (OTel simulated as absent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_managed_agent_run_works_without_otel(monkeypatch):
    """ManagedAgent.run must work cleanly when OTel is unavailable."""
    monkeypatch.setattr(_otel, "_HAS_OTEL", False)

    mock_config = _make_agent_config(
        "agent-foo",
        tracker_result=RunnerResult(
            content="no-otel",
            metrics=LDAIMetrics(success=True, tokens=None),
            raw=None,
        ),
    )
    mock_runner = MagicMock()

    agent = ManagedAgent(mock_config, mock_runner)
    result = await agent.run("hi")
    if result.evaluations is not None:
        await result.evaluations

    assert result.content == "no-otel"
