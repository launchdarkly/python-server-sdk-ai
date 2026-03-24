"""
Tests for OTel span annotation and baggage propagation.

These tests use the real opentelemetry-sdk (installed as a dev dependency)
to verify that LDAIConfigTracker correctly annotates spans and that
LDAIBaggageSpanProcessor correctly copies baggage to new spans.
"""
from unittest.mock import MagicMock, patch

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai.tracker import FeedbackKind, LDAIConfigTracker, TokenUsage

# Skip all tests in this module when opentelemetry-sdk is not installed.
pytest.importorskip("opentelemetry.sdk.trace", reason="opentelemetry-sdk not installed")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from ldai.observe import LDAIBaggageSpanProcessor, set_ai_config_baggage, detach_ai_config_baggage


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag("model-config")
        .variations(
            {
                "model": {"name": "fakeModel", "parameters": {}},
                "provider": {"name": "fakeProvider"},
                "messages": [{"role": "system", "content": "Hello!"}],
                "_ldMeta": {"enabled": True, "variationKey": "abcd", "version": 1},
            },
            "green",
        )
        .variation_for_all(0)
    )
    return td


@pytest.fixture
def ld_client(td: TestData) -> LDClient:
    config = Config("sdk-key", update_processor_class=td, send_events=False)
    client = LDClient(config=config)
    client.track = MagicMock()  # type: ignore
    return client


@pytest.fixture
def span_exporter():
    """Set up a local in-memory OTel provider and return (tracer, exporter).

    Uses a local TracerProvider rather than the global one so tests are
    isolated from each other. Spans created via start_as_current_span() are
    visible to trace.get_current_span() because OTel context propagation is
    independent of the global provider.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    yield tracer, exporter
    exporter.clear()


@pytest.fixture
def exporter_with_baggage_processor():
    """Set up a local provider with LDAIBaggageSpanProcessor and in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(LDAIBaggageSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    yield tracer, exporter
    exporter.clear()


# ---------------------------------------------------------------------------
# Tracker span annotation tests
# ---------------------------------------------------------------------------

def test_track_tokens_annotates_active_span(ld_client, span_exporter):
    tracer, exporter = span_exporter
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client, "var-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    with tracer.start_as_current_span("test-span"):
        tracker.track_tokens(TokenUsage(total=300, input=200, output=100))

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["ld.ai.metrics.tokens.total"] == 300
    assert attrs["ld.ai.metrics.tokens.input"] == 200
    assert attrs["ld.ai.metrics.tokens.output"] == 100


def test_track_duration_annotates_active_span(ld_client, span_exporter):
    tracer, exporter = span_exporter
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client, "var-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    with tracer.start_as_current_span("test-span"):
        tracker.track_duration(250)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["ld.ai.metrics.duration_ms"] == 250


def test_track_ttft_annotates_active_span(ld_client, span_exporter):
    tracer, exporter = span_exporter
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client, "var-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    with tracer.start_as_current_span("test-span"):
        tracker.track_time_to_first_token(80)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["ld.ai.metrics.time_to_first_token_ms"] == 80


def test_track_success_sets_span_status_ok(ld_client, span_exporter):
    from opentelemetry.trace import StatusCode
    tracer, exporter = span_exporter
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client, "var-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    with tracer.start_as_current_span("test-span"):
        tracker.track_success()

    spans = exporter.get_finished_spans()
    assert spans[0].status.status_code == StatusCode.OK


def test_track_error_sets_span_status_error(ld_client, span_exporter):
    from opentelemetry.trace import StatusCode
    tracer, exporter = span_exporter
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client, "var-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    with tracer.start_as_current_span("test-span"):
        tracker.track_error()

    spans = exporter.get_finished_spans()
    assert spans[0].status.status_code == StatusCode.ERROR


def test_track_feedback_annotates_active_span(ld_client, span_exporter):
    tracer, exporter = span_exporter
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client, "var-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    with tracer.start_as_current_span("test-span"):
        tracker.track_feedback({"kind": FeedbackKind.Positive})

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["ld.ai.metrics.feedback.kind"] == "positive"


def test_tracker_no_op_without_active_span(ld_client, span_exporter):
    """Tracker methods must not raise when no OTel span is active."""
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client, "var-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    # These must all succeed silently with no active span.
    tracker.track_tokens(TokenUsage(total=100, input=60, output=40))
    tracker.track_duration(100)
    tracker.track_time_to_first_token(50)
    tracker.track_success()
    tracker.track_error()
    tracker.track_feedback({"kind": FeedbackKind.Negative})

    exporter = span_exporter[1]
    assert len(exporter.get_finished_spans()) == 0


# ---------------------------------------------------------------------------
# LDAIBaggageSpanProcessor tests
# ---------------------------------------------------------------------------

def test_baggage_processor_stamps_config_key_on_child_span(exporter_with_baggage_processor):
    tracer, exporter = exporter_with_baggage_processor

    _, token = set_ai_config_baggage(
        config_key="my-config",
        variation_key="var-abc",
        model_name="gpt-4o",
        provider_name="openai",
    )
    try:
        with tracer.start_as_current_span("root-span"):
            with tracer.start_as_current_span("llm-span"):
                pass
    finally:
        detach_ai_config_baggage(token)

    spans = exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "llm-span")
    assert llm_span.attributes["ld.ai_config.key"] == "my-config"
    assert llm_span.attributes["ld.ai_config.variation_key"] == "var-abc"
    assert llm_span.attributes["ld.ai_config.model"] == "gpt-4o"
    assert llm_span.attributes["ld.ai_config.provider"] == "openai"


def test_baggage_processor_does_not_stamp_spans_outside_scope(exporter_with_baggage_processor):
    tracer, exporter = exporter_with_baggage_processor

    _, token = set_ai_config_baggage("my-config", "var-abc", "gpt-4o", "openai")
    try:
        with tracer.start_as_current_span("inside-span"):
            pass
    finally:
        detach_ai_config_baggage(token)

    # This span starts after detach; it must not carry AI Config attributes.
    with tracer.start_as_current_span("outside-span"):
        pass

    spans = exporter.get_finished_spans()
    outside = next(s for s in spans if s.name == "outside-span")
    assert "ld.ai_config.key" not in (outside.attributes or {})


def test_baggage_processor_skips_missing_model_and_provider(exporter_with_baggage_processor):
    tracer, exporter = exporter_with_baggage_processor

    _, token = set_ai_config_baggage("cfg", "v1", "", "")
    try:
        with tracer.start_as_current_span("span"):
            pass
    finally:
        detach_ai_config_baggage(token)

    spans = exporter.get_finished_spans()
    attrs = spans[0].attributes or {}
    assert attrs["ld.ai_config.key"] == "cfg"
    assert "ld.ai_config.model" not in attrs
    assert "ld.ai_config.provider" not in attrs
