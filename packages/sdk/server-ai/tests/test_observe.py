"""Tests for ldai.observe — OTel span annotation helpers."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build lightweight mock spans / tracers
# ---------------------------------------------------------------------------

def _make_span(recording: bool = True) -> MagicMock:
    span = MagicMock()
    span.is_recording.return_value = recording
    return span


def _make_tracer(span: MagicMock) -> MagicMock:
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=span)
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    return tracer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_otel(monkeypatch):
    """
    Patch the opentelemetry imports so tests run without the real SDK.
    Yields a namespace with 'span', 'trace', 'baggage', 'context' mocks.
    """
    span = _make_span(recording=True)

    trace_mod = MagicMock()
    trace_mod.get_current_span.return_value = span
    trace_mod.get_tracer.return_value = _make_tracer(span)
    trace_mod.StatusCode = MagicMock()
    trace_mod.StatusCode.OK = "OK"
    trace_mod.StatusCode.ERROR = "ERROR"

    baggage_mod = MagicMock()
    baggage_mod.set_baggage.side_effect = lambda key, val, context=None: context or {}
    baggage_mod.get_baggage.return_value = None

    context_mod = MagicMock()
    context_mod.get_current.return_value = {}
    context_mod.attach.return_value = object()
    context_mod.detach = MagicMock()

    # Patch the module-level attributes in observe.py
    import ldai.observe as obs
    monkeypatch.setattr(obs, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(obs, "_otel_trace", trace_mod)
    monkeypatch.setattr(obs, "_otel_baggage", baggage_mod)
    monkeypatch.setattr(obs, "_otel_context", context_mod)
    monkeypatch.setattr(obs, "StatusCode", trace_mod.StatusCode)

    yield {"span": span, "trace": trace_mod, "baggage": baggage_mod, "context": context_mod}


# ---------------------------------------------------------------------------
# annotate_span_with_ai_config_metadata
# ---------------------------------------------------------------------------

def test_annotate_ai_config_metadata_sets_attributes(patch_otel):
    from ldai.observe import annotate_span_with_ai_config_metadata
    span = patch_otel["span"]

    annotate_span_with_ai_config_metadata(
        config_key="my-config",
        variation_key="var-1",
        model_name="gpt-4",
        provider_name="openai",
        version=3,
        context_key="user-123",
    )

    span.set_attribute.assert_any_call("ld.ai_config.key", "my-config")
    span.set_attribute.assert_any_call("ld.ai_config.variation_key", "var-1")
    span.set_attribute.assert_any_call("ld.ai_config.version", 3)
    span.set_attribute.assert_any_call("ld.ai_config.context_key", "user-123")
    span.set_attribute.assert_any_call("ld.ai_config.model", "gpt-4")
    span.set_attribute.assert_any_call("ld.ai_config.provider", "openai")


def test_annotate_ai_config_metadata_omits_empty_model_provider(patch_otel):
    from ldai.observe import annotate_span_with_ai_config_metadata
    span = patch_otel["span"]

    annotate_span_with_ai_config_metadata("k", "v", "", "")

    attr_keys = [call.args[0] for call in span.set_attribute.call_args_list]
    assert "ld.ai_config.model" not in attr_keys
    assert "ld.ai_config.provider" not in attr_keys


def test_annotate_ai_config_no_op_when_no_recording_span(patch_otel):
    from ldai.observe import annotate_span_with_ai_config_metadata
    span = patch_otel["span"]
    span.is_recording.return_value = False

    annotate_span_with_ai_config_metadata("k", "v", "m", "p")

    span.set_attribute.assert_not_called()


# ---------------------------------------------------------------------------
# annotate_span_with_tokens
# ---------------------------------------------------------------------------

def test_annotate_tokens(patch_otel):
    from ldai.observe import annotate_span_with_tokens
    span = patch_otel["span"]

    annotate_span_with_tokens(100, 60, 40)

    span.set_attribute.assert_any_call("ld.ai.metrics.tokens.total", 100)
    span.set_attribute.assert_any_call("ld.ai.metrics.tokens.input", 60)
    span.set_attribute.assert_any_call("ld.ai.metrics.tokens.output", 40)


# ---------------------------------------------------------------------------
# annotate_span_with_duration / ttft
# ---------------------------------------------------------------------------

def test_annotate_duration(patch_otel):
    from ldai.observe import annotate_span_with_duration
    span = patch_otel["span"]
    annotate_span_with_duration(250)
    span.set_attribute.assert_called_once_with("ld.ai.metrics.duration_ms", 250)


def test_annotate_ttft(patch_otel):
    from ldai.observe import annotate_span_with_ttft
    span = patch_otel["span"]
    annotate_span_with_ttft(80)
    span.set_attribute.assert_called_once_with("ld.ai.metrics.time_to_first_token_ms", 80)


# ---------------------------------------------------------------------------
# annotate_span_success
# ---------------------------------------------------------------------------

def test_annotate_span_success_ok(patch_otel):
    from ldai.observe import annotate_span_success
    span = patch_otel["span"]
    trace_mod = patch_otel["trace"]
    annotate_span_success(True)
    span.set_status.assert_called_once_with(trace_mod.StatusCode.OK)


def test_annotate_span_success_error(patch_otel):
    from ldai.observe import annotate_span_success
    span = patch_otel["span"]
    trace_mod = patch_otel["trace"]
    annotate_span_success(False)
    span.set_status.assert_called_once_with(trace_mod.StatusCode.ERROR)


# ---------------------------------------------------------------------------
# annotate_span_with_feedback
# ---------------------------------------------------------------------------

def test_annotate_feedback(patch_otel):
    from ldai.observe import annotate_span_with_feedback
    span = patch_otel["span"]
    annotate_span_with_feedback("positive")
    span.set_attribute.assert_called_once_with("ld.ai.metrics.feedback.kind", "positive")


# ---------------------------------------------------------------------------
# annotate_span_with_judge_response
# ---------------------------------------------------------------------------

def test_annotate_judge_response(patch_otel):
    from ldai.observe import annotate_span_with_judge_response
    from ldai.providers.types import EvalScore, JudgeResponse

    span = patch_otel["span"]

    response = JudgeResponse(
        evals={"relevance": EvalScore(score=0.9, reasoning="looks good")},
        success=True,
        judge_config_key="my-judge",
    )
    annotate_span_with_judge_response(response)

    span.set_attribute.assert_any_call("ld.ai.judge.config_key", "my-judge")
    span.set_attribute.assert_any_call("ld.ai.judge.success", True)
    span.set_attribute.assert_any_call("ld.ai.judge.relevance.score", 0.9)
    span.set_attribute.assert_any_call("ld.ai.judge.relevance.reasoning", "looks good")


def test_annotate_judge_response_error(patch_otel):
    from ldai.observe import annotate_span_with_judge_response
    from ldai.providers.types import JudgeResponse

    span = patch_otel["span"]

    response = JudgeResponse(evals={}, success=False, error="timed out")
    annotate_span_with_judge_response(response)

    span.set_attribute.assert_any_call("ld.ai.judge.success", False)
    span.set_attribute.assert_any_call("ld.ai.judge.error", "timed out")


# ---------------------------------------------------------------------------
# _span_scope
# ---------------------------------------------------------------------------

def test_span_scope_reuses_existing_span(patch_otel):
    """If a recording span is already active, yield it without creating a new one."""
    from ldai.observe import _span_scope
    trace_mod = patch_otel["trace"]

    with _span_scope("ld.ai.test") as s:
        assert s is patch_otel["span"]

    trace_mod.get_tracer.assert_not_called()


def test_span_scope_creates_span_when_none(patch_otel):
    """When no span is active and create_if_none=True, a new internal span is created."""
    from ldai.observe import _span_scope
    trace_mod = patch_otel["trace"]

    # Simulate no active span
    trace_mod.get_current_span.return_value = None

    with _span_scope("ld.ai.test", create_if_none=True):
        pass

    trace_mod.get_tracer.assert_called_once_with("launchdarkly.ai")
    tracer = trace_mod.get_tracer.return_value
    tracer.start_as_current_span.assert_called_once_with("ld.ai.test")


def test_span_scope_yields_none_when_create_if_none_false(patch_otel):
    """When no span is active and create_if_none=False, yield None."""
    from ldai.observe import _span_scope
    trace_mod = patch_otel["trace"]

    trace_mod.get_current_span.return_value = None

    with _span_scope("ld.ai.test", create_if_none=False) as s:
        assert s is None


def test_span_scope_yields_none_when_otel_unavailable(monkeypatch):
    """When _OTEL_AVAILABLE is False, yield None."""
    import ldai.observe as obs
    monkeypatch.setattr(obs, "_OTEL_AVAILABLE", False)

    from ldai.observe import _span_scope
    with _span_scope("ld.ai.test") as s:
        assert s is None


# ---------------------------------------------------------------------------
# Baggage helpers
# ---------------------------------------------------------------------------

def test_set_ai_config_baggage(patch_otel):
    from ldai.observe import set_ai_config_baggage
    baggage_mod = patch_otel["baggage"]
    context_mod = patch_otel["context"]

    ctx, token = set_ai_config_baggage("cfg-key", "var-key", "gpt-4", "openai")

    assert token is not None
    # Verify all four keys were set in baggage
    baggage_keys = [call.args[0] for call in baggage_mod.set_baggage.call_args_list]
    assert "ld.ai_config.key" in baggage_keys
    assert "ld.ai_config.variation_key" in baggage_keys
    assert "ld.ai_config.model" in baggage_keys
    assert "ld.ai_config.provider" in baggage_keys
    context_mod.attach.assert_called_once()


def test_set_ai_config_baggage_noop_when_otel_unavailable(monkeypatch):
    import ldai.observe as obs
    monkeypatch.setattr(obs, "_OTEL_AVAILABLE", False)

    from ldai.observe import set_ai_config_baggage
    ctx, token = set_ai_config_baggage("k", "v", "m", "p")
    assert ctx is None
    assert token is None


def test_detach_ai_config_baggage(patch_otel):
    from ldai.observe import detach_ai_config_baggage
    context_mod = patch_otel["context"]

    token = object()
    detach_ai_config_baggage(token)
    context_mod.detach.assert_called_once_with(token)


def test_detach_ai_config_baggage_noop_when_none(patch_otel):
    from ldai.observe import detach_ai_config_baggage
    context_mod = patch_otel["context"]

    detach_ai_config_baggage(None)
    context_mod.detach.assert_not_called()


# ---------------------------------------------------------------------------
# LDAIBaggageSpanProcessor
# ---------------------------------------------------------------------------

def test_baggage_span_processor_copies_baggage_to_span(patch_otel):
    from ldai.observe import LDAIBaggageSpanProcessor
    baggage_mod = patch_otel["baggage"]

    baggage_values = {
        "ld.ai_config.key": "my-config",
        "ld.ai_config.variation_key": "v1",
        "ld.ai_config.model": "gpt-4",
        "ld.ai_config.provider": "openai",
    }
    baggage_mod.get_baggage.side_effect = lambda key, context=None: baggage_values.get(key)

    span = _make_span()
    processor = LDAIBaggageSpanProcessor()
    processor.on_start(span)

    span.set_attribute.assert_any_call("ld.ai_config.key", "my-config")
    span.set_attribute.assert_any_call("ld.ai_config.variation_key", "v1")
    span.set_attribute.assert_any_call("ld.ai_config.model", "gpt-4")
    span.set_attribute.assert_any_call("ld.ai_config.provider", "openai")


def test_baggage_span_processor_skips_empty_values(patch_otel):
    from ldai.observe import LDAIBaggageSpanProcessor
    baggage_mod = patch_otel["baggage"]
    baggage_mod.get_baggage.return_value = None  # nothing in baggage

    span = _make_span()
    processor = LDAIBaggageSpanProcessor()
    processor.on_start(span)

    span.set_attribute.assert_not_called()


# ---------------------------------------------------------------------------
# AICompletionConfig context-manager (baggage integration)
# ---------------------------------------------------------------------------

def test_completion_config_context_manager_sets_and_clears_baggage(patch_otel):
    from ldai.observe import set_ai_config_baggage, detach_ai_config_baggage
    import ldai.observe as obs

    set_called_with = {}
    detach_token = object()

    def mock_set(cfg_key, var_key, model, provider):
        set_called_with.update(
            config_key=cfg_key, variation_key=var_key, model=model, provider=provider
        )
        return {}, detach_token

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(obs, "set_ai_config_baggage", mock_set)
    detach_calls = []
    monkeypatch.setattr(obs, "detach_ai_config_baggage", lambda t: detach_calls.append(t))

    from ldai.models import AICompletionConfig, ModelConfig, ProviderConfig
    from ldai.tracker import LDAIConfigTracker
    from unittest.mock import MagicMock

    mock_ld_client = MagicMock()
    from ldclient import Context
    ctx = Context.create("user-1")
    tracker = LDAIConfigTracker(mock_ld_client, "var-key", "cfg-key", 1, "gpt-4", "openai", ctx)

    config = AICompletionConfig(
        key="cfg-key",
        enabled=True,
        model=ModelConfig("gpt-4"),
        provider=ProviderConfig("openai"),
        tracker=tracker,
    )

    with config as c:
        assert c is config
        assert set_called_with["config_key"] == "cfg-key"
        assert set_called_with["model"] == "gpt-4"

    assert detach_calls == [detach_token]
    monkeypatch.undo()


def test_completion_config_usable_without_with(patch_otel):
    """Normal attribute access still works; __enter__/__exit__ are never called."""
    from ldai.models import AICompletionConfig, ModelConfig

    config = AICompletionConfig(key="k", enabled=True, model=ModelConfig("m"))
    assert config.key == "k"
    assert config.model.name == "m"


def test_completion_config_equality_ignores_baggage_tokens():
    """Two identical configs compare as equal regardless of _baggage_tokens state."""
    from ldai.models import AICompletionConfig

    a = AICompletionConfig(key="k", enabled=True)
    b = AICompletionConfig(key="k", enabled=True)
    a._baggage_tokens.append(object())  # mutate internal state
    assert a == b


# ---------------------------------------------------------------------------
# No-op when OTel unavailable (end-to-end import check)
# ---------------------------------------------------------------------------

def test_all_helpers_noop_when_otel_unavailable(monkeypatch):
    import ldai.observe as obs
    monkeypatch.setattr(obs, "_OTEL_AVAILABLE", False)

    from ldai.observe import (
        annotate_span_success,
        annotate_span_with_ai_config_metadata,
        annotate_span_with_duration,
        annotate_span_with_feedback,
        annotate_span_with_tokens,
        annotate_span_with_ttft,
    )

    # None of these should raise
    annotate_span_with_ai_config_metadata("k", "v", "m", "p")
    annotate_span_with_tokens(1, 2, 3)
    annotate_span_with_duration(100)
    annotate_span_with_ttft(50)
    annotate_span_success(True)
    annotate_span_with_feedback("positive")
