"""
OpenTelemetry span annotation helpers for the LaunchDarkly AI SDK.

All helpers in this module are no-ops when ``opentelemetry-api`` is not installed.
Install the optional dependency with::

    pip install launchdarkly-server-sdk-ai[otel]
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Tuple

_otel_baggage = None  # type: ignore[assignment]
_otel_context = None  # type: ignore[assignment]
_otel_trace = None  # type: ignore[assignment]
StatusCode = None  # type: ignore[assignment]

try:
    from opentelemetry import baggage as _otel_baggage  # type: ignore[assignment]
    from opentelemetry import context as _otel_context  # type: ignore[assignment]
    from opentelemetry import trace as _otel_trace  # type: ignore[assignment]
    from opentelemetry.trace import StatusCode  # type: ignore[assignment]
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


@dataclass
class LDAIObserveConfig:
    """
    Configuration for OpenTelemetry span annotation behaviour.

    :param annotate_spans: When ``True`` (default), tracker methods annotate
        the active span with metric attributes.
    :param create_span_if_none: When ``True`` (default), the managed wrappers
        create an internal span if no recording span is already active.  Set to
        ``False`` to only annotate existing spans.
    """

    annotate_spans: bool = True
    create_span_if_none: bool = True


# ---------------------------------------------------------------------------
# Span name constants
# ---------------------------------------------------------------------------

SPAN_NAME_COMPLETION = "ld.ai.completion"
SPAN_NAME_AGENT = "ld.ai.agent"
SPAN_NAME_AGENT_GRAPH = "ld.ai.agent_graph"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_recording_span() -> Any:
    """Return the current recording span, or ``None`` if unavailable."""
    if not _OTEL_AVAILABLE:
        return None
    span = _otel_trace.get_current_span()
    if span is None or not span.is_recording():
        return None
    return span


@contextmanager
def _span_scope(name: str, create_if_none: bool = True):
    """
    Ensure a recording span is active for the duration of the ``with`` block.

    * If a recording span is already active, yield it as-is (no nesting).
    * If no span is active and *create_if_none* is ``True``, start an internal
      span via the global TracerProvider and yield it.
    * Otherwise yield ``None`` — annotation calls become no-ops.
    """
    if not _OTEL_AVAILABLE:
        yield None
        return

    existing = _otel_trace.get_current_span()
    if existing is not None and existing.is_recording():
        yield existing
        return

    if not create_if_none:
        yield None
        return

    tracer = _otel_trace.get_tracer("launchdarkly.ai")
    with tracer.start_as_current_span(name) as span:
        yield span


# ---------------------------------------------------------------------------
# Span annotation helpers
# ---------------------------------------------------------------------------

def annotate_span_with_ai_config_metadata(
    config_key: str,
    variation_key: str,
    model_name: str,
    provider_name: str,
    version: int = 0,
    context_key: str = "",
) -> None:
    """Annotate the active span with AI Config identification attributes."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai_config.key", config_key)
    span.set_attribute("ld.ai_config.variation_key", variation_key)
    span.set_attribute("ld.ai_config.version", version)
    span.set_attribute("ld.ai_config.context_key", context_key)
    if model_name:
        span.set_attribute("ld.ai_config.model", model_name)
    if provider_name:
        span.set_attribute("ld.ai_config.provider", provider_name)


def annotate_span_with_tokens(total: int, input_tokens: int, output_tokens: int) -> None:
    """Annotate the active span with token-usage attributes."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai.metrics.tokens.total", total)
    span.set_attribute("ld.ai.metrics.tokens.input", input_tokens)
    span.set_attribute("ld.ai.metrics.tokens.output", output_tokens)


def annotate_span_with_duration(duration_ms: int) -> None:
    """Annotate the active span with a duration attribute (milliseconds)."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai.metrics.duration_ms", duration_ms)


def annotate_span_with_ttft(ttft_ms: int) -> None:
    """Annotate the active span with a time-to-first-token attribute (ms)."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai.metrics.time_to_first_token_ms", ttft_ms)


def annotate_span_success(success: bool) -> None:
    """Set the active span status to OK or ERROR based on *success*."""
    if not _OTEL_AVAILABLE:
        return
    span = _get_recording_span()
    if span is None:
        return
    if success:
        span.set_status(StatusCode.OK)  # type: ignore[arg-type]
    else:
        span.set_status(StatusCode.ERROR)  # type: ignore[arg-type]


def annotate_span_with_feedback(kind: str) -> None:
    """Annotate the active span with a feedback kind attribute."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai.metrics.feedback.kind", kind)


def annotate_span_with_judge_response(judge_response: Any) -> None:
    """Annotate the active span with judge evaluation attributes."""
    span = _get_recording_span()
    if span is None:
        return

    # Avoid a hard import of JudgeResponse to keep observe.py provider-agnostic
    try:
        from ldai.providers.types import EvalScore, JudgeResponse  # noqa: F401
    except ImportError:
        return

    if not isinstance(judge_response, JudgeResponse):
        return

    if judge_response.judge_config_key:
        span.set_attribute("ld.ai.judge.config_key", judge_response.judge_config_key)
    span.set_attribute("ld.ai.judge.success", judge_response.success)
    if judge_response.error:
        span.set_attribute("ld.ai.judge.error", judge_response.error)

    for metric_key, eval_score in (judge_response.evals or {}).items():
        if isinstance(eval_score, EvalScore):
            span.set_attribute(f"ld.ai.judge.{metric_key}.score", eval_score.score)
            if eval_score.reasoning:
                span.set_attribute(f"ld.ai.judge.{metric_key}.reasoning", eval_score.reasoning)


# ---------------------------------------------------------------------------
# Baggage propagation
# ---------------------------------------------------------------------------

_BAGGAGE_KEY_CONFIG_KEY = "ld.ai_config.key"
_BAGGAGE_KEY_VARIATION_KEY = "ld.ai_config.variation_key"
_BAGGAGE_KEY_MODEL = "ld.ai_config.model"
_BAGGAGE_KEY_PROVIDER = "ld.ai_config.provider"


def set_ai_config_baggage(
    config_key: str,
    variation_key: str,
    model_name: str,
    provider_name: str,
) -> Tuple[Any, Any]:
    """
    Attach AI Config metadata to the OTel context via baggage.

    Returns ``(ctx, token)`` where *token* must be passed to
    :func:`detach_ai_config_baggage` when the scope exits.
    When OTel is unavailable both values are ``None``.
    """
    if not _OTEL_AVAILABLE:
        return None, None

    ctx = _otel_context.get_current()
    ctx = _otel_baggage.set_baggage(_BAGGAGE_KEY_CONFIG_KEY, config_key, context=ctx)
    ctx = _otel_baggage.set_baggage(_BAGGAGE_KEY_VARIATION_KEY, variation_key, context=ctx)
    if model_name:
        ctx = _otel_baggage.set_baggage(_BAGGAGE_KEY_MODEL, model_name, context=ctx)
    if provider_name:
        ctx = _otel_baggage.set_baggage(_BAGGAGE_KEY_PROVIDER, provider_name, context=ctx)

    token = _otel_context.attach(ctx)
    return ctx, token


def detach_ai_config_baggage(token: Any) -> None:
    """Remove AI Config baggage from the current OTel context."""
    if not _OTEL_AVAILABLE or token is None:
        return
    _otel_context.detach(token)


# ---------------------------------------------------------------------------
# LDAIBaggageSpanProcessor
# ---------------------------------------------------------------------------

class LDAIBaggageSpanProcessor:
    """
    An OpenTelemetry span processor that copies LaunchDarkly AI Config
    metadata from baggage onto every new span as attributes.

    Register with your ``TracerProvider``::

        from opentelemetry.sdk.trace import TracerProvider
        from ldai.observe import LDAIBaggageSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(LDAIBaggageSpanProcessor())
        trace.set_tracer_provider(provider)

    This enables auto-instrumented libraries (e.g. OpenLLMetry) to
    automatically inherit ``ld.ai_config.*`` attributes on their spans
    without any additional code.

    When ``opentelemetry-api`` is not installed this class is a no-op.
    """

    _BAGGAGE_TO_ATTRIBUTE = {
        _BAGGAGE_KEY_CONFIG_KEY: "ld.ai_config.key",
        _BAGGAGE_KEY_VARIATION_KEY: "ld.ai_config.variation_key",
        _BAGGAGE_KEY_MODEL: "ld.ai_config.model",
        _BAGGAGE_KEY_PROVIDER: "ld.ai_config.provider",
    }

    def on_start(self, span: Any, parent_context: Optional[Any] = None) -> None:
        if not _OTEL_AVAILABLE:
            return
        ctx = parent_context if parent_context is not None else _otel_context.get_current()
        for baggage_key, attr_key in self._BAGGAGE_TO_ATTRIBUTE.items():
            value = _otel_baggage.get_baggage(baggage_key, context=ctx)
            if value:
                span.set_attribute(attr_key, value)

    def on_end(self, span: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
