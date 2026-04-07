"""
OpenTelemetry span annotation helpers for the LaunchDarkly AI SDK.

All helpers in this module are no-ops when ``opentelemetry-api`` is not installed.
Install the optional dependency with::

    pip install launchdarkly-server-sdk-ai[otel]
"""

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Tuple

_log = logging.getLogger("ldai.observe")

if TYPE_CHECKING:
    from ldai.models import AIConfig

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


# ---------------------------------------------------------------------------
# Span name constants
# ---------------------------------------------------------------------------

SPAN_NAME_COMPLETION = "ld_ai.completion"
SPAN_NAME_AGENT = "ld_ai.agent"
SPAN_NAME_AGENT_GRAPH = "ld_ai.agent_graph"
SPAN_NAME_JUDGE = "ld_ai.judge"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_recording_span() -> Any:
    """Return the current recording span, or ``None`` if unavailable."""
    if not _OTEL_AVAILABLE:
        _log.info("[ldai:observe] _get_recording_span: OTel not available")
        return None
    span = _otel_trace.get_current_span()
    if span is None or not span.is_recording():
        _log.info("[ldai:observe] _get_recording_span: no active recording span (span=%r, recording=%r)",
                   span, span.is_recording() if span is not None else None)
        return None
    return span


@contextmanager
def _span_scope(name: str, context: Any = None):
    """
    Start a new span for the duration of the ``with`` block.

    If a span is already active it becomes the parent (standard OTel
    child-span behaviour).  Yields ``None`` when OTel is unavailable.

    Pass ``context`` to pin the parent explicitly — useful when the caller
    is about to yield to an async executor and the ContextVar may drift
    before the generator resumes inside ``start_as_current_span``.
    """
    if not _OTEL_AVAILABLE:
        _log.info("[ldai:observe] _span_scope('%s'): OTel not available, yielding None", name)
        yield None
        return

    _log.info("[ldai:observe] _span_scope('%s'): starting span", name)
    tracer = _otel_trace.get_tracer("launchdarkly.ai")
    with tracer.start_as_current_span(name, context=context) as span:
        _log.info("[ldai:observe] _span_scope('%s'): span started — %r", name, span)
        yield span
    _log.info("[ldai:observe] _span_scope('%s'): span ended", name)


# ---------------------------------------------------------------------------
# Span annotation helpers
# ---------------------------------------------------------------------------

def annotate_span_with_ai_config_metadata(config: 'AIConfig') -> None:
    """Annotate the active span with AI Config identification attributes."""
    span = _get_recording_span()
    if span is None:
        _log.info("[ldai:observe] annotate_span_with_ai_config_metadata: no recording span, skipping (config.key=%r)", config.key)
        return
    _log.info("[ldai:observe] annotate_span_with_ai_config_metadata: annotating span for config.key=%r", config.key)
    span.set_attribute("ld_ai.ai_config.key", config.key)
    tracker = config.tracker
    if tracker is not None:
        span.set_attribute("ld_ai.ai_config.variation_key", tracker._variation_key)
        span.set_attribute("ld_ai.ai_config.version", tracker._version)
        span.set_attribute("ld_ai.ai_config.context_key", tracker._context.key)
    model_name = config.model.name if config.model else ""
    provider_name = config.provider.name if config.provider else ""
    if model_name:
        span.set_attribute("ld_ai.ai_config.model", model_name)
    if provider_name:
        span.set_attribute("ld_ai.ai_config.provider", provider_name)


def annotate_span_with_graph_metadata(tracker: Any) -> None:
    """Annotate the active span with AI Graph Config identification attributes."""
    span = _get_recording_span()
    if span is None:
        _log.info("[ldai:observe] annotate_span_with_graph_metadata: no recording span, skipping (graph_key=%r)", tracker._graph_key)
        return
    _log.info("[ldai:observe] annotate_span_with_graph_metadata: annotating span for graph_key=%r", tracker._graph_key)
    span.set_attribute("ld_ai.ai_config.key", tracker._graph_key)
    span.set_attribute("ld_ai.ai_config.variation_key", tracker._variation_key)
    span.set_attribute("ld_ai.ai_config.version", tracker._version)
    span.set_attribute("ld_ai.ai_config.context_key", tracker._context.key)


def annotate_span_with_tokens(total: int, input_tokens: int, output_tokens: int) -> None:
    """Annotate the active span with token-usage attributes."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld_ai.metrics.tokens.total", total)
    span.set_attribute("ld_ai.metrics.tokens.input", input_tokens)
    span.set_attribute("ld_ai.metrics.tokens.output", output_tokens)


def annotate_span_with_duration(duration_ms: int) -> None:
    """Annotate the active span with a duration attribute (milliseconds)."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld_ai.metrics.duration_ms", duration_ms)


def annotate_span_with_ttft(ttft_ms: int) -> None:
    """Annotate the active span with a time-to-first-token attribute (ms)."""
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld_ai.metrics.time_to_first_token_ms", ttft_ms)


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
    span.set_attribute("ld_ai.metrics.feedback.kind", kind)


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
        span.set_attribute("ld_ai.judge.config_key", judge_response.judge_config_key)
    span.set_attribute("ld_ai.judge.success", judge_response.success)
    if judge_response.error:
        span.set_attribute("ld_ai.judge.error", judge_response.error)

    for metric_key, eval_score in (judge_response.evals or {}).items():
        if isinstance(eval_score, EvalScore):
            span.set_attribute(f"ld_ai.judge.{metric_key}.score", eval_score.score)
            if eval_score.reasoning:
                span.set_attribute(f"ld_ai.judge.{metric_key}.reasoning", eval_score.reasoning)


# ---------------------------------------------------------------------------
# Baggage propagation
# ---------------------------------------------------------------------------

_BAGGAGE_KEY_CONFIG_KEY = "ld_ai.ai_config.key"
_BAGGAGE_KEY_VARIATION_KEY = "ld_ai.ai_config.variation_key"
_BAGGAGE_KEY_MODEL = "ld_ai.ai_config.model"
_BAGGAGE_KEY_PROVIDER = "ld_ai.ai_config.provider"


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
    automatically inherit ``ld_ai.ai_config.*`` attributes on their spans
    without any additional code.

    When ``opentelemetry-api`` is not installed this class is a no-op.
    """

    _BAGGAGE_TO_ATTRIBUTE = {
        _BAGGAGE_KEY_CONFIG_KEY: "ld_ai.ai_config.key",
        _BAGGAGE_KEY_VARIATION_KEY: "ld_ai.ai_config.variation_key",
        _BAGGAGE_KEY_MODEL: "ld_ai.ai_config.model",
        _BAGGAGE_KEY_PROVIDER: "ld_ai.ai_config.provider",
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
