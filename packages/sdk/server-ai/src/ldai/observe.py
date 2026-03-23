"""
LLM observability integration for the LaunchDarkly AI Config SDK.

This module provides:

1. **LDAIObserveConfig** — developer-friendly dataclass that controls how the SDK
   writes LLM metrics and AI Config metadata onto OpenTelemetry spans.
   Pass it to LDAIClient to opt in/out of features::

       from ldai import LDAIClient
       from ldai.observe import LDAIObserveConfig

       # defaults: annotate active spans, create an internal span when none exists
       aiclient = LDAIClient(ld_client)

       # disable all span annotation (LD analytics events still fire)
       aiclient = LDAIClient(ld_client, observe=LDAIObserveConfig(annotate_spans=False))

       # annotate active spans only; don't create internal spans
       aiclient = LDAIClient(ld_client, observe=LDAIObserveConfig(create_span_if_none=False))

2. **Span annotation helpers** — write LLM metrics (tokens, duration, success,
   feedback) and AI Config metadata onto the currently active OTel span.
   No-ops when opentelemetry-api is not installed.

3. **LDAIBaggageSpanProcessor** — a SpanProcessor that copies LaunchDarkly AI
   Config metadata from OTel baggage onto every new span.  Useful when using
   config_scope() with auto-instrumented LLM libraries (e.g. OpenLLMetry)::

       from opentelemetry.sdk.trace import TracerProvider
       from ldai.observe import LDAIBaggageSpanProcessor

       provider = TracerProvider()
       provider.add_span_processor(LDAIBaggageSpanProcessor())

All public symbols in this module are safe to call when opentelemetry-api is
not installed — they silently do nothing.  LDAIBaggageSpanProcessor requires
opentelemetry-sdk.
"""

from contextlib import contextmanager
from dataclasses import dataclass

try:
    from opentelemetry import baggage as _otel_baggage
    from opentelemetry import context as _otel_context
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import StatusCode
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    StatusCode = None  # type: ignore[assignment]

# LDAIBaggageSpanProcessor subclasses SpanProcessor from the OTel SDK when
# available.  When the SDK is not installed we fall back to object so the
# class can still be imported without error.
try:
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessorBase
    _SDK_AVAILABLE = True
except ImportError:
    _SpanProcessorBase = object  # type: ignore[assignment,misc]
    _SDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Developer-facing configuration
# ---------------------------------------------------------------------------

@dataclass
class LDAIObserveConfig:
    """
    Controls how the LaunchDarkly AI SDK writes observability data onto spans.

    Pass an instance to :class:`ldai.LDAIClient` at construction time::

        from ldai import LDAIClient
        from ldai.observe import LDAIObserveConfig

        # All defaults — recommended for most applications
        aiclient = LDAIClient(ld_client)

        # Disable span annotation; LD analytics events still fire normally
        aiclient = LDAIClient(ld_client, observe=LDAIObserveConfig(annotate_spans=False))

        # Annotate existing spans only; don't create an internal span when
        # no OTel span is active at call time
        aiclient = LDAIClient(ld_client, observe=LDAIObserveConfig(create_span_if_none=False))

    Attributes:
        annotate_spans: When True (default), the SDK writes AI Config metadata
            (key, variation, model, provider) and LLM metrics (token counts,
            duration, success/error, feedback) as attributes onto the active
            OTel span.  Set to False to disable all span annotation while
            keeping LaunchDarkly analytics tracking intact.

        create_span_if_none: When True (default) and ``annotate_spans`` is also
            True, the SDK creates an internal ``ld.ai.completion`` span when no
            OTel span is active at the time of the LLM call.  The span is
            exported through whatever ``TracerProvider`` is globally registered
            (e.g. the LaunchDarkly Observability plugin).  Set to False if you
            only want to annotate spans you create yourself.
    """

    annotate_spans: bool = True
    create_span_if_none: bool = True


# ---------------------------------------------------------------------------
# Baggage key constants
# ---------------------------------------------------------------------------

_BAGGAGE_CONFIG_KEY = "ld.ai_config.key"
_BAGGAGE_VARIATION_KEY = "ld.ai_config.variation_key"
_BAGGAGE_MODEL_KEY = "ld.ai_config.model"
_BAGGAGE_PROVIDER_KEY = "ld.ai_config.provider"

_INTERNAL_SPAN_NAME = "ld.ai.completion"
_TRACER_NAME = "launchdarkly-server-sdk-ai"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_recording_span():
    """Return the active OTel span if it is recording, otherwise None."""
    if not _OTEL_AVAILABLE:
        return None
    span = _otel_trace.get_current_span()
    if span is None or not span.is_recording():
        return None
    return span


@contextmanager
def _span_scope(name: str = _INTERNAL_SPAN_NAME, create_if_none: bool = True):
    """
    Context manager that ensures an active recording span for its duration.

    - If a recording span already exists it is yielded as-is (no new span).
    - If no recording span exists and ``create_if_none`` is True, an internal
      span is created via the global TracerProvider and made current.
    - Otherwise yields None; all annotation calls inside will be no-ops.

    Requires opentelemetry-sdk when creating a new span; safe to call when
    only opentelemetry-api is installed (falls back to yield None).
    """
    span = _get_recording_span()
    if span is not None:
        yield span
    elif create_if_none and _SDK_AVAILABLE and _OTEL_AVAILABLE:
        tracer = _otel_trace.get_tracer(_TRACER_NAME)
        with tracer.start_as_current_span(name) as new_span:
            yield new_span
    else:
        yield None


# ---------------------------------------------------------------------------
# Span annotation helpers (called by LDAIConfigTracker)
# ---------------------------------------------------------------------------

def annotate_span_with_ai_config_metadata(
    config_key: str,
    variation_key: str,
    model_name: str,
    provider_name: str,
    version: int = 0,
    context_key: str = "",
    enabled: bool = True,
) -> None:
    """
    Write AI Config identity attributes onto the currently active OTel span.

    Attributes written:
      ld.ai_config.key           — AI Config flag key
      ld.ai_config.variation_key — evaluated variation key
      ld.ai_config.version       — variation version
      ld.ai_config.context_key   — LaunchDarkly context key
      ld.ai_config.enabled       — whether the AI Config is enabled (mode)
      ld.ai_config.model         — model name (omitted when empty)
      ld.ai_config.provider      — provider name (omitted when empty)

    No-op when opentelemetry-api is not installed or no recording span is active.
    """
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai_config.key", config_key)
    span.set_attribute("ld.ai_config.variation_key", variation_key)
    if version:
        span.set_attribute("ld.ai_config.version", version)
    if context_key:
        span.set_attribute("ld.ai_config.context_key", context_key)
    span.set_attribute("ld.ai_config.enabled", enabled)
    if model_name:
        span.set_attribute("ld.ai_config.model", model_name)
    if provider_name:
        span.set_attribute("ld.ai_config.provider", provider_name)


def annotate_span_with_tokens(total: int, input_tokens: int, output_tokens: int) -> None:
    """
    Write token usage attributes onto the currently active OTel span.

      ld.ai.metrics.tokens.total  — total token count
      ld.ai.metrics.tokens.input  — prompt / input tokens
      ld.ai.metrics.tokens.output — completion / output tokens

    No-op when opentelemetry-api is not installed or no recording span is active.
    """
    span = _get_recording_span()
    if span is None:
        return
    if total > 0:
        span.set_attribute("ld.ai.metrics.tokens.total", total)
    if input_tokens > 0:
        span.set_attribute("ld.ai.metrics.tokens.input", input_tokens)
    if output_tokens > 0:
        span.set_attribute("ld.ai.metrics.tokens.output", output_tokens)


def annotate_span_with_duration(duration_ms: int) -> None:
    """
    Write ``ld.ai.metrics.duration_ms`` onto the currently active OTel span.

    No-op when opentelemetry-api is not installed or no recording span is active.
    """
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai.metrics.duration_ms", duration_ms)


def annotate_span_with_ttft(ttft_ms: int) -> None:
    """
    Write ``ld.ai.metrics.time_to_first_token_ms`` onto the currently active OTel span.

    No-op when opentelemetry-api is not installed or no recording span is active.
    """
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai.metrics.time_to_first_token_ms", ttft_ms)


def annotate_span_success(success: bool) -> None:
    """
    Set the active span status to OK or ERROR.

    No-op when opentelemetry-api is not installed or no recording span is active.
    """
    if not _OTEL_AVAILABLE:
        return
    span = _get_recording_span()
    if span is None:
        return
    span.set_status(StatusCode.OK if success else StatusCode.ERROR)


def annotate_span_with_feedback(kind: str) -> None:
    """
    Write ``ld.ai.metrics.feedback.kind`` onto the currently active OTel span.

    No-op when opentelemetry-api is not installed or no recording span is active.
    """
    span = _get_recording_span()
    if span is None:
        return
    span.set_attribute("ld.ai.metrics.feedback.kind", kind)


def annotate_span_with_judge_response(judge_response) -> None:
    """
    Write judge evaluation results onto the currently active OTel span.

    For each eval in the response, two attributes are written using the
    sanitized metric key as a namespace:

      ld.ai.judge.<metric>.score     — numeric score between 0 and 1
      ld.ai.judge.<metric>.reasoning — reasoning text

    Plus top-level judge attributes:

      ld.ai.judge.config_key — key of the judge AI Config
      ld.ai.judge.success    — whether the evaluation completed successfully
      ld.ai.judge.error      — error message (only when evaluation failed)

    Metric keys like ``$ld:ai:judge:relevance`` are sanitized to
    ``relevance`` (``$`` stripped, ``:``-separated segments, last segment used).

    No-op when opentelemetry-api is not installed or no recording span is active.
    """
    span = _get_recording_span()
    if span is None:
        return

    if judge_response.judge_config_key:
        span.set_attribute("ld.ai.judge.config_key", judge_response.judge_config_key)
    span.set_attribute("ld.ai.judge.success", judge_response.success)
    if judge_response.error:
        span.set_attribute("ld.ai.judge.error", judge_response.error)

    for metric_key, eval_score in (judge_response.evals or {}).items():
        # Sanitize metric key: strip leading '$', use last ':'-separated segment
        clean = metric_key.lstrip("$").split(":")[-1] if metric_key else metric_key
        span.set_attribute(f"ld.ai.judge.{clean}.score", eval_score.score)
        if eval_score.reasoning:
            span.set_attribute(f"ld.ai.judge.{clean}.reasoning", eval_score.reasoning)


# ---------------------------------------------------------------------------
# Baggage helpers (used by LDAIClient.config_scope())
# ---------------------------------------------------------------------------

def set_ai_config_baggage(
    config_key: str,
    variation_key: str,
    model_name: str,
    provider_name: str,
):
    """
    Attach AI Config metadata to the active OTel context via baggage.

    Returns ``(ctx, token)``.  The token must be passed to
    :func:`detach_ai_config_baggage` to clean up.  Returns ``(None, None)``
    when opentelemetry-api is not installed.
    """
    if not _OTEL_AVAILABLE:
        return None, None

    ctx = _otel_baggage.set_baggage(_BAGGAGE_CONFIG_KEY, config_key)
    ctx = _otel_baggage.set_baggage(_BAGGAGE_VARIATION_KEY, variation_key, context=ctx)
    if model_name:
        ctx = _otel_baggage.set_baggage(_BAGGAGE_MODEL_KEY, model_name, context=ctx)
    if provider_name:
        ctx = _otel_baggage.set_baggage(_BAGGAGE_PROVIDER_KEY, provider_name, context=ctx)

    token = _otel_context.attach(ctx)
    return ctx, token


def detach_ai_config_baggage(token) -> None:
    """
    Remove AI Config baggage from the OTel context.

    No-op when opentelemetry-api is not installed or token is None.
    """
    if not _OTEL_AVAILABLE or token is None:
        return
    _otel_context.detach(token)


# ---------------------------------------------------------------------------
# LDAIBaggageSpanProcessor
# ---------------------------------------------------------------------------

class LDAIBaggageSpanProcessor(_SpanProcessorBase):
    """
    An OTel SpanProcessor that copies LaunchDarkly AI Config metadata from
    OTel baggage onto every new span as span attributes.

    Useful when using :meth:`LDAIClient.config_scope` together with
    auto-instrumented LLM libraries (e.g. OpenLLMetry), so that spans created
    inside the scope automatically carry AI Config metadata.

    Baggage key                -> Span attribute
    ld.ai_config.key           -> ld.ai_config.key
    ld.ai_config.variation_key -> ld.ai_config.variation_key
    ld.ai_config.model         -> ld.ai_config.model
    ld.ai_config.provider      -> ld.ai_config.provider

    Register once at application startup::

        from opentelemetry.sdk.trace import TracerProvider
        from ldai.observe import LDAIBaggageSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(LDAIBaggageSpanProcessor())
        trace.set_tracer_provider(provider)

    Requires opentelemetry-sdk (not just opentelemetry-api).
    """

    _BAGGAGE_TO_ATTRIBUTE = {
        _BAGGAGE_CONFIG_KEY:    "ld.ai_config.key",
        _BAGGAGE_VARIATION_KEY: "ld.ai_config.variation_key",
        _BAGGAGE_MODEL_KEY:     "ld.ai_config.model",
        _BAGGAGE_PROVIDER_KEY:  "ld.ai_config.provider",
    }

    def on_start(self, span, parent_context=None):
        """Copy LD AI Config baggage entries onto the starting span as attributes."""
        if not _OTEL_AVAILABLE:
            return
        ctx = parent_context if parent_context is not None else _otel_context.get_current()
        for baggage_key, attr_key in self._BAGGAGE_TO_ATTRIBUTE.items():
            value = _otel_baggage.get_baggage(baggage_key, context=ctx)
            if value:
                span.set_attribute(attr_key, value)

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        pass
