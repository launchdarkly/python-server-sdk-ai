"""
Optional OpenTelemetry span instrumentation for managed run methods.

This module is a strict no-op when ``opentelemetry-api`` is not installed.
Install the optional dependency with::

    pip install launchdarkly-server-sdk-ai[otel]

The helper :func:`run_span` returns either a real
``start_as_current_span`` context manager (when OTel is available) or
``contextlib.nullcontext()`` otherwise. It is safe to use in both
synchronous ``with`` blocks and as the surrounding scope for an awaited
coroutine.
"""

from contextlib import nullcontext
from typing import Any, ContextManager

try:
    from opentelemetry import trace as _otel_trace

    _tracer: Any = _otel_trace.get_tracer("ld.ai")
    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised when OTel is absent
    _tracer = None
    _HAS_OTEL = False


# Attribute name used to identify the LaunchDarkly AI config a managed
# type was created with. Defined once and referenced everywhere — never
# hardcode the string at call sites.
_CONFIG_KEY_ATTR = "ld.ai.config_key"


def run_span(span_name: str, config_key: str) -> ContextManager[Any]:
    """
    Return a context manager that emits an OTel span for a managed run.

    When OpenTelemetry is installed the returned context manager starts
    a new span as the current span, sets the :data:`_CONFIG_KEY_ATTR`
    attribute to *config_key*, and ends the span on exit. When OTel is
    not installed this is :class:`contextlib.nullcontext` and adds no
    overhead.

    :param span_name: The span name to emit (e.g. ``ld.ai.agent.run``).
    :param config_key: The AI config key the managed type was created with.
    :return: A context manager suitable for use with ``with`` or ``async with``-style
        wrapping around an awaited coroutine.
    """
    if not _HAS_OTEL:
        return nullcontext()

    cm = _tracer.start_as_current_span(span_name)

    class _SpanScope:
        def __enter__(self) -> Any:
            span = cm.__enter__()
            try:
                span.set_attribute(_CONFIG_KEY_ATTR, config_key)
            except Exception:  # pragma: no cover - defensive
                pass
            return span

        def __exit__(self, exc_type, exc, tb) -> Any:
            return cm.__exit__(exc_type, exc, tb)

    return _SpanScope()
