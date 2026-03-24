"""
Backward-compatibility shim — import from ldai.observe instead.

LDAIOtelConfig is a deprecated alias for LDAIObserveConfig.
"""

from ldai.observe import (  # noqa: F401
    LDAIObserveConfig as LDAIOtelConfig,
    LDAIBaggageSpanProcessor,
    annotate_span_with_ai_config_metadata,
    annotate_span_with_tokens,
    annotate_span_with_duration,
    annotate_span_with_ttft,
    annotate_span_success,
    annotate_span_with_feedback,
    set_ai_config_baggage,
    detach_ai_config_baggage,
    _span_scope,
    _get_recording_span,
)
