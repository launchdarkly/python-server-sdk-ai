import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from ldai.tracker import LDAIConfigTracker

_log = logging.getLogger("ldai.models")


@dataclass
class LDMessage:
    role: Literal['system', 'user', 'assistant']
    content: str

    def to_dict(self) -> dict:
        """
        Render the given message as a dictionary object.
        """
        return {
            'role': self.role,
            'content': self.content,
        }


class ModelConfig:
    """
    Configuration related to the model.
    """

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None, custom: Optional[Dict[str, Any]] = None):
        """
        :param name: The name of the model.
        :param parameters: Additional model-specific parameters.
        :param custom: Additional customer provided data.
        """
        self._name = name
        self._parameters = parameters
        self._custom = custom

    @property
    def name(self) -> str:
        """
        The name of the model.
        """
        return self._name

    def get_parameter(self, key: str) -> Any:
        """
        Retrieve model-specific parameters.

        Accessing a named, typed attribute (e.g. name) will result in the call
        being delegated to the appropriate property.
        """
        if key == 'name':
            return self.name

        if self._parameters is None:
            return None

        return self._parameters.get(key)

    def get_custom(self, key: str) -> Any:
        """
        Retrieve customer provided data.
        """
        if self._custom is None:
            return None

        return self._custom.get(key)

    def to_dict(self) -> dict:
        """
        Render the given model config as a dictionary object.
        """
        return {
            'name': self._name,
            'parameters': self._parameters,
            'custom': self._custom,
        }


class ProviderConfig:
    """
    Configuration related to the provider.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """
        The name of the provider.
        """
        return self._name

    def to_dict(self) -> dict:
        """
        Render the given provider config as a dictionary object.
        """
        return {
            'name': self._name,
        }


# ============================================================================
# Judge Types
# ============================================================================

@dataclass(frozen=True)
class JudgeConfiguration:
    """
    Configuration for judge attachment to AI Configs.
    """

    @dataclass(frozen=True)
    class Judge:
        """
        Configuration for a single judge attachment.
        """
        key: str
        sampling_rate: float

        def to_dict(self) -> dict:
            """
            Render the judge as a dictionary object.
            """
            return {
                'key': self.key,
                'samplingRate': self.sampling_rate,
            }

    judges: List['JudgeConfiguration.Judge']

    def to_dict(self) -> dict:
        """
        Render the judge configuration as a dictionary object.
        """
        return {
            'judges': [judge.to_dict() for judge in self.judges],
        }


# ============================================================================
# Base AI Config Types
# ============================================================================

@dataclass(frozen=True)
class AIConfigDefault:
    """
    Base AI Config interface for default implementations with optional enabled property.
    """
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None

    @classmethod
    def disabled(cls):
        """
        Returns a new disabled config default with enabled set to false.
        When called on a subclass, returns an instance of that subclass.
        """
        return cls(enabled=False)

    def _base_to_dict(self) -> Dict[str, Any]:
        """
        Render the base config fields as a dictionary object.
        """
        return {
            '_ldMeta': {
                'enabled': self.enabled or False,
            },
            'model': self.model.to_dict() if self.model else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }


@dataclass(frozen=True)
class AIConfig:
    """
    Base AI Config interface without mode-specific fields.

    Subclasses that set ``_span_name`` are context managers that open an
    OpenTelemetry span and set baggage for the duration of the block::

        with client.completion_config(key, ctx, default) as config:
            response = openai_client.chat.completions.create(...)
            config.tracker.track_success()
    """
    # Subclasses set this to the appropriate SPAN_NAME_* constant.
    _span_name: ClassVar[str] = ""

    key: str
    enabled: bool
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None
    tracker: Optional[LDAIConfigTracker] = None
    # Mutable lists inside a frozen dataclass — store OTel state across
    # __enter__/__exit__. List mutation does not violate frozen semantics.
    _baggage_tokens: list = field(default_factory=list, repr=False, compare=False, hash=False)
    _span_scopes: list = field(default_factory=list, repr=False, compare=False, hash=False)

    def __enter__(self) -> 'AIConfig':
        from ldai.observe import (
            _OTEL_AVAILABLE,
            _otel_context,
            _span_scope,
            annotate_span_with_ai_config_metadata,
            set_ai_config_baggage,
        )
        _log.info("[ldai:models] AIConfig.__enter__: key=%r span_name=%r tracker=%r",
                   self.key, self._span_name, self.tracker)
        if self.tracker is not None:
            if self._span_name:
                _log.info("[ldai:models] AIConfig.__enter__: starting span scope '%s' for key=%r", self._span_name, self.key)
                # Capture the current OTel context now so start_as_current_span
                # uses the right parent even if the ContextVar drifts before the
                # generator resumes inside the scope.
                ctx = _otel_context.get_current() if _OTEL_AVAILABLE and _otel_context is not None else None
                scope = _span_scope(self._span_name, context=ctx)
                scope.__enter__()
                self._span_scopes.append(scope)
            annotate_span_with_ai_config_metadata(self)
            _, token = set_ai_config_baggage(
                self.key,
                self.tracker._variation_key,
                self.model.name if self.model else "",
                self.provider.name if self.provider else "",
            )
        else:
            _log.info("[ldai:models] AIConfig.__enter__: no tracker for key=%r, skipping span/baggage", self.key)
            token = None
        self._baggage_tokens.append(token)
        return self

    def __exit__(self, *exc) -> None:
        from ldai.observe import detach_ai_config_baggage
        _log.info("[ldai:models] AIConfig.__exit__: key=%r exc=%r", self.key, exc)
        if self._baggage_tokens:
            detach_ai_config_baggage(self._baggage_tokens.pop())
        if self._span_scopes:
            self._span_scopes.pop().__exit__(*exc)

    def _base_to_dict(self) -> Dict[str, Any]:
        """
        Render the base config fields as a dictionary object.
        """
        return {
            '_ldMeta': {
                'enabled': self.enabled,
            },
            'model': self.model.to_dict() if self.model else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }


# ============================================================================
# Completion Config Types
# ============================================================================

@dataclass(frozen=True)
class AICompletionConfigDefault(AIConfigDefault):
    """
    Default Completion AI Config (default mode).
    """
    messages: Optional[List[LDMessage]] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> dict:
        """
        Render the given default values as an AICompletionConfigDefault-compatible dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


@dataclass(frozen=True)
class AICompletionConfig(AIConfig):
    """Completion AI Config (default mode)."""

    _span_name: ClassVar[str] = "ld.ai.completion"

    messages: Optional[List[LDMessage]] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> dict:
        """
        Render the given completion config as a dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


# ============================================================================
# Agent Config Types
# ============================================================================

@dataclass(frozen=True)
class AIAgentConfigDefault(AIConfigDefault):
    """
    Default Agent-specific AI Config with instructions.
    """
    instructions: Optional[str] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config default as a dictionary object.
        """
        result = self._base_to_dict()
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


@dataclass(frozen=True)
class AIAgentConfig(AIConfig):
    """Agent-specific AI Config with instructions."""

    _span_name: ClassVar[str] = "ld.ai.agent"

    instructions: Optional[str] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config as a dictionary object.
        """
        result = self._base_to_dict()
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


# ============================================================================
# Judge Config Types
# ============================================================================

@dataclass(frozen=True)
class AIJudgeConfigDefault(AIConfigDefault):
    """
    Default Judge-specific AI Config with required evaluation metric key.
    """
    messages: Optional[List[LDMessage]] = None
    # Deprecated: evaluation_metric_key is used instead
    evaluation_metric_keys: Optional[List[str]] = None
    evaluation_metric_key: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Render the given judge config default as a dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        result['evaluationMetricKey'] = self.evaluation_metric_key
        # Include deprecated evaluationMetricKeys for backward compatibility
        if self.evaluation_metric_keys:
            result['evaluationMetricKeys'] = self.evaluation_metric_keys
        return result


@dataclass(frozen=True)
class AIJudgeConfig(AIConfig):
    """Judge-specific AI Config with required evaluation metric key."""

    _span_name: ClassVar[str] = "ld.ai.judge"

    # Deprecated: evaluation_metric_key is used instead
    evaluation_metric_keys: List[str] = field(default_factory=list)
    messages: Optional[List[LDMessage]] = None
    evaluation_metric_key: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Render the given judge config as a dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        result['evaluationMetricKey'] = self.evaluation_metric_key
        return result


# ============================================================================
# Agent Request Config
# ============================================================================

@dataclass
class AIAgentConfigRequest:
    """
    Configuration for a single agent request.

    Combines agent key with its specific default configuration and variables.
    """
    key: str
    default: Optional[AIAgentConfigDefault] = None
    variables: Optional[Dict[str, Any]] = None


# Type alias for multiple agents
AIAgents = Dict[str, AIAgentConfig]

# Type alias for all AI Config variants
AIConfigKind = Union[AIAgentConfig, AICompletionConfig, AIJudgeConfig]

# ============================================================================
# AI Config Agent Graph Edge Type
# ============================================================================


@dataclass
class Edge:
    """
    Edge configuration for an agent graph.
    """

    key: str
    source_config: str
    target_config: str
    handoff: Optional[dict] = field(default_factory=dict)


# ============================================================================
# AI Config Agent Graph
# ============================================================================
@dataclass
class AIAgentGraphConfig:
    """
    Agent graph configuration.
    """

    key: str
    root_config_key: str
    edges: List[Edge]
    enabled: bool = True


# ============================================================================
# Deprecated Type Aliases for Backward Compatibility
# ============================================================================

# Note: AIConfig is now defined above as a base class (line 169).
# For backward compatibility, code should migrate to:
# - Use AICompletionConfigDefault for default/input values
# - Use AICompletionConfig for return values

# Deprecated: Use AIAgentConfigDefault instead
LDAIAgentDefaults = AIAgentConfigDefault

# Deprecated: Use AIAgentConfigRequest instead
LDAIAgentConfig = AIAgentConfigRequest

# Deprecated: Use AIAgentConfig instead (note: this was the old return type)
LDAIAgent = AIAgentConfig
