from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    from ldai.evaluator import Evaluator

from typing_extensions import Self


@dataclass(frozen=True)
class LDTool:
    """
    A single tool entry from the root-level tools map in an AI Config flag variation.
    Distinct from model.parameters.tools[] which is the raw array passed to LLM providers.
    """
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    custom_parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        result: Dict[str, Any] = {'name': self.name}
        if self.description is not None:
            result['description'] = self.description
        if self.type is not None:
            result['type'] = self.type
        if self.parameters is not None:
            result['parameters'] = self.parameters
        if self.custom_parameters is not None:
            result['customParameters'] = self.custom_parameters  # camelCase in wire format
        return result


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

    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        custom: Optional[Dict[str, Any]] = None,
        region: Optional[str] = None,
        model_key: Optional[str] = None,
        model_version: Optional[int] = None,
    ):
        """
        :param name: The name of the model.
        :param parameters: Additional model-specific parameters.
        :param custom: Additional customer provided data.
        :param region: The region the model is deployed in.
        :param model_key: The stable, unique key of the model.
        :param model_version: The pinned version of the model.
        """
        self._name = name
        self._parameters = parameters
        self._custom = custom
        self._region = region
        self._model_key = model_key
        self._model_version = model_version

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

    @property
    def region(self) -> Optional[str]:
        """
        The region the model is deployed in.
        """
        return self._region

    @property
    def model_key(self) -> Optional[str]:
        """
        The stable, unique key of the model (used for direct lookup; distinct from
        ``name``, which is not guaranteed unique).
        """
        return self._model_key

    @property
    def model_version(self) -> Optional[int]:
        """
        The pinned version of the model that this config variation references.
        """
        return self._model_version

    def to_dict(self) -> dict:
        """
        Render the given model config as a dictionary object.
        """
        result: Dict[str, Any] = {
            'name': self._name,
            'parameters': self._parameters,
            'custom': self._custom,
            'region': self._region,
        }
        if self._model_key:
            result['modelKey'] = self._model_key
        if self._model_version is not None:
            result['modelVersion'] = self._model_version
        return result


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
    def disabled(cls) -> Self:
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

    Instances are always created by the SDK client, which injects a real
    ``create_tracker`` factory.  User code should never need to construct
    this directly -- use the ``*Default`` variants for default values.

    ``create_tracker`` is a zero-argument callable: each invocation creates a
    new tracker for a fresh AI run. Each call mints a new ``runId`` (a UUIDv4)
    that LaunchDarkly uses to correlate the run's events in metrics views.
    Call it once per AI run; metrics from different ``runId``s cannot be
    combined.
    """
    key: str
    enabled: bool
    #: Factory that creates a new tracker for a fresh AI run. Each call mints a
    #: new ``runId`` (a UUIDv4) so LaunchDarkly can correlate the run's events
    #: in metrics views. Call this once per AI run; metrics from different
    #: ``runId``s cannot be combined.
    create_tracker: Callable[[], Any]
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None

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
    tools: Optional[Dict[str, 'LDTool']] = None

    def to_dict(self) -> dict:
        """
        Render the given default values as an AICompletionConfigDefault-compatible dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        if self.tools is not None:
            result['tools'] = {k: v.to_dict() for k, v in self.tools.items()}
        return result


@dataclass(frozen=True)
class AICompletionConfig(AIConfig):
    """
    Completion AI Config (default mode).
    """
    evaluator: 'Evaluator' = field(kw_only=True, repr=False, compare=False, hash=False)
    messages: Optional[List[LDMessage]] = None
    judge_configuration: Optional[JudgeConfiguration] = None
    tools: Optional[Dict[str, 'LDTool']] = None

    def to_dict(self) -> dict:
        """
        Render the given completion config as a dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        if self.tools is not None:
            result['tools'] = {k: v.to_dict() for k, v in self.tools.items()}
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
    tools: Optional[Dict[str, 'LDTool']] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config default as a dictionary object.
        """
        result = self._base_to_dict()
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        if self.tools is not None:
            result['tools'] = {k: v.to_dict() for k, v in self.tools.items()}
        return result


@dataclass(frozen=True)
class AIAgentConfig(AIConfig):
    """
    Agent-specific AI Config with instructions.
    """
    evaluator: 'Evaluator' = field(kw_only=True, repr=False, compare=False, hash=False)
    instructions: Optional[str] = None
    judge_configuration: Optional[JudgeConfiguration] = None
    tools: Optional[Dict[str, 'LDTool']] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config as a dictionary object.
        """
        result = self._base_to_dict()
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        if self.tools is not None:
            result['tools'] = {k: v.to_dict() for k, v in self.tools.items()}
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
    evaluation_metric_key: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Render the given judge config default as a dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        result['evaluationMetricKey'] = self.evaluation_metric_key
        return result


@dataclass(frozen=True)
class AIJudgeConfig(AIConfig):
    """
    Judge-specific AI Config with required evaluation metric key.
    """
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
