from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from ldai.tracker import LDAIConfigTracker


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


@dataclass(frozen=True)
class JudgeConfiguration:
    """
    Configuration for judge attachment to AI Configs.
    """
    judges: List[Judge]

    def to_dict(self) -> dict:
        """
        Render the judge configuration as a dictionary object.
        """
        return {
            'judges': [judge.to_dict() for judge in self.judges],
        }


# ============================================================================
# Completion Config Types
# ============================================================================

@dataclass(frozen=True)
class AICompletionConfigDefault:
    """
    Default Completion AI Config (default mode).
    """
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    messages: Optional[List[LDMessage]] = None
    provider: Optional[ProviderConfig] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> dict:
        """
        Render the given default values as an AICompletionConfigDefault-compatible dictionary object.
        """
        result = {
            '_ldMeta': {
                'enabled': self.enabled or False,
            },
            'model': self.model.to_dict() if self.model else None,
            'messages': [message.to_dict() for message in self.messages] if self.messages else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


@dataclass(frozen=True)
class AICompletionConfig:
    """
    Completion AI Config (default mode).
    """
    enabled: bool
    model: Optional[ModelConfig] = None
    messages: Optional[List[LDMessage]] = None
    provider: Optional[ProviderConfig] = None
    tracker: Optional[LDAIConfigTracker] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> dict:
        """
        Render the given completion config as a dictionary object.
        """
        result = {
            '_ldMeta': {
                'enabled': self.enabled,
            },
            'model': self.model.to_dict() if self.model else None,
            'messages': [message.to_dict() for message in self.messages] if self.messages else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


# ============================================================================
# Agent Config Types
# ============================================================================


@dataclass(frozen=True)
class AIAgentConfigDefault:
    """
    Default Agent-specific AI Config with instructions.
    """
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None
    instructions: Optional[str] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config default as a dictionary object.
        """
        result: Dict[str, Any] = {
            '_ldMeta': {
                'enabled': self.enabled or False,
            },
            'model': self.model.to_dict() if self.model else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


@dataclass(frozen=True)
class AIAgentConfig:
    """
    Agent-specific AI Config with instructions.
    """
    enabled: bool
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None
    instructions: Optional[str] = None
    tracker: Optional[LDAIConfigTracker] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config as a dictionary object.
        """
        result: Dict[str, Any] = {
            '_ldMeta': {
                'enabled': self.enabled,
            },
            'model': self.model.to_dict() if self.model else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


# ============================================================================
# Judge Config Types
# ============================================================================

@dataclass(frozen=True)
class AIJudgeConfigDefault:
    """
    Default Judge-specific AI Config with required evaluation metric key.
    """
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    messages: Optional[List[LDMessage]] = None
    provider: Optional[ProviderConfig] = None
    evaluation_metric_keys: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """
        Render the given judge config default as a dictionary object.
        """
        result = {
            '_ldMeta': {
                'enabled': self.enabled or False,
            },
            'model': self.model.to_dict() if self.model else None,
            'messages': [message.to_dict() for message in self.messages] if self.messages else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }
        if self.evaluation_metric_keys is not None:
            result['evaluationMetricKeys'] = self.evaluation_metric_keys
        return result


@dataclass(frozen=True)
class AIJudgeConfig:
    """
    Judge-specific AI Config with required evaluation metric key.
    """
    enabled: bool
    evaluation_metric_keys: List[str]
    model: Optional[ModelConfig] = None
    messages: Optional[List[LDMessage]] = None
    provider: Optional[ProviderConfig] = None
    tracker: Optional[LDAIConfigTracker] = None

    def to_dict(self) -> dict:
        """
        Render the given judge config as a dictionary object.
        """
        result = {
            '_ldMeta': {
                'enabled': self.enabled,
            },
            'evaluationMetricKeys': self.evaluation_metric_keys,
            'model': self.model.to_dict() if self.model else None,
            'messages': [message.to_dict() for message in self.messages] if self.messages else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }
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
    default_value: AIAgentConfigDefault
    variables: Optional[Dict[str, Any]] = None


# Type alias for multiple agents
AIAgents = Dict[str, AIAgentConfig]

