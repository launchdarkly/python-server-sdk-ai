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


@dataclass(frozen=True)
class AIConfig:
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    messages: Optional[List[LDMessage]] = None
    provider: Optional[ProviderConfig] = None

    def to_dict(self) -> dict:
        """
        Render the given default values as an AIConfig-compatible dictionary object.
        """
        return {
            '_ldMeta': {
                'enabled': self.enabled or False,
            },
            'model': self.model.to_dict() if self.model else None,
            'messages': [message.to_dict() for message in self.messages] if self.messages else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }


@dataclass(frozen=True)
class LDAIAgent:
    """
    Represents an AI agent configuration with instructions and model settings.

    An agent is similar to an AIConfig but focuses on instructions rather than messages,
    making it suitable for AI assistant/agent use cases.
    """
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None
    instructions: Optional[str] = None
    tracker: Optional[LDAIConfigTracker] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent as a dictionary object.
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
        return result


@dataclass(frozen=True)
class LDAIAgentDefaults:
    """
    Default values for AI agent configurations.

    Similar to LDAIAgent but without tracker and with optional enabled field,
    used as fallback values when agent configurations are not available.
    """
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None
    instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent defaults as a dictionary object.
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
        return result


@dataclass
class LDAIAgentConfig:
    """
    Configuration for individual agent in batch requests.

    Combines agent key with its specific default configuration and variables.
    """
    key: str
    default_value: LDAIAgentDefaults
    variables: Optional[Dict[str, Any]] = None


# Type alias for multiple agents
LDAIAgents = Dict[str, LDAIAgent]

