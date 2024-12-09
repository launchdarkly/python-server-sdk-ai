from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import chevron
from ldclient import Context
from ldclient.client import LDClient

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


class LDAIClient:
    """The LaunchDarkly AI SDK client object."""

    def __init__(self, client: LDClient):
        self.client = client

    def config(
        self,
        key: str,
        context: Context,
        default_value: AIConfig,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AIConfig, LDAIConfigTracker]:
        """
        Get the value of a model configuration.

        :param key: The key of the model configuration.
        :param context: The context to evaluate the model configuration in.
        :param default_value: The default value of the model configuration.
        :param variables: Additional variables for the model configuration.
        :return: The value of the model configuration along with a tracker used for gathering metrics.
        """
        variation = self.client.variation(key, context, default_value.to_dict())

        all_variables = {}
        if variables:
            all_variables.update(variables)
        all_variables['ldctx'] = context.to_dict()

        messages = None
        if 'messages' in variation and isinstance(variation['messages'], list) and all(
            isinstance(entry, dict) for entry in variation['messages']
        ):
            messages = [
                LDMessage(
                    role=entry['role'],
                    content=self.__interpolate_template(
                        entry['content'], all_variables
                    ),
                )
                for entry in variation['messages']
            ]

        provider_config = None
        if 'provider' in variation and isinstance(variation['provider'], dict):
            provider = variation['provider']
            provider_config = ProviderConfig(provider.get('name', ''))

        model = None
        if 'model' in variation and isinstance(variation['model'], dict):
            parameters = variation['model'].get('parameters', None)
            custom = variation['model'].get('custom', None)
            model = ModelConfig(
                name=variation['model']['name'],
                parameters=parameters,
                custom=custom
            )

        tracker = LDAIConfigTracker(
            self.client,
            variation.get('_ldMeta', {}).get('versionKey', ''),
            key,
            context,
        )

        enabled = variation.get('_ldMeta', {}).get('enabled', False)
        config = AIConfig(
            enabled=bool(enabled),
            model=model,
            messages=messages,
            provider=provider_config,
        )

        return config, tracker

    def __interpolate_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Interpolate the template with the given variables.

        :template: The template string.
        :variables: The variables to interpolate into the template.
        :return: The interpolated string.
        """
        return chevron.render(template, variables)
