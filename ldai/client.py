from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import chevron
from dataclasses_json import dataclass_json
from ldclient import Context
from ldclient.client import LDClient

from ldai.tracker import LDAIConfigTracker


@dataclass_json
@dataclass
class LDMessage:
    role: Literal['system', 'user', 'assistant']
    content: str


class ModelConfig:
    """
    Configuration related to the model.
    """

    def __init__(self, id: str, parameters: Optional[Dict[str, Any]] = None, custom: Optional[Dict[str, Any]] = None):
        """
        :param id: The ID of the model.
        :param parameters: Additional model-specific parameters.
        :param custom: Additional customer provided data.
        """
        self._id = id
        self._parameters = parameters
        self._custom = custom

    @property
    def id(self) -> str:
        """
        The ID of the model.
        """
        return self._id

    def get_parameter(self, key: str) -> Any:
        """
        Retrieve model-specific parameters.

        Accessing a named, typed attribute (e.g. id) will result in the call
        being delegated to the appropriate property.
        """
        if key == 'id':
            return self.id

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


class ProviderConfig:
    """
    Configuration related to the provider.
    """

    def __init__(self, id: str):
        self._id = id

    @property
    def id(self) -> str:
        """
        The ID of the provider.
        """
        return self._id


class AIConfig:
    def __init__(self, tracker: LDAIConfigTracker, enabled: bool, model: Optional[ModelConfig], messages: Optional[List[LDMessage]], provider: Optional[ProviderConfig] = None):
        self.tracker = tracker
        self.enabled = enabled
        self.model = model
        self.messages = messages
        self.provider = provider


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
    ) -> AIConfig:
        """
        Get the value of a model configuration.

        :param key: The key of the model configuration.
        :param context: The context to evaluate the model configuration in.
        :param default_value: The default value of the model configuration.
        :param variables: Additional variables for the model configuration.
        :return: The value of the model configuration.
        """
        variation = self.client.variation(key, context, default_value)

        all_variables = {}
        if variables:
            all_variables.update(variables)
        all_variables['ldctx'] = context

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
            provider_config = ProviderConfig(provider.get('id', ''))

        model = None
        if 'model' in variation and isinstance(variation['model'], dict):
            parameters = variation['model'].get('parameters', None)
            custom = variation['model'].get('custom', None)
            model = ModelConfig(
                id=variation['model']['id'],
                parameters=parameters,
                custom=custom
            )

        enabled = variation.get('_ldMeta', {}).get('enabled', False)
        return AIConfig(
            tracker=LDAIConfigTracker(
                self.client,
                variation.get('_ldMeta', {}).get('versionKey', ''),
                key,
                context,
            ),
            enabled=bool(enabled),
            model=model,
            messages=messages,
            provider=provider_config,
        )

    def __interpolate_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Interpolate the template with the given variables.

        :template: The template string.
        :variables: The variables to interpolate into the template.
        :return: The interpolated string.
        """
        return chevron.render(template, variables)
