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

    def __init__(self, id: str, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, attributes: dict = {}):
        """
        :param id: The ID of the model.
        :param temperature: Turning parameter for randomness versus determinism. Exact effect will be determined by the model.
        :param max_tokens: The maximum number of tokens.
        :param attributes: Additional model-specific attributes.
        """
        self._id = id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._attributes = attributes

    @property
    def id(self) -> str:
        """
        The ID of the model.
        """
        return self._id

    @property
    def temperature(self) -> Optional[float]:
        """"
        Turning parameter for randomness versus determinism. Exact effect will be determined by the model.
        """
        return self._temperature

    @property
    def max_tokens(self) -> Optional[int]:
        """
        The maximum number of tokens.
        """

        return self._max_tokens

    def get_attribute(self, key: str) -> Any:
        """
        Retrieve model-specific attributes.

        Accessing a named, typed attribute (e.g. id) will result in the call
        being delegated to the appropriate property.
        """
        if key == 'id':
            return self.id
        if key == 'temperature':
            return self.temperature
        if key == 'maxTokens':
            return self.max_tokens

        return self._attributes.get(key)


class AIConfig:
    def __init__(self, tracker: LDAIConfigTracker, enabled: bool, model: Optional[ModelConfig], prompt: Optional[List[LDMessage]]):
        self.tracker = tracker
        self.enabled = enabled
        self.model = model
        self.prompt = prompt


class LDAIClient:
    """The LaunchDarkly AI SDK client object."""

    def __init__(self, client: LDClient):
        self.client = client

    def model_config(
        self,
        key: str,
        context: Context,
        default_value: AIConfig,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AIConfig:
        """
        Get the value of a model configuration asynchronously.

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

        prompt = None
        if 'prompt' in variation and isinstance(variation['prompt'], list) and all(
            isinstance(entry, dict) for entry in variation['prompt']
        ):
            prompt = [
                LDMessage(
                    role=entry['role'],
                    content=self.__interpolate_template(
                        entry['content'], all_variables
                    ),
                )
                for entry in variation['prompt']
            ]

        model = None
        if 'model' in variation:
            model = ModelConfig(
                id=variation['model']['modelId'],
                temperature=variation['model'].get('temperature'),
                max_tokens=variation['model'].get('maxTokens'),
                attributes=variation['model'],
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
            prompt=prompt
        )

    def __interpolate_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Interpolate the template with the given variables.

        :template: The template string.
        :variables: The variables to interpolate into the template.
        :return: The interpolated string.
        """
        return chevron.render(template, variables)
