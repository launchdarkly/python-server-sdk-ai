from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import chevron
from ldclient import Context
from ldclient.client import LDClient

from ldai.tracker import LDAIConfigTracker


@dataclass
class LDMessage:
    role: Literal['system', 'user', 'assistant']
    content: str


@dataclass
class AIConfigData:
    model: Optional[dict]
    prompt: Optional[List[LDMessage]]


class AIConfig:
    def __init__(self, config: AIConfigData, tracker: LDAIConfigTracker, enabled: bool):
        self.config = config
        self.tracker = tracker
        self.enabled = enabled


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

        if isinstance(variation['prompt'], list) and all(
            isinstance(entry, dict) for entry in variation['prompt']
        ):
            variation['prompt'] = [
                LDMessage(
                    role=entry['role'],
                    content=self.__interpolate_template(
                        entry['content'], all_variables
                    ),
                )
                for entry in variation['prompt']
            ]

        enabled = variation.get('_ldMeta', {}).get('enabled', False)
        return AIConfig(
            config=AIConfigData(model=variation['model'], prompt=variation['prompt']),
            tracker=LDAIConfigTracker(
                self.client,
                variation.get('_ldMeta', {}).get('versionKey', ''),
                key,
                context,
            ),
            enabled=bool(enabled),
        )

    def __interpolate_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Interpolate the template with the given variables.

        :template: The template string.
        :variables: The variables to interpolate into the template.
        :return: The interpolated string.
        """
        return chevron.render(template, variables)
