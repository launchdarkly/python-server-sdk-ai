import os
from typing import Optional

from ldai.models import AIConfigKind
from ldai.providers import AIProvider
from ldai_openai.openai_model_runner import OpenAIModelRunner
from openai import AsyncOpenAI


class OpenAIRunnerFactory(AIProvider):
    """
    OpenAI connector for the LaunchDarkly AI SDK.

    Acts as a per-provider factory. Instantiate with no arguments to read
    credentials from the environment (``OPENAI_API_KEY``), then call
    ``create_model(config)`` to obtain a configured ``OpenAIModelRunner``.

    For advanced use, pass an explicit ``AsyncOpenAI`` client.
    """

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the OpenAI connector.

        :param client: An AsyncOpenAI client instance (created from env if omitted)
        """
        self._client = client if client is not None else AsyncOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        )

    def create_model(self, config: AIConfigKind) -> OpenAIModelRunner:
        """
        Create a configured OpenAIModelRunner for the given AI config.

        Reuses the underlying AsyncOpenAI client so connection pooling is preserved.

        :param config: The LaunchDarkly AI configuration
        :return: OpenAIModelRunner ready to invoke the model
        """
        config_dict = config.to_dict()
        model_dict = config_dict.get('model') or {}
        model_name = model_dict.get('name', '')
        parameters = model_dict.get('parameters') or {}
        return OpenAIModelRunner(self._client, model_name, parameters)

    def get_client(self) -> AsyncOpenAI:
        """
        Return the underlying AsyncOpenAI client.

        :return: The AsyncOpenAI client instance
        """
        return self._client
