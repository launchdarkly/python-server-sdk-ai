import os
from typing import TYPE_CHECKING, Any, Optional

from ldai.models import AIConfigKind
from ldai.providers import AIProvider, ToolRegistry
from openai import AsyncOpenAI

from ldai_openai.openai_model_runner import OpenAIModelRunner

if TYPE_CHECKING:
    from ldai_openai.openai_agent_runner import OpenAIAgentRunner


class OpenAIRunnerFactory(AIProvider):
    """OpenAI ``AIProvider`` implementation for the LaunchDarkly AI SDK."""

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the OpenAI connector.

        :param client: An AsyncOpenAI client instance (created from env if omitted)
        """
        self._client = client if client is not None else AsyncOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        )

    def _extract_model_config(self, config: AIConfigKind) -> tuple:
        """
        Extract model name and parameters from an AI config.

        :param config: The LaunchDarkly AI configuration
        :return: Tuple of (model_name, parameters)
        """
        config_dict = config.to_dict()
        model_dict = config_dict.get('model') or {}
        return model_dict.get('name', ''), model_dict.get('parameters') or {}

    def create_model(self, config: AIConfigKind) -> OpenAIModelRunner:
        """
        Create a configured OpenAIModelRunner for the given AI config.

        Reuses the underlying AsyncOpenAI client so connection pooling is preserved.

        :param config: The LaunchDarkly AI configuration
        :return: OpenAIModelRunner ready to invoke the model
        """
        model_name, parameters = self._extract_model_config(config)
        return OpenAIModelRunner(self._client, model_name, parameters)

    def create_agent_graph(self, graph_def: Any, tools: ToolRegistry) -> Any:
        """
        Create a configured OpenAIAgentGraphRunner for the given graph definition.

        :param graph_def: The AgentGraphDefinition to execute
        :param tools: Registry mapping tool names to callables
        :return: OpenAIAgentGraphRunner ready to execute the graph
        """
        from ldai_openai.openai_agent_graph_runner import OpenAIAgentGraphRunner
        return OpenAIAgentGraphRunner(graph_def, tools)

    def create_agent(self, config: Any, tools: Optional[ToolRegistry] = None) -> 'OpenAIAgentRunner':
        """
        Create a configured OpenAIAgentRunner for the given AI agent config.

        :param config: The LaunchDarkly AI agent configuration
        :param tools: ToolRegistry mapping tool names to callables
        :return: OpenAIAgentRunner ready to run the agent
        """
        from ldai_openai.openai_agent_runner import OpenAIAgentRunner

        model_name, base_parameters = self._extract_model_config(config)
        parameters = dict(base_parameters)
        tool_definitions = parameters.pop('tools', []) or []
        instructions = (config.instructions or '') if hasattr(config, 'instructions') else ''

        return OpenAIAgentRunner(
            model_name,
            parameters,
            instructions,
            tool_definitions,
            tools or {},
        )

    def get_client(self) -> AsyncOpenAI:
        """
        Return the underlying AsyncOpenAI client.

        :return: The AsyncOpenAI client instance
        """
        return self._client
