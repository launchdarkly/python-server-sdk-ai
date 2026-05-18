from typing import TYPE_CHECKING, Any, Optional

from ldai.models import AIConfigKind
from ldai.providers import AIProvider, ToolRegistry

from ldai_bedrock.bedrock_model_runner import BedrockModelRunner

if TYPE_CHECKING:
    from ldai_bedrock.bedrock_agent_runner import BedrockAgentRunner


class BedrockRunnerFactory(AIProvider):
    """Amazon Bedrock ``AIProvider`` implementation for the LaunchDarkly AI SDK."""

    def __init__(self, client: Any = None, region_name: Optional[str] = None):
        """
        Initialize the Bedrock connector.

        :param client: A pre-built ``bedrock-runtime`` boto3 client.  When
            omitted, one is constructed lazily on first use using the standard
            AWS credential chain.
        :param region_name: Optional region name passed through to ``boto3.client``
            when the connector creates the client itself.  Ignored when
            ``client`` is supplied.
        """
        self._client = client
        self._region_name = region_name

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3
            kwargs: dict = {}
            if self._region_name:
                kwargs['region_name'] = self._region_name
            self._client = boto3.client('bedrock-runtime', **kwargs)
        return self._client

    def get_client(self) -> Any:
        """Return the underlying ``bedrock-runtime`` boto3 client."""
        return self._get_client()

    def _extract_model_config(self, config: AIConfigKind) -> tuple:
        """
        Extract model id and parameters from an AI config.

        :param config: The LaunchDarkly AI configuration
        :return: Tuple of (model_id, parameters)
        """
        config_dict = config.to_dict()
        model_dict = config_dict.get('model') or {}
        return model_dict.get('name', ''), model_dict.get('parameters') or {}

    def create_model(self, config: AIConfigKind, multi_turn: bool = True) -> BedrockModelRunner:
        """
        Create a configured ``BedrockModelRunner`` for the given AI config.

        Reuses the underlying ``bedrock-runtime`` client so connection pooling
        is preserved across model invocations.

        :param config: The LaunchDarkly AI configuration.  ``model.name`` is
            interpreted as a Bedrock ``modelId`` (e.g.
            ``anthropic.claude-3-5-sonnet-20240620-v1:0``).
        :param multi_turn: When ``True`` (the default) the runner accumulates
            successful exchanges into its conversation history.  Pass ``False``
            to keep history fixed at the configured baseline across ``run()``
            calls.
        :return: BedrockModelRunner ready to invoke the model
        """
        model_id, parameters = self._extract_model_config(config)
        parameters = dict(parameters)
        config_messages = list(getattr(config, 'messages', None) or [])
        return BedrockModelRunner(
            self._get_client(),
            model_id,
            parameters,
            config_messages,
            multi_turn=multi_turn,
        )

    def create_agent(self, config: Any, tools: Optional[ToolRegistry] = None) -> 'BedrockAgentRunner':
        """
        CAUTION:
        This feature is experimental and should NOT be considered ready for production use.
        It may change or be removed without notice and is not subject to backwards
        compatibility guarantees.

        Create a configured ``BedrockAgentRunner`` for the given AI agent config.

        The agent runner is backed by the Strands Agents SDK (``strands-agents``),
        which is an optional dependency of this package.  Install it with::

            pip install launchdarkly-server-sdk-ai-bedrock[agents]

        :param config: The LaunchDarkly AI agent configuration
        :param tools: ToolRegistry mapping tool names to callables
        :return: BedrockAgentRunner ready to run the agent
        """
        from ldai_bedrock.bedrock_agent_runner import BedrockAgentRunner

        model_id, base_parameters = self._extract_model_config(config)
        parameters = dict(base_parameters)
        tool_definitions = parameters.pop('tools', []) or []
        instructions = (config.instructions or '') if hasattr(config, 'instructions') else ''

        return BedrockAgentRunner(
            model_id,
            parameters,
            instructions,
            tool_definitions,
            tools or {},
        )

    def create_agent_graph(
        self,
        graph_def: Any,
        tools: ToolRegistry,
    ) -> Any:
        """
        Agent graph execution is not yet supported by the Bedrock provider.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Agent graph is not yet supported by the bedrock provider'
        )
