"""Abstract base class for AI connectors."""

from abc import ABC
from typing import Any, Dict, List, Optional

from ldai import log
from ldai.models import LDMessage
from ldai.providers.types import ChatResponse, StructuredResponse


class AIProvider(ABC):
    """
    Abstract base class for AI provider connectors.

    An AIProvider is a per-provider factory: it is instantiated once per provider
    (with no arguments — credentials are read from environment variables) and is
    responsible for constructing focused runtime capability objects via
    create_model(), create_agent(), and create_agent_graph().

    The invoke_model() / invoke_structured_model() methods remain on this base
    class for compatibility and will migrate to ModelExecutor in PR 2.
    """

    async def invoke_model(self, messages: List[LDMessage]) -> ChatResponse:
        """
        Invoke the chat model with an array of messages.

        Default implementation takes no action and returns a placeholder response.
        Connector implementations should override this method.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ChatResponse containing the model's response
        """
        log.warn('invoke_model not implemented by this connector')

        from ldai.models import LDMessage
        from ldai.providers.types import LDAIMetrics

        return ChatResponse(
            message=LDMessage(role='assistant', content=''),
            metrics=LDAIMetrics(success=False, usage=None),
        )

    async def invoke_structured_model(
        self,
        messages: List[LDMessage],
        response_structure: Dict[str, Any],
    ) -> StructuredResponse:
        """
        Invoke the chat model with structured output support.

        Default implementation takes no action and returns a placeholder response.
        Connector implementations should override this method.

        :param messages: Array of LDMessage objects representing the conversation
        :param response_structure: Dictionary of output configurations keyed by output name
        :return: StructuredResponse containing the structured data
        """
        log.warn('invoke_structured_model not implemented by this connector')

        from ldai.providers.types import LDAIMetrics

        return StructuredResponse(
            data={},
            raw_response='',
            metrics=LDAIMetrics(success=False, usage=None),
        )

    def create_model(self, config: Any) -> Optional['AIProvider']:
        """
        Create a configured model executor for the given AI config.

        Default implementation warns. Provider connectors should override this method.

        :param config: The LaunchDarkly AI configuration
        :return: Configured AIProvider instance, or None if unsupported
        """
        log.warn('create_model not implemented by this connector')
        return None

    def create_agent(self, config: Any, tools: Any) -> Optional[Any]:
        """
        Create a configured agent executor for the given AI config and tool registry.

        Default implementation warns. Provider connectors should override this method.

        :param config: The LaunchDarkly AI agent configuration
        :param tools: Tool registry mapping tool names to callables
        :return: AgentExecutor instance, or None if unsupported
        """
        log.warn('create_agent not implemented by this connector')
        return None

    def create_agent_graph(self, graph_def: Any, tools: Any) -> Optional[Any]:
        """
        Create a configured agent graph executor for the given graph definition and tools.

        Default implementation warns. Provider connectors should override this method.

        :param graph_def: The agent graph definition
        :param tools: Tool registry mapping tool names to callables
        :return: AgentGraphExecutor instance, or None if unsupported
        """
        log.warn('create_agent_graph not implemented by this connector')
        return None

