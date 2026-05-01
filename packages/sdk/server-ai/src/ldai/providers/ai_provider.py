from abc import ABC
from typing import Any, Optional

from ldai import log
from ldai.providers.types import ToolRegistry


class AIProvider(ABC):
    """
    Abstract base class for AI providers.

    An AIProvider is a per-provider factory: it is instantiated once per provider
    (with no arguments — credentials are read from environment variables) and is
    responsible for constructing focused runtime capability objects via
    create_model(), create_agent(), and create_agent_graph().
    """

    def create_model(self, config: Any) -> Optional[Any]:
        """
        Create a configured model executor for the given AI config.

        Default implementation warns. Provider implementations should override this method.

        :param config: The LaunchDarkly AI configuration
        :return: Configured model runner instance, or None if unsupported
        """
        log.warning('create_model not implemented by this provider')
        return None

    def create_agent(self, config: Any, tools: Optional[ToolRegistry] = None) -> Optional[Any]:
        """
        CAUTION:
        This feature is experimental and should NOT be considered ready for production use.
        It may change or be removed without notice and is not subject to backwards
        compatibility guarantees.

        Create a configured agent executor for the given AI config and tool registry.

        Default implementation warns. Provider implementations should override this method.

        :param config: The LaunchDarkly AI agent configuration
        :param tools: Tool registry mapping tool names to callables
        :return: AgentExecutor instance, or None if unsupported
        """
        log.warning('create_agent not implemented by this provider')
        return None

    def create_agent_graph(
        self,
        graph_def: Any,
        tools: Any,
    ) -> Optional[Any]:
        """
        CAUTION:
        This feature is experimental and should NOT be considered ready for production use.
        It may change or be removed without notice and is not subject to backwards
        compatibility guarantees.

        Create a configured agent graph executor for the given graph definition and tools.

        Default implementation warns. Provider implementations should override this method.

        :param graph_def: The agent graph definition
        :param tools: Tool registry mapping tool names to callables
        :return: AgentGraphExecutor instance, or None if unsupported
        """
        log.warning('create_agent_graph not implemented by this provider')
        return None
