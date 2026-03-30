from typing import Any

from ldai.models import AIConfigKind
from ldai.providers import AIProvider, ToolRegistry

from ldai_langchain.langchain_helper import create_langchain_model
from ldai_langchain.langchain_model_runner import LangChainModelRunner


class LangChainRunnerFactory(AIProvider):
    """LangChain ``AIProvider`` implementation for the LaunchDarkly AI SDK."""

    def create_agent_graph(self, graph_def: Any, tools: ToolRegistry) -> Any:
        """
        Create a configured LangGraphAgentGraphRunner for the given graph definition.

        :param graph_def: The AgentGraphDefinition to execute
        :param tools: Registry mapping tool names to callables (langchain-compatible)
        :return: LangGraphAgentGraphRunner ready to execute the graph
        """
        from ldai_langchain.langgraph_agent_graph_runner import (
            LangGraphAgentGraphRunner,
        )
        return LangGraphAgentGraphRunner(graph_def, tools)

    def create_model(self, config: AIConfigKind) -> LangChainModelRunner:
        """
        Create a configured LangChainModelRunner for the given AI config.

        :param config: The LaunchDarkly AI configuration
        :return: LangChainModelRunner ready to invoke the model
        """
        llm = create_langchain_model(config)
        return LangChainModelRunner(llm)
