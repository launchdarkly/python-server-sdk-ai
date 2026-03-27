from typing import Any, Optional

from langchain.agents import create_agent as lc_create_agent
from ldai.models import AIConfigKind
from ldai.providers import AIProvider, ToolRegistry

from ldai_langchain.langchain_agent_runner import LangChainAgentRunner
from ldai_langchain.langchain_helper import (
    build_structured_tools,
    create_langchain_model,
)
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

    def create_agent(self, config: Any, tools: Optional[ToolRegistry] = None) -> LangChainAgentRunner:
        """
        Create a configured LangChainAgentRunner for the given AI agent config.

        :param config: The LaunchDarkly AI agent configuration
        :param tools: ToolRegistry mapping tool names to callables
        :return: LangChainAgentRunner ready to run the agent
        """
        instructions = (config.instructions or '') if hasattr(config, 'instructions') else ''
        llm = create_langchain_model(config)
        lc_tools = build_structured_tools(config, tools or {})

        agent = lc_create_agent(
            llm,
            tools=lc_tools or None,
            system_prompt=instructions or None,
        )
        return LangChainAgentRunner(agent)
