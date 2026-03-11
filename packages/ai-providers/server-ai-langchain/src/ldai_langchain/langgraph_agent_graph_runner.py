"""LangGraph agent graph runner for LaunchDarkly AI SDK."""

import operator
import time
from typing import Annotated, Any, List

from ldai.agent_graph import AgentGraphDefinition, AgentGraphNode
from ldai.providers.types import LDAIMetrics
from ldai.runners.agent_graph_runner import AgentGraphRunner
from ldai.runners.types import AgentGraphResult, ToolRegistry


class LangGraphAgentGraphRunner(AgentGraphRunner):
    """
    AgentGraphRunner implementation for LangGraph.

    Builds a LangGraph StateGraph from an AgentGraphDefinition and
    ToolRegistry via traverse(), compiles it, and executes it with
    ainvoke(). Auto-tracks latency and invocation success/failure via
    the graph's AIGraphTracker.

    Requires ``langgraph`` to be installed.
    """

    def __init__(self, graph: AgentGraphDefinition, tools: ToolRegistry):
        """
        Initialize the runner.

        :param graph: The AgentGraphDefinition to execute
        :param tools: Registry mapping tool names to callables (langchain-compatible)
        """
        self._graph = graph
        self._tools = tools

    async def run(self, input: Any) -> AgentGraphResult:
        """
        Run the agent graph with the given input.

        Builds a LangGraph StateGraph from the AgentGraphDefinition, compiles
        it, and invokes it. Tracks latency and invocation success/failure.

        :param input: The string prompt to send to the agent graph
        :return: AgentGraphResult with the final output and metrics
        """
        tracker = self._graph.get_tracker()
        start_time = time.time()
        try:
            try:
                from langchain.chat_models import init_chat_model
                from langchain_core.messages import AnyMessage, HumanMessage
                from langgraph.graph import END, START, StateGraph
                from typing_extensions import TypedDict
            except ImportError as exc:
                raise ImportError(
                    "langgraph is required for LangGraphAgentGraphRunner. "
                    "Install it with: pip install langgraph"
                ) from exc

            class WorkflowState(TypedDict):
                messages: Annotated[List[AnyMessage], operator.add]

            agent_builder: StateGraph = StateGraph(WorkflowState)
            root_node = self._graph.root()
            root_key = root_node.get_key() if root_node else None
            tools_ref = self._tools

            def handle_traversal(node: AgentGraphNode, ctx: dict) -> None:
                node_config = node.get_config()
                node_key = node.get_key()

                model = None
                if node_config.model:
                    lc_model = init_chat_model(model=node_config.model.name)
                    tool_defs = node_config.model.get_parameter('tools') or []
                    tool_fns = [
                        tools_ref[t.get('name', '')]
                        for t in tool_defs
                        if t.get('name', '') in tools_ref
                    ]
                    if tool_fns:
                        lc_model = lc_model.bind_tools(tool_fns)
                    model = lc_model

                def invoke(state: WorkflowState) -> WorkflowState:
                    if model:
                        response = model.invoke(state['messages'])
                        return {'messages': [response]}
                    return state

                invoke.__name__ = node_key

                agent_builder.add_node(name=node_key, node=invoke)

                if node_key == root_key:
                    agent_builder.add_edge(START, node_key)

                if node.is_terminal():
                    agent_builder.add_edge(node_key, END)

                for edge in node.get_edges():
                    agent_builder.add_edge(node_key, edge.target_config)

                return None

            self._graph.traverse(fn=handle_traversal)
            compiled = agent_builder.compile()

            result = await compiled.ainvoke(
                {'messages': [HumanMessage(content=str(input))]}
            )
            duration = int((time.time() - start_time) * 1000)

            output = ''
            messages = result.get('messages', [])
            if messages:
                last = messages[-1]
                if hasattr(last, 'content'):
                    output = str(last.content)

            if tracker:
                tracker.track_latency(duration)
                tracker.track_invocation_success()

            return AgentGraphResult(
                output=output,
                raw=result,
                metrics=LDAIMetrics(success=True),
            )
        except Exception:
            duration = int((time.time() - start_time) * 1000)
            if tracker:
                tracker.track_latency(duration)
                tracker.track_invocation_failure()
            return AgentGraphResult(
                output='',
                raw=None,
                metrics=LDAIMetrics(success=False),
            )
