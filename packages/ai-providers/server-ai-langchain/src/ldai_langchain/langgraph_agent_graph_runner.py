import operator
import time
from typing import Annotated, Any, List

from ldai import log
from ldai.agent_graph import AgentGraphDefinition, AgentGraphNode
from ldai.providers import AgentGraphResult, AgentGraphRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics

from ldai_langchain.langchain_helper import (
    build_tools,
    create_langchain_model,
    extract_last_message_content,
    get_ai_metrics_from_response,
    get_ai_usage_from_response,
    get_tool_calls_from_response,
    sum_token_usage_from_messages,
)


class LangGraphAgentGraphRunner(AgentGraphRunner):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use. 
    It may change or be removed without notice and is not subject to backwards 
    compatibility guarantees.

    AgentGraphRunner implementation for LangGraph.

    Compiles and runs the agent graph with LangGraph and automatically records
    graph- and node-level AI metric data to the LaunchDarkly trackers on the
    graph definition and each node.

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
        start_ns = time.perf_counter_ns()
        try:
            from langchain_core.messages import AnyMessage, HumanMessage
            from langgraph.graph import END, START, StateGraph
            from typing_extensions import TypedDict

            class WorkflowState(TypedDict):
                messages: Annotated[List[Any], operator.add]

            agent_builder: StateGraph = StateGraph(WorkflowState)
            root_node = self._graph.root()
            root_key = root_node.get_key() if root_node else None
            tools_ref = self._tools
            exec_path: List[str] = []

            def handle_traversal(node: AgentGraphNode, ctx: dict) -> None:
                node_config = node.get_config()
                node_key = node.get_key()
                node_tracker = node_config.tracker

                model = None
                if node_config.model:
                    lc_model = create_langchain_model(node_config)
                    tool_fns = build_tools(node_config, tools_ref)
                    model = lc_model.bind_tools(tool_fns) if tool_fns else lc_model

                def invoke(state: WorkflowState) -> WorkflowState:
                    exec_path.append(node_key)
                    if not model:
                        return {'messages': []}
                    gk = tracker.graph_key if tracker is not None else None
                    if node_tracker:
                        response = node_tracker.track_metrics_of(
                            lambda: model.invoke(state['messages']),
                            get_ai_metrics_from_response,
                            graph_key=gk,
                        )
                        node_tracker.track_tool_calls(
                            get_tool_calls_from_response(response),
                            graph_key=tracker.graph_key if tracker is not None else None,
                        )
                    else:
                        response = model.invoke(state['messages'])

                    return {'messages': [response]}

                invoke.__name__ = node_key

                agent_builder.add_node(node_key, invoke)

                if node_key == root_key:
                    agent_builder.add_edge(START, node_key)

                if node.is_terminal():
                    agent_builder.add_edge(node_key, END)

                for edge in node.get_edges():
                    agent_builder.add_edge(node_key, edge.target_config)

                return None

            self._graph.traverse(fn=handle_traversal)
            compiled = agent_builder.compile()

            result = await compiled.ainvoke(  # type: ignore[call-overload]
                {'messages': [HumanMessage(content=str(input))]}
            )
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000

            messages = result.get('messages', [])
            output = extract_last_message_content(messages)

            if tracker:
                tracker.track_path(exec_path)
                tracker.track_latency(duration)
                tracker.track_invocation_success()
                tracker.track_total_tokens(
                    sum_token_usage_from_messages(messages)
                )

            return AgentGraphResult(
                output=output,
                raw=result,
                metrics=LDAIMetrics(success=True),
            )
        except Exception as exc:
            if isinstance(exc, ImportError):
                log.warning(
                    "langgraph is required for LangGraphAgentGraphRunner. "
                    "Install it with: pip install langgraph"
                )
            else:
                log.warning(f'LangGraphAgentGraphRunner run failed: {exc}')
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000
            if tracker:
                tracker.track_latency(duration)
                tracker.track_invocation_failure()
            return AgentGraphResult(
                output='',
                raw=None,
                metrics=LDAIMetrics(success=False),
            )
