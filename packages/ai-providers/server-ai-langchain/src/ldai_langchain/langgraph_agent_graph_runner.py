"""LangGraph agent graph runner for LaunchDarkly AI SDK."""

import time
from typing import Annotated, Any, Dict, List, Optional, Set, Tuple

from ldai import log
from ldai.agent_graph import AgentGraphDefinition, AgentGraphNode
from ldai.providers import AgentGraphResult, AgentGraphRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics

from ldai_langchain.langchain_helper import (
    build_structured_tools,
    create_langchain_model,
    extract_last_message_content,
    sum_token_usage_from_messages,
)
from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler


def _make_handoff_tool(child_key: str, description: str) -> Any:
    """
    Create a tool that transfers control to ``child_key``.

    Uses the ``@tool`` decorator with ``InjectedState`` + ``InjectedToolCallId``
    so LangGraph's ToolNode handles the ``Command`` return value correctly.
    The tool explicitly creates a ToolMessage in ``Command.update`` to satisfy
    the LangChain/OpenAI message-chain contract.
    """
    from typing import Annotated as _Annotated

    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool
    from langchain_core.tools.base import InjectedToolCallId
    from langgraph.prebuilt import InjectedState
    from langgraph.types import Command

    tool_name = f"transfer_to_{child_key.replace('-', '_')}"

    @tool(tool_name, description=description)
    def handoff(
        state: _Annotated[Any, InjectedState],  # noqa: ARG001
        tool_call_id: _Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = ToolMessage(
            content=f'Transferred to {child_key}',
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return Command(goto=child_key, update={'messages': [tool_message]})

    return handoff


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
        self._compiled: Any = None
        self._fn_name_to_config_key: Dict[str, str] = {}
        self._node_keys: Set[str] = set()

    def _ensure_compiled(self) -> None:
        """Build and cache the compiled graph if not already done."""
        if self._compiled is None:
            compiled, fn_name_to_config_key, node_keys = self._build_graph()
            self._compiled = compiled
            self._fn_name_to_config_key = fn_name_to_config_key
            self._node_keys = node_keys

    def _build_graph(self) -> Tuple[Any, Dict[str, str], Set[str]]:
        """
        Build and compile the LangGraph StateGraph from the AgentGraphDefinition.

        :return: Tuple of (compiled_graph, fn_name_to_config_key, node_keys) where
            fn_name_to_config_key maps tool function __name__ to LD config key, and
            node_keys is the set of all agent node keys in the graph.
        """
        from langchain_core.messages import SystemMessage
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode, tools_condition
        from typing_extensions import TypedDict

        class WorkflowState(TypedDict):
            messages: Annotated[List[Any], add_messages]

        agent_builder: StateGraph = StateGraph(WorkflowState)
        root_node = self._graph.root()
        root_key = root_node.get_key() if root_node else None
        tools_ref = self._tools
        graph_structure: List[str] = []
        fn_name_to_config_key: Dict[str, str] = {}
        node_keys: Set[str] = set()

        def handle_traversal(node: AgentGraphNode, ctx: dict) -> None:
            node_config = node.get_config()
            node_key = node.get_key()
            node_keys.add(node_key)
            instructions = node_config.instructions if hasattr(node_config, 'instructions') else None
            outgoing_edges = node.get_edges()

            lc_model = None
            tool_fns: list = []
            if node_config.model:
                # We send an empty tool registry to avoid binding tools to the model.
                lc_model = create_langchain_model(node_config)

                tool_fns = build_structured_tools(node_config, tools_ref)

                # Map tool name -> LD config key for callback attribution.
                # build_structured_tools returns StructuredTool instances with tool.name set
                # to the LD config key, so tool.name IS the config key.
                for tool in tool_fns:
                    tool_name = getattr(tool, 'name', None)
                    if tool_name:
                        fn_name_to_config_key[tool_name] = tool_name

            # For nodes with multiple children, create a handoff tool per child so the
            # LLM decides which agent to route to.  Uses Command(goto=child_key) so
            # LangGraph routes to the target without looping back here.
            handoff_fns: list = []
            if lc_model and len(outgoing_edges) > 1:
                for edge in outgoing_edges:
                    child_node = self._graph.get_node(edge.target_config)
                    description = (
                        (edge.handoff or {}).get('description')
                        or (
                            child_node.get_config().instructions[:120]
                            if child_node and child_node.get_config().instructions
                            else None
                        )
                        or f"Transfer control to {edge.target_config}"
                    )
                    handoff_fns.append(_make_handoff_tool(edge.target_config, description))

            all_tools = tool_fns + handoff_fns
            model: Any
            if lc_model and all_tools:
                # When handoff tools are present, disable parallel tool calls so the LLM
                # picks exactly one destination rather than routing to multiple children.
                bind_kwargs: Dict[str, Any] = {'parallel_tool_calls': False} if handoff_fns else {}
                model = lc_model.bind_tools(all_tools, **bind_kwargs)
            else:
                model = lc_model

            def make_node_fn(bound_model: Any, node_instructions: Any, nk: str):
                async def invoke(state: WorkflowState) -> dict:
                    if not bound_model:
                        return {'messages': []}
                    msgs = list(state['messages'])
                    if node_instructions:
                        msgs = [SystemMessage(content=node_instructions)] + msgs
                    response = await bound_model.ainvoke(msgs)
                    return {'messages': [response]}

                invoke.__name__ = nk
                return invoke

            invoke_fn = make_node_fn(model, instructions, node_key)
            agent_builder.add_node(node_key, invoke_fn)

            if node_key == root_key:
                agent_builder.add_edge(START, node_key)

            # Collect node info for graph structure log
            tool_names = [str(getattr(t, 'name', None) or getattr(t, '__name__', t)) for t in tool_fns]
            edge_targets = [e.target_config for e in outgoing_edges]
            node_desc = node_key
            if tool_names:
                node_desc += f"[tools:{','.join(tool_names)}]"
            if handoff_fns:
                node_desc += f"[handoff:{','.join(edge_targets)}]"
            elif edge_targets:
                node_desc += f"→{','.join(edge_targets)}"
            else:
                node_desc += "(terminal)"
            graph_structure.append(node_desc)

            if all_tools:
                tools_node_key = f"{node_key}__tools"
                agent_builder.add_node(tools_node_key, ToolNode(all_tools))

                if not handoff_fns:
                    # No handoff tools: standard loop-back after tool execution.
                    after_loop = outgoing_edges[0].target_config if outgoing_edges else END
                    if len(outgoing_edges) > 1:
                        log.warning(
                            f"Node '{node_key}' has {len(outgoing_edges)} outgoing edges but no handoff "
                            "tools; only the first edge will be used after the tool loop. "
                            "Use handoff tools for multi-child routing."
                        )
                    agent_builder.add_edge(tools_node_key, node_key)
                    agent_builder.add_conditional_edges(
                        node_key,
                        tools_condition,
                        {"tools": tools_node_key, END: after_loop},
                    )
                elif not tool_fns:
                    # Only handoff tools: no loop-back needed.
                    # Command(goto=child_key) handles routing to the target.
                    agent_builder.add_conditional_edges(
                        node_key,
                        tools_condition,
                        {"tools": tools_node_key, END: END},
                    )
                else:
                    # Both functional and handoff tools. A static loop-back edge would
                    # fan-out with Command(goto=child_key) from handoff tools, so use a
                    # conditional edge that only loops back for functional tool results.
                    handoff_names_set = frozenset(getattr(t, 'name', '') for t in handoff_fns)

                    def make_after_tools_router(parent_key: str, ht_names: frozenset):
                        def route(state: WorkflowState) -> str:
                            msgs = state['messages']
                            if msgs:
                                last = msgs[-1]
                                if hasattr(last, 'name') and last.name in ht_names:
                                    return END
                            return parent_key
                        return route

                    agent_builder.add_conditional_edges(
                        tools_node_key,
                        make_after_tools_router(node_key, handoff_names_set),
                        {node_key: node_key, END: END},
                    )
                    agent_builder.add_conditional_edges(
                        node_key,
                        tools_condition,
                        {"tools": tools_node_key, END: END},
                    )
            else:
                if node.is_terminal():
                    agent_builder.add_edge(node_key, END)
                for edge in outgoing_edges:
                    agent_builder.add_edge(node_key, edge.target_config)

            return None

        self._graph.traverse(fn=handle_traversal)

        graph_key_str = self._graph._agent_graph.key or 'unknown'
        log.debug(
            f"LangGraphAgentGraphRunner: graph='{graph_key_str}', root='{root_key}', "
            f"structure: {' | '.join(graph_structure)}"
        )

        compiled = agent_builder.compile()
        return compiled, fn_name_to_config_key, node_keys

    async def run(self, input: Any) -> AgentGraphResult:
        """
        Run the agent graph with the given input.

        Builds a LangGraph StateGraph from the AgentGraphDefinition, compiles
        it, and invokes it. Uses a LangChain callback handler to collect
        per-node metrics, then flushes them to LaunchDarkly trackers.

        :param input: The string prompt to send to the agent graph
        :return: AgentGraphResult with the final output and metrics
        """
        tracker = self._graph.create_tracker() if self._graph.create_tracker is not None else None
        start_ns = time.perf_counter_ns()

        try:
            from langchain_core.messages import HumanMessage

            self._ensure_compiled()
            handler = LDMetricsCallbackHandler(self._node_keys, self._fn_name_to_config_key)

            result = await self._compiled.ainvoke(  # type: ignore[call-overload]
                {'messages': [HumanMessage(content=str(input))]},
                config={'callbacks': [handler], 'recursion_limit': 25},
            )

            duration = (time.perf_counter_ns() - start_ns) // 1_000_000
            messages = result.get('messages', [])
            output = extract_last_message_content(messages)

            # Flush per-node metrics to LD trackers
            handler.flush(self._graph)

            # Graph-level metrics
            if tracker:
                tracker.track_path(handler.path)
                tracker.track_duration(duration)
                tracker.track_invocation_success()
                tracker.track_total_tokens(sum_token_usage_from_messages(messages))

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
                tracker.track_duration(duration)
                tracker.track_invocation_failure()
            return AgentGraphResult(
                output='',
                raw=None,
                metrics=LDAIMetrics(success=False),
            )
