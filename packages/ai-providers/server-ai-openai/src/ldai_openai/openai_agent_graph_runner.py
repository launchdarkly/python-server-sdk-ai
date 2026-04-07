import logging
import re
import time
from typing import Any, Dict, List, Optional

from ldai import log
from ldai.agent_graph import AgentGraphDefinition, AgentGraphNode
from ldai.providers import AgentGraphResult, AgentGraphRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage

from ldai_openai.openai_helper import (
    extract_usage_from_request_entry,
    get_ai_usage_from_response,
    get_tool_calls_from_run_items,
    is_agent_tool_instance,
    registry_value_to_agent_tool,
)

_log = logging.getLogger("ldai_openai.openai_agent_graph_runner")


def _sanitize_agent_name(key: str) -> str:
    """Replace characters invalid for OpenAI function names with underscores."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', key)


class _RunState:
    """Mutable state shared across handoff and tool callbacks during a single run."""

    def __init__(self, last_handoff_ns: int, last_node_key: str) -> None:
        self.last_handoff_ns = last_handoff_ns
        self.last_node_key = last_node_key


def _make_span_hooks(RunHooks: Any, graph: AgentGraphDefinition, name_map: Dict[str, str]) -> Any:
    """
    Build a RunHooks subclass at call time (after ``agents`` is imported) so the
    class is a genuine subclass of RunHooks, satisfying the SDK's isinstance check.

    Uses ``tracer.start_span()`` (not ``start_as_current_span``) so no ContextVar
    token is created — OTel ContextVars cannot cross the async-task boundaries that
    the OpenAI Agents SDK uses between hooks and agent execution.
    """
    open_spans: Dict[str, Any] = {}

    class _LDAIAgentSpanHooks(RunHooks):  # type: ignore[misc]
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            from ldai.observe import _OTEL_AVAILABLE, _otel_context, _otel_trace
            node_key = name_map.get(agent.name, agent.name)
            _log.info("[ldai:openai] agent_start node_key=%r", node_key)
            # End any span from the previous agent — on_agent_end only fires for the
            # final agent; intermediate agents hand off without triggering on_agent_end.
            for name in list(open_spans):
                open_spans.pop(name).end()
            if not _OTEL_AVAILABLE:
                return
            node = graph.get_node(node_key)
            if node is None:
                return
            node_config = node.get_config()
            tracer = _otel_trace.get_tracer("launchdarkly.ai")
            parent_ctx = _otel_context.get_current()
            # Use start_as_current_span(end_on_exit=False) so the span is the active
            # current span while attributes are annotated, then stays open (not ended)
            # after the with-block exits. Both enter and exit happen within this same
            # on_agent_start call, so no cross-async-context ContextVar issues.
            with tracer.start_as_current_span("ld.ai.agent", context=parent_ctx, end_on_exit=False) as span:
                from ldai.observe import annotate_span_with_ai_config_metadata
                annotate_span_with_ai_config_metadata(node_config)
            open_spans[agent.name] = span

        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            node_key = name_map.get(agent.name, agent.name)
            _log.info("[ldai:openai] agent_end node_key=%r", node_key)
            span = open_spans.pop(agent.name, None)
            if span is not None:
                span.end()

    return _LDAIAgentSpanHooks()


class OpenAIAgentGraphRunner(AgentGraphRunner):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    AgentGraphRunner implementation for the OpenAI Agents SDK.

    Runs the agent graph with the OpenAI Agents SDK and automatically records
    graph- and node-level AI metric data to the LaunchDarkly trackers on the
    graph definition and each node.

    Requires ``openai-agents`` to be installed.
    """

    def __init__(self, graph: AgentGraphDefinition, tools: ToolRegistry):
        """
        Initialize the runner.

        :param graph: The AgentGraphDefinition to execute
        :param tools: Registry mapping tool names to callables or native ``Tool`` instances
        """
        self._graph = graph
        self._tools = tools
        self._agent_name_map: Dict[str, str] = {}
        self._tool_name_map: Dict[str, str] = {}

    async def run(self, input: Any) -> AgentGraphResult:
        """
        Run the agent graph with the given input.

        Builds the agent tree via reverse_traverse, then invokes the root
        agent with Runner.run(). Tracks path, latency, and invocation
        success/failure.

        :param input: The string prompt to send to the agent graph
        :return: AgentGraphResult with the final output and metrics
        """
        tracker = self._graph.get_tracker()
        path: List[str] = []
        root_node = self._graph.root()
        root_key = root_node.get_key() if root_node else ''
        if root_key:
            path.append(root_key)

        start_ns = time.perf_counter_ns()
        state = _RunState(last_handoff_ns=start_ns, last_node_key=root_key)
        try:
            from agents import Runner, RunHooks
            _log.info("[ldai:openai] invoke called for root_key=%r tracker=%r", root_key, tracker)
            root_agent = self._build_agents(path, state)
            hooks = _make_span_hooks(RunHooks, self._graph, self._agent_name_map)
            result = await Runner.run(root_agent, str(input), hooks=hooks)
            self._flush_final_segment(state, tracker, result)
            self._track_tool_calls(result, tracker)

            duration = (time.perf_counter_ns() - start_ns) // 1_000_000

            if tracker:
                tracker.track_path(path)
                tracker.track_latency(duration)
                tracker.track_invocation_success()
                token_usage = get_ai_usage_from_response(result)
                if token_usage is not None:
                    tracker.track_total_tokens(token_usage)

            return AgentGraphResult(
                output=str(result.final_output),
                raw=result,
                metrics=LDAIMetrics(success=True),
            )
        except Exception as exc:
            if isinstance(exc, ImportError):
                log.warning(
                    "openai-agents is required for OpenAIAgentGraphRunner. "
                    "Install it with: pip install openai-agents"
                )
            else:
                log.warning(f'OpenAIAgentGraphRunner run failed: {exc}')
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000
            if tracker:
                tracker.track_latency(duration)
                tracker.track_invocation_failure()
            return AgentGraphResult(
                output='',
                raw=None,
                metrics=LDAIMetrics(success=False),
            )

    def _build_agents(self, path: List[str], state: _RunState) -> Any:
        """
        Build the agent tree from the graph definition via reverse_traverse.

        Agents are constructed from terminal nodes upward so that handoff
        targets exist before the agents that hand off to them.

        :param path: Mutable list to accumulate the execution path
        :param state: Shared run state for tracking handoff timing and last node
        :return: The root Agent instance
        """
        try:
            from agents import (
                Agent,
                Handoff,
                handoff,
            )
            from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
        except ImportError as exc:
            raise ImportError(
                "openai-agents is required for OpenAIAgentGraphRunner. "
                "Install it with: pip install openai-agents"
            ) from exc

        tracker = self._graph.get_tracker()
        name_map: Dict[str, str] = {}
        tool_name_map: Dict[str, str] = {}

        def build_node(node: AgentGraphNode, ctx: dict) -> Any:
            node_config = node.get_config()
            config_tracker = node_config.tracker
            model = node_config.model

            if not model:
                raise ValueError(f"Model not set for node '{node_config.key}'")

            tool_defs = model.get_parameter('tools') or []
            sanitized_name = _sanitize_agent_name(node_config.key)
            name_map[sanitized_name] = node_config.key

            # --- handoffs ---
            agent_handoffs: List[Handoff] = []
            for edge in node.get_edges():
                target_key = edge.target_config
                agent_handoffs.append(
                    handoff(
                        agent=ctx[target_key],
                        on_handoff=self._make_on_handoff(
                            node_config.key,
                            target_key,
                            path,
                            tracker,
                            config_tracker,
                            state,
                        ),
                    )
                )

            # --- tools ---
            agent_tools: List[Any] = []
            for tool_def in tool_defs:
                tool_name = tool_def.get('name', '')

                tool_fn = self._tools.get(tool_name)
                if not tool_fn:
                    continue

                # Map runtime tool name → LD config key for metrics (function __name__
                # for callables; identity for native tool instances — see get_tool_calls_from_run_items).
                if is_agent_tool_instance(tool_fn):
                    tool_name_map[tool_fn.name] = tool_name
                else:
                    tool_name_map[tool_fn.__name__] = tool_name
                agent_tools.append(registry_value_to_agent_tool(tool_fn))

            return Agent(
                name=sanitized_name,
                model=model.name,
                instructions=f'{RECOMMENDED_PROMPT_PREFIX} {node_config.instructions or ""}',
                handoffs=list(agent_handoffs),
                tools=list(agent_tools),
            )

        root = self._graph.reverse_traverse(fn=build_node)
        self._agent_name_map = name_map
        self._tool_name_map = tool_name_map
        return root

    def _make_on_handoff(
        self,
        src: str,
        tgt: str,
        path: List[str],
        tracker: Any,
        config_tracker: Any,
        state: _RunState,
    ):
        def on_handoff(run_ctx: Any) -> None:
            self._handle_handoff(
                run_ctx, src, tgt, path, tracker, config_tracker, state
            )
        return on_handoff

    def _handle_handoff(
        self,
        run_ctx: Any,
        src: str,
        tgt: str,
        path: List[str],
        tracker: Any,
        config_tracker: Any,
        state: _RunState,
    ) -> None:
        _log.info("[ldai:openai] handoff from src=%r to tgt=%r", src, tgt)
        path.append(tgt)
        state.last_node_key = tgt
        if tracker:
            tracker.track_handoff_success(src, tgt)

        now_ns = time.perf_counter_ns()
        duration_ms = (now_ns - state.last_handoff_ns) // 1_000_000
        state.last_handoff_ns = now_ns

        usage: Optional[TokenUsage] = None
        try:
            usage = extract_usage_from_request_entry(
                run_ctx.usage.request_usage_entries[-1]
            )
        except Exception:
            pass

        gk = tracker.graph_key if tracker is not None else None
        if config_tracker is not None:
            if usage is not None:
                config_tracker.track_tokens(usage, graph_key=gk)
            if duration_ms is not None:
                config_tracker.track_duration(int(duration_ms), graph_key=gk)
            config_tracker.track_success(graph_key=gk)

    def _flush_final_segment(
        self,
        state: _RunState,
        tracker: Any,
        result: Any,
    ) -> None:
        """Record duration/tokens for the last active agent (no handoff after it)."""
        if not state.last_node_key:
            return
        node = self._graph.get_node(state.last_node_key)
        if node is None:
            return
        config_tracker = node.get_config().tracker
        if config_tracker is None:
            return

        now_ns = time.perf_counter_ns()
        duration_ms = (now_ns - state.last_handoff_ns) // 1_000_000

        usage: Optional[TokenUsage] = None
        try:
            usage = extract_usage_from_request_entry(
                result.context_wrapper.usage.request_usage_entries[-1]
            )
        except Exception:
            pass

        gk = tracker.graph_key if tracker is not None else None
        if usage is not None:
            config_tracker.track_tokens(usage, graph_key=gk)
        config_tracker.track_duration(int(duration_ms), graph_key=gk)
        config_tracker.track_success(graph_key=gk)

    def _track_tool_calls(self, result: Any, tracker: Any) -> None:
        """Track all tool calls from the run result, attributed to the node that called them."""
        gk = tracker.graph_key if tracker is not None else None
        for agent_name, tool_fn_name in get_tool_calls_from_run_items(result.new_items):
            agent_key = self._agent_name_map.get(agent_name, agent_name)
            tool_name = self._tool_name_map.get(tool_fn_name)
            if tool_name is None:
                continue
            node = self._graph.get_node(agent_key)
            if node is None:
                continue
            config_tracker = node.get_config().tracker
            if config_tracker is not None:
                config_tracker.track_tool_call(tool_name, graph_key=gk)
