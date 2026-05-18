import re
import time
from typing import Any, Dict, List

from ldai import log
from ldai.agent_graph import AgentGraphDefinition, AgentGraphNode
from ldai.providers import AgentGraphRunner, ToolRegistry
from ldai.providers.types import (
    AgentGraphRunnerResult,
    AIGraphMetrics,
    EvalRequest,
    LDAIMetrics,
)

from ldai_openai.openai_helper import (
    extract_usage_from_request_entry,
    get_ai_usage_from_response,
    get_tool_calls_from_run_items,
    is_agent_tool_instance,
    registry_value_to_agent_tool,
)


def _sanitize_agent_name(key: str) -> str:
    """Replace characters invalid for OpenAI function names with underscores."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', key)


class _RunState:
    """Mutable state shared across handoff and tool callbacks during a single run."""

    def __init__(self, last_handoff_ns: int, last_node_key: str) -> None:
        self.last_handoff_ns = last_handoff_ns
        self.last_node_key = last_node_key


class OpenAIAgentGraphRunner(AgentGraphRunner):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    AgentGraphRunner implementation for the OpenAI Agents SDK.

    Runs the agent graph with the OpenAI Agents SDK and collects graph- and
    node-level metrics.  Tracking events are emitted by the managed layer
    (:class:`~ldai.ManagedAgentGraph`) from the returned
    :class:`~ldai.providers.types.AgentGraphRunnerResult`.

    Requires ``openai-agents`` to be installed.
    """

    def __init__(
        self,
        graph: AgentGraphDefinition,
        tools: ToolRegistry,
    ):
        """
        Initialize the runner.

        :param graph: The AgentGraphDefinition to execute
        :param tools: Registry mapping tool names to callables or native ``Tool`` instances
        """
        self._graph = graph
        self._tools = tools
        self._agent_name_map: Dict[str, str] = {}
        self._tool_name_map: Dict[str, str] = {}
        self._node_metrics: Dict[str, LDAIMetrics] = {}

    async def run(self, input: str) -> AgentGraphRunnerResult:
        """
        Run the agent graph with the given input.

        Builds the agent tree via reverse_traverse, then invokes the root
        agent with Runner.run(). Collects path, latency, and per-node metrics.
        Graph-level tracking events are emitted by the managed layer.

        :param input: The string prompt to send to the agent graph
        :return: AgentGraphRunnerResult with the final content and AIGraphMetrics
        """
        self._node_metrics = {}
        path: List[str] = []
        root_node = self._graph.root()
        root_key = root_node.get_key() if root_node else ''
        if root_key:
            path.append(root_key)

        start_ns = time.perf_counter_ns()
        state = _RunState(last_handoff_ns=start_ns, last_node_key=root_key)
        try:
            from agents import Runner
            root_agent = self._build_agents(path, state)
            if root_key:
                self._node_metrics[root_key] = LDAIMetrics(success=False)
            result = await Runner.run(root_agent, input)
            self._flush_final_segment(state, result)
            self._collect_tool_calls(result)
            eval_requests = self._extract_eval_requests(result, input)

            duration_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
            token_usage = get_ai_usage_from_response(result)

            return AgentGraphRunnerResult(
                content=str(result.final_output),
                raw=result,
                metrics=AIGraphMetrics(
                    success=True,
                    path=path,
                    duration_ms=duration_ms,
                    tokens=token_usage,
                    node_metrics=self._node_metrics,
                ),
                eval_requests=eval_requests if eval_requests else None,
            )
        except Exception as exc:
            if isinstance(exc, ImportError):
                log.warning(
                    "openai-agents is required for OpenAIAgentGraphRunner. "
                    "Install it with: pip install openai-agents"
                )
            else:
                log.warning(f'OpenAIAgentGraphRunner run failed: {exc}')
            duration_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
            return AgentGraphRunnerResult(
                content='',
                raw=None,
                metrics=AIGraphMetrics(
                    success=False,
                    path=path,
                    duration_ms=duration_ms,
                    node_metrics=self._node_metrics,
                ),
            )

    def _build_agents(
        self, path: List[str], state: _RunState
    ) -> Any:
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

        name_map: Dict[str, str] = {}
        tool_name_map: Dict[str, str] = {}

        def build_node(node: AgentGraphNode, ctx: dict) -> Any:
            node_config = node.get_config()
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
        state: _RunState,
    ):
        def on_handoff(run_ctx: Any) -> None:
            self._handle_handoff(run_ctx, src, tgt, path, state)
        return on_handoff

    def _handle_handoff(
        self,
        run_ctx: Any,
        src: str,
        tgt: str,
        path: List[str],
        state: _RunState,
    ) -> None:
        path.append(tgt)

        now_ns = time.perf_counter_ns()
        duration_ms = (now_ns - state.last_handoff_ns) // 1_000_000
        state.last_handoff_ns = now_ns

        src_metrics = self._node_metrics.get(src)
        if src_metrics is not None:
            src_metrics.success = True
            src_metrics.duration_ms = int(duration_ms)
            try:
                src_metrics.tokens = extract_usage_from_request_entry(
                    run_ctx.usage.request_usage_entries[-1]
                )
            except Exception:
                pass

        self._node_metrics[tgt] = LDAIMetrics(success=False)
        state.last_node_key = tgt

    def _flush_final_segment(self, state: _RunState, result: Any) -> None:
        """Record duration/tokens for the last active agent (no handoff after it)."""
        if not state.last_node_key:
            return
        metrics = self._node_metrics.get(state.last_node_key)
        if metrics is None:
            return

        metrics.success = True
        now_ns = time.perf_counter_ns()
        metrics.duration_ms = int((now_ns - state.last_handoff_ns) // 1_000_000)

        try:
            metrics.tokens = extract_usage_from_request_entry(
                result.context_wrapper.usage.request_usage_entries[-1]
            )
        except Exception:
            pass

    def _extract_eval_requests(
        self, result: Any, user_input: str
    ) -> List[EvalRequest]:
        """
        Extract per-node input/output pairs from a finished run result.

        Walks ``result.new_items`` in order. Each ``MessageOutputItem`` is the
        text produced by an agent on a particular activation; when followed by
        a ``HandoffOutputItem`` (or end of run) the message is treated as that
        agent's final output. Nodes whose ``AIAgentConfig`` has no
        ``judge_configuration`` with at least one judge contribute no entries.

        Returns an empty list when no items match (import failure, no
        configured judges, empty outputs, etc.).
        """
        try:
            from agents.items import (
                HandoffOutputItem,
                ItemHelpers,
                MessageOutputItem,
            )
        except ImportError:
            return []

        new_items = getattr(result, 'new_items', None) or []
        requests: List[EvalRequest] = []

        # Last MessageOutputItem text seen per agent name (sanitized).
        last_output_by_agent: Dict[str, str] = {}
        # Pending node activations waiting to be flushed when the agent
        # transitions away (via handoff) or the run ends.
        pending: List[tuple] = []  # list of (node_key, input_text)
        current_agent_name: Any = None
        # Prompt the next activation will receive. Starts as the user input;
        # after a handoff becomes the source agent's last message — that is
        # what the target agent sees as its trigger.
        current_input: str = user_input

        for item in new_items:
            if isinstance(item, MessageOutputItem):
                agent_name = getattr(item.agent, 'name', None)
                if not agent_name:
                    continue
                if current_agent_name != agent_name:
                    # New activation — record input for this node.
                    node_key = self._agent_name_map.get(agent_name, agent_name)
                    pending.append((node_key, current_input))
                    current_agent_name = agent_name
                text = ItemHelpers.extract_text(item.raw_item) or ''
                last_output_by_agent[agent_name] = text
            elif isinstance(item, HandoffOutputItem):
                src_name = getattr(item.source_agent, 'name', None)
                if src_name and src_name in last_output_by_agent:
                    src_output = last_output_by_agent[src_name]
                    self._flush_eval_request(requests, pending, src_name, src_output)
                    # The target agent's input is the source agent's final
                    # output that triggered the handoff.
                    current_input = src_output or current_input
                current_agent_name = None

        # Flush any agent activation that did not end in a handoff (end of run).
        if current_agent_name is not None:
            final_output = last_output_by_agent.get(current_agent_name) or str(
                getattr(result, 'final_output', '') or ''
            )
            self._flush_eval_request(
                requests, pending, current_agent_name, final_output
            )

        return requests

    def _flush_eval_request(
        self,
        out: List[EvalRequest],
        pending: List[tuple],
        agent_name: str,
        output_text: str,
    ) -> None:
        """Append the pending activation for ``agent_name`` if the node has judges and output is non-empty."""
        if not pending:
            return
        node_key, input_text = pending.pop()
        if not output_text or not output_text.strip():
            return
        node = self._graph.get_node(node_key)
        if node is None:
            return
        cfg = node.get_config()
        jc = getattr(cfg, 'judge_configuration', None)
        if jc is None or not getattr(jc, 'judges', None):
            return
        out.append(
            EvalRequest(
                node_key=node_key,
                input=input_text,
                output=output_text,
            )
        )

    def _collect_tool_calls(self, result: Any) -> None:
        """Collect all tool calls from the run result, attributed to the node that called them."""
        for agent_name, tool_fn_name in get_tool_calls_from_run_items(result.new_items):
            agent_key = self._agent_name_map.get(agent_name, agent_name)
            tool_name = self._tool_name_map.get(tool_fn_name)
            if tool_name is None:
                continue
            metrics = self._node_metrics.get(agent_key)
            if metrics is not None:
                if metrics.tool_calls is None:
                    metrics.tool_calls = [tool_name]
                else:
                    metrics.tool_calls.append(tool_name)
