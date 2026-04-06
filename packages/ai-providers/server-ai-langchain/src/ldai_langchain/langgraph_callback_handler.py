import time
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult
from ldai.agent_graph import AgentGraphDefinition
from ldai.tracker import TokenUsage

from ldai_langchain.langchain_helper import get_ai_usage_from_response


class LDMetricsCallbackHandler(BaseCallbackHandler):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    LangChain callback handler that collects per-node metrics during a LangGraph run.

    Records token usage, tool calls, and duration for each agent node in the graph,
    then flushes them to LaunchDarkly trackers after the run completes via ``flush()``.
    """

    def __init__(self, node_keys: Set[str], fn_name_to_config_key: Dict[str, str]):
        """
        Initialize the handler.

        :param node_keys: Set of LangGraph node keys that represent agent nodes
            (excludes ``__tools`` suffix nodes).
        :param fn_name_to_config_key: Mapping from tool function ``__name__`` to
            the LD config key for that tool (e.g. ``'fetch_weather'`` -> ``'get_weather_open_meteo'``).
        """
        super().__init__()
        self._node_keys = node_keys
        self._fn_name_to_config_key = fn_name_to_config_key

        # run_id -> node_key for active chain runs
        self._run_to_node: Dict[UUID, str] = {}
        # accumulated token usage per node
        self._node_tokens: Dict[str, TokenUsage] = {}
        # tool config keys called per node
        self._node_tool_calls: Dict[str, List[str]] = {}
        # start time (ns) per node — only set while running
        self._node_start_ns: Dict[str, int] = {}
        # accumulated duration (ms) per node
        self._node_duration_ms: Dict[str, int] = {}
        # execution path in order (deduplicated)
        self._path: List[str] = []
        self._path_set: Set[str] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> List[str]:
        """Execution path through the graph in order."""
        return list(self._path)

    @property
    def node_tokens(self) -> Dict[str, TokenUsage]:
        """Accumulated token usage per node key."""
        return dict(self._node_tokens)

    @property
    def node_tool_calls(self) -> Dict[str, List[str]]:
        """Tool config keys called per node key."""
        return {k: list(v) for k, v in self._node_tool_calls.items()}

    @property
    def node_durations_ms(self) -> Dict[str, int]:
        """Accumulated duration in milliseconds per node key."""
        return dict(self._node_duration_ms)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record start of a chain run; attribute to the matching agent node."""
        if name is None:
            return

        if name in self._node_keys:
            self._run_to_node[run_id] = name
            self._node_start_ns[name] = time.perf_counter_ns()
            if name not in self._path_set:
                self._path.append(name)
                self._path_set.add(name)
        elif name.endswith('__tools'):
            stripped = name[: -len('__tools')]
            if stripped in self._node_keys:
                # Attribute tool events to the owning agent node
                self._run_to_node[run_id] = stripped

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record end of a chain run and accumulate elapsed duration."""
        node_key = self._run_to_node.get(run_id)
        if node_key is None:
            return
        start_ns = self._node_start_ns.pop(node_key, None)
        if start_ns is not None:
            elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
            self._node_duration_ms[node_key] = (
                self._node_duration_ms.get(node_key, 0) + elapsed_ms
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Accumulate token usage for the node that owns this LLM call."""
        if parent_run_id is None:
            return
        node_key = self._run_to_node.get(parent_run_id)
        if node_key is None:
            return

        try:
            gen = response.generations[0][0]
        except (IndexError, TypeError):
            return
        if not isinstance(gen, ChatGeneration):
            return
        message = gen.message
        usage = get_ai_usage_from_response(message)
        if usage is None:
            return

        existing = self._node_tokens.get(node_key)
        if existing is None:
            self._node_tokens[node_key] = usage
        else:
            self._node_tokens[node_key] = TokenUsage(
                total=existing.total + usage.total,
                input=existing.input + usage.input,
                output=existing.output + usage.output,
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool invocation for the owning agent node."""
        if parent_run_id is None or name is None:
            return
        node_key = self._run_to_node.get(parent_run_id)
        if node_key is None:
            return

        config_key = self._fn_name_to_config_key.get(name)
        if config_key is None:
            # Tool is not a registered functional tool (e.g. a handoff tool) — skip tracking.
            return
        if node_key not in self._node_tool_calls:
            self._node_tool_calls[node_key] = []
        self._node_tool_calls[node_key].append(config_key)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self, graph: AgentGraphDefinition, graph_tracker: Any) -> None:
        """
        Emit all collected per-node metrics to the LaunchDarkly trackers.

        Call this once after the graph run completes.

        :param graph: The AgentGraphDefinition whose nodes hold the LD config trackers.
        :param graph_tracker: The AIGraphTracker for the overall graph (may be None).
        """
        gk = graph_tracker.graph_key if graph_tracker is not None else None
        for node_key in self._path:
            node = graph.get_node(node_key)
            if not node:
                continue
            config_tracker = node.get_config().tracker
            if not config_tracker:
                continue

            usage = self._node_tokens.get(node_key)
            if usage:
                config_tracker.track_tokens(usage, graph_key=gk)

            duration = self._node_duration_ms.get(node_key)
            if duration is not None:
                config_tracker.track_duration(duration, graph_key=gk)

            config_tracker.track_success(graph_key=gk)

            for tool_key in self._node_tool_calls.get(node_key, []):
                config_tracker.track_tool_call(tool_key, graph_key=gk)
