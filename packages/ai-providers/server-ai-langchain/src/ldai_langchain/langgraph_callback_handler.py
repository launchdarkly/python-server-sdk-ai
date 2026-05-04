import time
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage

from ldai_langchain.langchain_helper import get_ai_usage_from_response


class LDMetricsCallbackHandler(BaseCallbackHandler):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    LangChain callback handler that collects per-node metrics during a LangGraph run.

    Records token usage, tool calls, and duration for each agent node in the graph.
    Each node's :class:`~ldai.providers.types.LDAIMetrics` is built incrementally
    as callbacks fire.  Access the ``node_metrics`` property after the run completes
    to retrieve the accumulated per-node metrics.
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
        # start time (ns) per active run_id — keyed by run_id to handle re-entrant nodes
        self._node_start_ns: Dict[UUID, int] = {}
        # per-node metrics, built incrementally as callbacks fire
        self._node_metrics: Dict[str, LDAIMetrics] = {}
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
    def node_metrics(self) -> Dict[str, LDAIMetrics]:
        """Per-node metrics keyed by node key."""
        return dict(self._node_metrics)

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
            self._node_start_ns[run_id] = time.perf_counter_ns()
            if name not in self._path_set:
                self._path.append(name)
                self._path_set.add(name)
                self._node_metrics[name] = LDAIMetrics(success=False)
        elif name.endswith('__tools'):
            stripped = name[: -len('__tools')]
            if stripped in self._node_keys:
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
        start_ns = self._node_start_ns.pop(run_id, None)
        if start_ns is not None:
            elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
            metrics = self._node_metrics.get(node_key)
            if metrics is not None:
                metrics.success = True
                metrics.duration_ms = (metrics.duration_ms or 0) + elapsed_ms

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

        metrics = self._node_metrics.get(node_key)
        if metrics is None:
            return
        existing = metrics.usage
        if existing is None:
            metrics.usage = usage
        else:
            metrics.usage = TokenUsage(
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
            return
        metrics = self._node_metrics.get(node_key)
        if metrics is None:
            return
        if metrics.tool_calls is None:
            metrics.tool_calls = [config_key]
        else:
            metrics.tool_calls.append(config_key)
