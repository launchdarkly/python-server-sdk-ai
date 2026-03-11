"""OpenAI agent graph runner for LaunchDarkly AI SDK."""

import time
from typing import Any, List, Optional

from ldai.agent_graph import AgentGraphDefinition, AgentGraphNode
from ldai.providers.types import LDAIMetrics
from ldai.runners.agent_graph_runner import AgentGraphRunner
from ldai.runners.types import AgentGraphResult, ToolRegistry
from ldai.tracker import TokenUsage


def _to_openai_name(name: str) -> str:
    """Convert a hyphenated tool/node name to an underscore-separated OpenAI function name."""
    return name.replace('-', '_')


class OpenAIAgentGraphRunner(AgentGraphRunner):
    """
    AgentGraphRunner implementation for the OpenAI Agents SDK.

    Builds agents from an AgentGraphDefinition and a ToolRegistry via
    reverse_traverse, executes them with Runner.run(), and auto-tracks
    path, tool calls, handoffs, latency, and invocation success/failure
    via the graph's AIGraphTracker.

    Requires ``openai-agents`` to be installed.
    """

    def __init__(self, graph: AgentGraphDefinition, tools: ToolRegistry):
        """
        Initialize the runner.

        :param graph: The AgentGraphDefinition to execute
        :param tools: Registry mapping OpenAI-formatted tool names to callables
        """
        self._graph = graph
        self._tools = tools

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
        if root_node:
            path.append(root_node.get_key())

        start_time = time.time()
        try:
            try:
                from agents import Runner
            except ImportError as exc:
                raise ImportError(
                    "openai-agents is required for OpenAIAgentGraphRunner. "
                    "Install it with: pip install openai-agents"
                ) from exc

            root_agent = self._build_agents(path)
            result = await Runner.run(root_agent, str(input))
            duration = int((time.time() - start_time) * 1000)

            if tracker:
                tracker.track_path(path)
                tracker.track_latency(duration)
                tracker.track_invocation_success()

            return AgentGraphResult(
                output=str(result.final_output),
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

    def _build_agents(self, path: List[str]) -> Any:
        """
        Build the agent tree from the graph definition via reverse_traverse.

        Agents are constructed from terminal nodes upward so that handoff
        targets exist before the agents that hand off to them.

        :param path: Mutable list to accumulate the execution path
        :return: The root Agent instance
        """
        try:
            from agents import Agent, FunctionTool, Handoff, RunContextWrapper, Tool, handoff
            from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
            from agents.tool_context import ToolContext
        except ImportError as exc:
            raise ImportError(
                "openai-agents is required for OpenAIAgentGraphRunner. "
                "Install it with: pip install openai-agents"
            ) from exc

        tracker = self._graph.get_tracker()

        def build_node(node: AgentGraphNode, ctx: dict) -> Any:
            node_config = node.get_config()
            config_tracker = node_config.tracker
            model = node_config.model

            if not model:
                raise ValueError(f"Model not set for node '{node_config.key}'")

            tool_defs = model.get_parameter('tools') or []

            # --- handoffs ---
            agent_handoffs: List[Handoff] = []
            for edge in node.get_edges():
                target_key = edge.target_config

                def _make_on_handoff(src: str, tgt: str):
                    def on_handoff(run_ctx: RunContextWrapper) -> None:
                        path.append(tgt)
                        if tracker:
                            tracker.track_handoff_success(src, tgt)
                            tracker.track_node_invocation(src)
                        if config_tracker:
                            try:
                                usage_entry = run_ctx.usage.request_usage_entries[-1]
                                config_tracker.track_tokens(
                                    TokenUsage(
                                        total=usage_entry.total_tokens,
                                        input=usage_entry.input_tokens,
                                        output=usage_entry.output_tokens,
                                    )
                                )
                            except Exception:
                                pass
                            config_tracker.track_success()
                    return on_handoff

                agent_handoffs.append(
                    handoff(
                        agent=ctx[target_key],
                        on_handoff=_make_on_handoff(node_config.key, target_key),
                    )
                )

            # --- tools ---
            agent_tools: List[Tool] = []
            for tool_def in tool_defs:
                tool_name_raw = tool_def.get('name', '')
                tool_name = _to_openai_name(tool_name_raw)
                tool_fn = self._tools.get(tool_name) or self._tools.get(tool_name_raw)
                if not tool_fn:
                    continue

                def _make_tool(
                    name: str,
                    raw_name: str,
                    fn: Any,
                    description: str,
                    params_schema: dict,
                    cfg_key: str,
                ) -> FunctionTool:
                    def wrapped(tool_ctx: ToolContext, tool_args: str) -> Any:
                        import json
                        try:
                            args = json.loads(tool_args)
                        except Exception:
                            args = {}
                        path.append(raw_name)
                        if tracker:
                            tracker.track_tool_call(config_key=cfg_key, tool_key=name)
                        return fn(**args)

                    return FunctionTool(
                        name=f'tool_{name}',
                        description=description,
                        params_json_schema=params_schema,
                        on_invoke_tool=wrapped,
                    )

                agent_tools.append(
                    _make_tool(
                        tool_name,
                        tool_name_raw,
                        tool_fn,
                        tool_def.get('description', ''),
                        tool_def.get('parameters', {}),
                        node_config.key,
                    )
                )

            return Agent(
                name=_to_openai_name(node_config.key),
                instructions=f'[RECOMMENDED_PROMPT_PREFIX] {node_config.instructions or ""}',
                handoffs=list(agent_handoffs),
                tools=list(agent_tools),
            )

        return self._graph.reverse_traverse(fn=build_node)
