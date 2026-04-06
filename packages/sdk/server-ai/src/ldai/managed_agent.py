"""ManagedAgent — LaunchDarkly managed wrapper for agent invocations."""

from ldai.models import AIAgentConfig
from ldai.observe import SPAN_NAME_AGENT, annotate_span_with_ai_config_metadata, _span_scope
from ldai.providers import AgentResult, AgentRunner
from ldai.tracker import LDAIConfigTracker


class ManagedAgent:
    """
    LaunchDarkly managed wrapper for AI agent invocations.

    Holds an AgentRunner and an LDAIConfigTracker. Handles tracking automatically.
    Obtain an instance via ``LDAIClient.create_agent()``.
    """

    def __init__(
        self,
        ai_config: AIAgentConfig,
        tracker: LDAIConfigTracker,
        agent_runner: AgentRunner,
    ):
        self._ai_config = ai_config
        self._tracker = tracker
        self._agent_runner = agent_runner

    async def run(self, input: str) -> AgentResult:
        """
        Run the agent with the given input string.

        :param input: The user prompt or input to the agent
        :return: AgentResult containing the agent's output and metrics
        """
        observe = self._tracker._observe_config
        with _span_scope(SPAN_NAME_AGENT, create_if_none=observe.create_span_if_none):
            if observe.annotate_spans:
                annotate_span_with_ai_config_metadata(
                    self._tracker._config_key,
                    self._tracker._variation_key,
                    self._tracker._model_name,
                    self._tracker._provider_name,
                    version=self._tracker._version,
                    context_key=self._tracker._context.key,
                )
            return await self._tracker.track_metrics_of_async(
                lambda: self._agent_runner.run(input),
                lambda result: result.metrics,
            )

    def get_agent_runner(self) -> AgentRunner:
        """
        Return the underlying AgentRunner for advanced use.

        :return: The AgentRunner instance.
        """
        return self._agent_runner

    def get_config(self) -> AIAgentConfig:
        """Return the AI agent config."""
        return self._ai_config

    def get_tracker(self) -> LDAIConfigTracker:
        """Return the config tracker."""
        return self._tracker
