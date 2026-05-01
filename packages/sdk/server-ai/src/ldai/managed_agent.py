"""ManagedAgent — LaunchDarkly managed wrapper for agent invocations."""

from ldai.models import AIAgentConfig
from ldai.providers.runner import Runner
from ldai.providers.types import ManagedResult


class ManagedAgent:
    """
    LaunchDarkly managed wrapper for AI agent invocations.

    Holds a Runner. Handles tracking automatically via ``create_tracker()``.
    Obtain an instance via ``LDAIClient.create_agent()``.
    """

    def __init__(
        self,
        ai_config: AIAgentConfig,
        agent_runner: Runner,
    ):
        self._ai_config = ai_config
        self._agent_runner = agent_runner

    async def run(self, input: str) -> ManagedResult:
        """
        Run the agent with the given input string.

        :param input: The user prompt or input to the agent
        :return: ManagedResult containing the agent's output and metric summary
        """
        tracker = self._ai_config.create_tracker()
        result = await tracker.track_metrics_of_async(
            lambda r: r.metrics,
            lambda: self._agent_runner.run(input),
        )
        return ManagedResult(
            content=result.content,
            metrics=tracker.get_summary(),
            raw=result.raw,
        )

    def get_agent_runner(self) -> Runner:
        """
        Return the underlying runner for advanced use.

        :return: The Runner instance.
        """
        return self._agent_runner

    def get_config(self) -> AIAgentConfig:
        """Return the AI agent config."""
        return self._ai_config
