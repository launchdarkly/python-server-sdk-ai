"""ManagedAgent — LaunchDarkly managed wrapper for agent invocations."""

from ldai.models import AIAgentConfig
from ldai.providers import AgentResult, AgentRunner


class ManagedAgent:
    """
    LaunchDarkly managed wrapper for AI agent invocations.

    Holds an AgentRunner. Handles tracking automatically via ``create_tracker()``.
    Obtain an instance via ``LDAIClient.create_agent()``.
    """

    def __init__(
        self,
        ai_config: AIAgentConfig,
        agent_runner: AgentRunner,
    ):
        self._ai_config = ai_config
        self._agent_runner = agent_runner

    async def run(self, input: str) -> AgentResult:
        """
        Run the agent with the given input string.

        :param input: The user prompt or input to the agent
        :return: AgentResult containing the agent's output and metrics
        """
        tracker = self._ai_config.create_tracker()
        return await tracker.track_metrics_of_async(
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
