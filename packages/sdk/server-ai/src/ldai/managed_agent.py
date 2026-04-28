"""ManagedAgent — LaunchDarkly managed wrapper for agent invocations."""

from typing import Union

from ldai.models import AIAgentConfig
from ldai.providers import AgentResult, AgentRunner
from ldai.providers.runner import Runner
from ldai.providers.types import ManagedResult, RunnerResult


class ManagedAgent:
    """
    LaunchDarkly managed wrapper for AI agent invocations.

    Holds an AgentRunner or Runner. Handles tracking automatically via
    ``create_tracker()``.
    Obtain an instance via ``LDAIClient.create_agent()``.
    """

    def __init__(
        self,
        ai_config: AIAgentConfig,
        agent_runner: Union[Runner, AgentRunner],
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
        result: Union[RunnerResult, AgentResult] = await tracker.track_metrics_of_async(
            lambda r: r.metrics,
            lambda: self._agent_runner.run(input),
        )
        # Support both RunnerResult (content) and legacy AgentResult (output)
        content = result.content if isinstance(result, RunnerResult) else result.output  # type: ignore[union-attr]
        return ManagedResult(
            content=content,
            metrics=tracker.get_summary(),
            raw=result.raw,
        )

    def get_agent_runner(self) -> Union[Runner, AgentRunner]:
        """
        Return the underlying runner for advanced use.

        :return: The Runner or AgentRunner instance.
        """
        return self._agent_runner

    def get_config(self) -> AIAgentConfig:
        """Return the AI agent config."""
        return self._ai_config
