"""ManagedAgent — LaunchDarkly managed wrapper for agent invocations."""

import asyncio
from typing import List

from ldai import log
from ldai.models import AIAgentConfig
from ldai.providers.runner import Runner
from ldai.providers.types import JudgeResult, ManagedResult
from ldai.tracker import LDAIConfigTracker


class ManagedAgent:
    """
    LaunchDarkly managed wrapper for AI agent invocations.

    Holds a Runner. Handles tracking and judge evaluation
    dispatch automatically via ``create_tracker()``.
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

        Invokes the runner, tracks metrics, and dispatches judge evaluations
        asynchronously.  Returns immediately; awaiting ``result.evaluations``
        guarantees both evaluation and tracking complete.

        :param input: The user prompt or input to the agent
        :return: ManagedResult containing the agent's output, metric summary,
            and an optional evaluations task
        """
        tracker = self._ai_config.create_tracker()
        result = await tracker.track_metrics_of_async(
            lambda r: r.metrics,
            lambda: self._agent_runner.run(input),
        )

        evaluations_task = self._track_judge_results(tracker, input, result.content)

        return ManagedResult(
            content=result.content,
            metrics=tracker.get_summary(),
            raw=result.raw,
            evaluations=evaluations_task,
        )

    def _track_judge_results(
        self,
        tracker: LDAIConfigTracker,
        input_text: str,
        output_text: str,
    ) -> asyncio.Task[List[JudgeResult]]:
        evaluator_task = self._ai_config.evaluator.evaluate(input_text, output_text)

        async def _run_and_track(eval_task: asyncio.Task) -> List[JudgeResult]:
            results = await eval_task
            for r in results:
                if r.success:
                    try:
                        tracker.track_judge_result(r)
                    except Exception as exc:
                        log.warning("Judge evaluation failed: %s", exc)
                else:
                    log.warning("Judge evaluation failed: %s", r.error_message)
            return results

        return asyncio.create_task(_run_and_track(evaluator_task))

    def get_agent_runner(self) -> Runner:
        """
        Return the underlying runner for advanced use.

        :return: The Runner instance.
        """
        return self._agent_runner

    def get_config(self) -> AIAgentConfig:
        """Return the AI agent config."""
        return self._ai_config
