"""Evaluator implementation for coordinating multiple judges."""

import asyncio
from typing import Dict, List

from ldai import log
from ldai.judge import Judge
from ldai.models import JudgeConfiguration
from ldai.providers.types import JudgeResult


class Evaluator:
    """
    Coordinates multiple judge evaluations for a single AI config invocation.

    Instances are created by the SDK client via ``_build_evaluator()`` and injected
    into ``AIConfig`` objects (and runners) at construction time. User code should
    not need to construct this directly.
    """

    def __init__(self, judges: Dict[str, Judge], judge_configuration: JudgeConfiguration):
        """
        Initialize the Evaluator.

        :param judges: Mapping of judge config key to initialized Judge instances
        :param judge_configuration: The judge configuration specifying which judges to run
        """
        self._judges = judges
        self._judge_configuration = judge_configuration

    @classmethod
    def noop(cls) -> 'Evaluator':
        return cls({}, JudgeConfiguration(judges=[]))

    def evaluate(
        self,
        input_text: str,
        output_text: str,
    ) -> 'asyncio.Task[List[JudgeResult]]':
        """
        Run all configured judges against the given input/output pair.

        Schedules the judge evaluations as an asyncio Task and returns it
        immediately. The caller can await the task to get results or pass it
        to tracking helpers.

        :param input_text: The input that was provided to the AI model
        :param output_text: The AI-generated output to evaluate
        :return: An asyncio Task that resolves to a list of JudgeResult instances
        """
        return asyncio.create_task(self._run_judges(input_text, output_text))

    async def _run_judges(
        self,
        input_text: str,
        output_text: str,
    ) -> List[JudgeResult]:
        """
        Execute all configured judges and collect results.

        :param input_text: The input that was provided to the AI model
        :param output_text: The AI-generated output to evaluate
        :return: List of JudgeResult instances (one per configured judge that was found)
        """
        if not self._judge_configuration.judges:
            log.debug('No judges configured, no evaluations to run')
            return []
        results: List[JudgeResult] = []
        for jc in self._judge_configuration.judges:
            judge = self._judges.get(jc.key)
            if not judge:
                log.warning(f'Judge not enabled: {jc.key}')
                continue
            results.append(await judge.evaluate(input_text, output_text, jc.sampling_rate))
        return results
