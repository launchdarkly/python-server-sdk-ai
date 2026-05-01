"""Evaluator implementation for coordinating multiple judges."""

from __future__ import annotations

import asyncio
from typing import List

from ldai import log
from ldai.judge import Judge
from ldai.providers.types import JudgeResult


class Evaluator:
    """
    Coordinates multiple judge evaluations for a single AI config invocation.

    Instances are created by the SDK client via ``_build_evaluator()`` and injected
    into ``AIConfig`` objects (and runners) at construction time. User code should
    not need to construct this directly.
    """

    def __init__(self, judges: List[Judge]):
        """
        Initialize the Evaluator.

        :param judges: List of initialized Judge instances. Each Judge already
            carries its own ``sample_rate`` set at construction time.
        """
        self._judges = judges

    @classmethod
    def noop(cls) -> Evaluator:
        return cls([])

    def evaluate(
        self,
        input_text: str,
        output_text: str,
    ) -> asyncio.Task[List[JudgeResult]]:
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
        :return: List of JudgeResult instances (one per configured judge)
        """
        if not self._judges:
            log.debug('No judges configured, no evaluations to run')
            return []
        results: List[JudgeResult] = []
        for judge in self._judges:
            results.append(await judge.evaluate(input_text, output_text))
        return results
