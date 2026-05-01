"""Tests for ManagedModel — specifically the evaluations tracking chain."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from ldai.evaluator import Evaluator
from ldai.managed_model import ManagedModel
from ldai.models import AICompletionConfig, LDMessage, ModelConfig, ProviderConfig
from ldai.providers.types import JudgeResult, LDAIMetrics, ManagedResult, RunnerResult
from ldai.tracker import LDAIConfigTracker, LDAIMetricSummary



def _make_runner_result(content: str = 'response text') -> RunnerResult:
    return RunnerResult(
        content=content,
        metrics=LDAIMetrics(success=True, usage=None),
    )


def _make_summary() -> LDAIMetricSummary:
    summary = LDAIMetricSummary()
    summary._success = True
    return summary


def _make_config_with_tracker(evaluator: Evaluator) -> tuple[AICompletionConfig, MagicMock]:
    """Build an AICompletionConfig with a fully-mocked tracker."""
    mock_tracker = MagicMock(spec=LDAIConfigTracker)
    mock_tracker.track_metrics_of_async = AsyncMock(return_value=_make_runner_result())
    mock_tracker.get_summary = MagicMock(return_value=_make_summary())
    config = AICompletionConfig(
        key='test-config',
        enabled=True,
        create_tracker=MagicMock(return_value=mock_tracker),
        model=ModelConfig('gpt-4'),
        provider=ProviderConfig('openai'),
        messages=[],
        evaluator=evaluator,
    )
    return config, mock_tracker


class TestManagedModelRunReturnsImmediately:
    """run() must return before the evaluations task resolves."""

    @pytest.mark.asyncio
    async def test_run_returns_managed_result(self):
        """run() should return a ManagedResult with content from the runner."""
        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_empty_eval())
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_runner_result('hi'))

        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(return_value=_make_runner_result('hi'))
        mock_tracker.get_summary = MagicMock(return_value=_make_summary())
        config = AICompletionConfig(
            key='test-config',
            enabled=True,
            create_tracker=MagicMock(return_value=mock_tracker),
            model=ModelConfig('gpt-4'),
            provider=ProviderConfig('openai'),
            messages=[],
            evaluator=evaluator,
        )

        model = ManagedModel(config, mock_runner)
        result = await model.run('Hello')

        assert isinstance(result, ManagedResult)
        assert result.content == 'hi'
        assert isinstance(result.metrics, LDAIMetricSummary)
        # Cleanup the still-pending evaluations task.
        if result.evaluations is not None:
            await result.evaluations

    @pytest.mark.asyncio
    async def test_run_returns_before_evaluations_resolve(self):
        """run() should return a ManagedResult before evaluations complete."""
        barrier = asyncio.Event()

        async def _slow_evaluate(input_text: str, output_text: str) -> List[JudgeResult]:
            await barrier.wait()
            return []

        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_slow_evaluate(i, o))
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_runner_result())

        config, _tracker = _make_config_with_tracker(evaluator)
        model = ManagedModel(config, mock_runner)
        result = await model.run('Hello')

        assert result is not None
        assert result.evaluations is not None
        assert not result.evaluations.done(), "evaluations task should still be pending"

        barrier.set()
        await result.evaluations

    @pytest.mark.asyncio
    async def test_await_evaluations_collects_results(self):
        """await result.evaluations should return the list of JudgeResult instances."""
        judge_result = JudgeResult(
            judge_config_key='judge-key',
            success=True,
            sampled=True,
            metric_key='$ld:ai:judge:relevance',
            score=0.9,
            reasoning='Good response',
        )

        async def _evaluate_coro(input_text: str, output_text: str) -> List[JudgeResult]:
            return [judge_result]

        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_evaluate_coro(i, o))
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_runner_result())

        config, _tracker = _make_config_with_tracker(evaluator)
        model = ManagedModel(config, mock_runner)
        result = await model.run('Hello')

        results = await result.evaluations  # type: ignore[misc]
        assert results == [judge_result]

    @pytest.mark.asyncio
    async def test_tracking_fires_inside_awaited_chain(self):
        """tracker.track_judge_result() must be called when evaluations are awaited."""
        judge_result = JudgeResult(
            judge_config_key='judge-key',
            success=True,
            sampled=True,
            metric_key='$ld:ai:judge:relevance',
            score=0.85,
            reasoning='Relevant answer',
        )

        async def _evaluate_coro(input_text: str, output_text: str) -> List[JudgeResult]:
            return [judge_result]

        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_evaluate_coro(i, o))
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_runner_result())

        config, mock_tracker = _make_config_with_tracker(evaluator)
        mock_tracker.track_judge_result = MagicMock()

        model = ManagedModel(config, mock_runner)
        result = await model.run('Hello')

        # Tracking should NOT have fired yet (before we await evaluations)
        mock_tracker.track_judge_result.assert_not_called()

        # Now await the evaluations task — tracking fires inside the chain
        await result.evaluations  # type: ignore[misc]

        mock_tracker.track_judge_result.assert_called_once_with(judge_result)

    @pytest.mark.asyncio
    async def test_tracking_not_called_for_failed_judge_result(self):
        """tracker.track_judge_result() should NOT be called for unsuccessful judge results."""
        failed_result = JudgeResult(
            success=False,
            sampled=True,
            metric_key='$ld:ai:judge:relevance',
            error_message='Judge evaluation failed',
        )

        async def _evaluate_coro(input_text: str, output_text: str) -> List[JudgeResult]:
            return [failed_result]

        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_evaluate_coro(i, o))
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_runner_result())

        config, mock_tracker = _make_config_with_tracker(evaluator)
        mock_tracker.track_judge_result = MagicMock()

        model = ManagedModel(config, mock_runner)
        result = await model.run('Hello')
        await result.evaluations  # type: ignore[misc]

        mock_tracker.track_judge_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_evaluator_returns_empty_list(self):
        """With a noop evaluator, awaiting evaluations should return an empty list."""
        evaluator = Evaluator.noop()

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_runner_result())

        config, _tracker = _make_config_with_tracker(evaluator)
        model = ManagedModel(config, mock_runner)
        result = await model.run('Hello')
        results = await result.evaluations  # type: ignore[misc]

        assert results == []


async def _empty_eval() -> List[JudgeResult]:
    return []
