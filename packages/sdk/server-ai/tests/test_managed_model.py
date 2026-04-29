"""Tests for ManagedModel — specifically the evaluations tracking chain."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ldai.evaluator import Evaluator
from ldai.managed_model import ManagedModel
from ldai.models import AICompletionConfig, LDMessage, ModelConfig, ProviderConfig
from ldai.providers.types import JudgeResult, LDAIMetrics, ModelResponse
from ldai.tracker import LDAIConfigTracker



def _make_model_response(content: str = 'response text') -> ModelResponse:
    return ModelResponse(
        message=LDMessage(role='assistant', content=content),
        metrics=LDAIMetrics(success=True, usage=None),
    )


class TestManagedModelInvokeReturnsImmediately:
    """invoke() must return before the evaluations task resolves."""

    @pytest.mark.asyncio
    async def test_invoke_returns_before_evaluations_resolve(self):
        """invoke() should return a ModelResponse before evaluations complete."""
        # Set up a barrier so the evaluation coroutine doesn't complete until we release it
        barrier = asyncio.Event()

        async def _slow_evaluate(input_text: str, output_text: str) -> List[JudgeResult]:
            await barrier.wait()
            return []

        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_slow_evaluate(i, o))
        )

        mock_runner = MagicMock()
        mock_runner.invoke_model = AsyncMock(return_value=_make_model_response())

        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(return_value=_make_model_response())
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
        response = await model.invoke('Hello')

        # invoke() returned — evaluations task should still be pending
        assert response is not None
        assert response.evaluations is not None
        assert not response.evaluations.done(), "evaluations task should still be pending"

        # Release the barrier and let it finish cleanly
        barrier.set()
        await response.evaluations

    @pytest.mark.asyncio
    async def test_await_evaluations_collects_results(self):
        """await response.evaluations should return the list of JudgeResult instances."""
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
        mock_runner.invoke_model = AsyncMock(return_value=_make_model_response())

        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(return_value=_make_model_response())
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
        response = await model.invoke('Hello')

        results = await response.evaluations  # type: ignore[misc]
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
        mock_runner.invoke_model = AsyncMock(return_value=_make_model_response())

        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(return_value=_make_model_response())
        mock_tracker.track_judge_result = MagicMock()

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
        response = await model.invoke('Hello')

        # Tracking should NOT have fired yet (before we await evaluations)
        mock_tracker.track_judge_result.assert_not_called()

        # Now await the evaluations task — tracking fires inside the chain
        await response.evaluations  # type: ignore[misc]

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
        mock_runner.invoke_model = AsyncMock(return_value=_make_model_response())

        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(return_value=_make_model_response())
        mock_tracker.track_judge_result = MagicMock()

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
        response = await model.invoke('Hello')
        await response.evaluations  # type: ignore[misc]

        mock_tracker.track_judge_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_evaluator_returns_empty_list(self):
        """With a noop evaluator, awaiting evaluations should return an empty list."""
        evaluator = Evaluator.noop()

        mock_runner = MagicMock()
        mock_runner.invoke_model = AsyncMock(return_value=_make_model_response())

        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(return_value=_make_model_response())

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
        response = await model.invoke('Hello')
        results = await response.evaluations  # type: ignore[misc]

        assert results == []
