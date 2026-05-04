"""Tests for ManagedAgent."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, ManagedAgent
from ldai.evaluator import Evaluator
from ldai.managed_agent import ManagedAgent
from ldai.models import AIAgentConfig, AIAgentConfigDefault, ModelConfig, ProviderConfig
from ldai.providers.types import JudgeResult, LDAIMetrics, ManagedResult, RunnerResult
from ldai.tracker import LDAIConfigTracker, LDAIMetricSummary


def _make_summary(success: bool = True) -> LDAIMetricSummary:
    summary = LDAIMetricSummary()
    summary._success = success
    return summary


def _make_noop_evaluator_config() -> MagicMock:
    """Build a minimal mock AIAgentConfig with a noop evaluator and a mock tracker."""
    mock_config = MagicMock(spec=AIAgentConfig)
    mock_tracker = MagicMock(spec=LDAIConfigTracker)
    mock_tracker.track_metrics_of_async = AsyncMock(
        return_value=RunnerResult(
            content="Test response",
            raw=None,
            metrics=LDAIMetrics(success=True, usage=None),
        )
    )
    mock_tracker.get_summary = MagicMock(return_value=_make_summary(True))
    mock_config.create_tracker = MagicMock(return_value=mock_tracker)
    mock_config.evaluator = Evaluator.noop()
    return mock_config


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('customer-support-agent')
        .variations({
            'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.3}},
            'provider': {'name': 'openai'},
            'instructions': 'You are a helpful customer support agent.',
            '_ldMeta': {'enabled': True, 'variationKey': 'agent-v1', 'version': 1},
        })
        .variation_for_all(0)
    )
    td.update(
        td.flag('disabled-agent')
        .variations({
            'model': {'name': 'gpt-4'},
            '_ldMeta': {'enabled': False, 'variationKey': 'disabled-v1', 'version': 1},
        })
        .variation_for_all(0)
    )
    return td


@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config('sdk-key', update_processor_class=td, send_events=False)
    return LDClient(config=config)


@pytest.fixture
def ldai_client(client: LDClient) -> LDAIClient:
    return LDAIClient(client)


class TestManagedAgentRun:
    """Tests for ManagedAgent.run."""

    @pytest.mark.asyncio
    async def test_run_delegates_to_agent_runner(self):
        """Should delegate run() to the underlying AgentRunner and return ManagedResult."""
        mock_config = _make_noop_evaluator_config()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=RunnerResult(
                content="Test response",
                metrics=LDAIMetrics(success=True, usage=None),
                raw=None,
            )
        )

        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

        assert isinstance(result, ManagedResult)
        assert result.content == "Test response"
        assert result.metrics.success is True
        mock_config.create_tracker.assert_called_once()
        mock_config.create_tracker.return_value.track_metrics_of_async.assert_called_once()
        # evaluations should be present (from noop evaluator)
        if result.evaluations is not None:
            await result.evaluations

    @pytest.mark.asyncio
    async def test_run_uses_create_tracker_for_fresh_tracker(self):
        """Should use create_tracker() factory for a fresh tracker per invocation."""
        mock_config = MagicMock(spec=AIAgentConfig)
        fresh_tracker = MagicMock(spec=LDAIConfigTracker)
        fresh_tracker.track_metrics_of_async = AsyncMock(
            return_value=RunnerResult(
                content="Fresh tracker response",
                metrics=LDAIMetrics(success=True, usage=None),
                raw=None,
            )
        )
        fresh_tracker.get_summary = MagicMock(return_value=_make_summary(True))
        mock_config.create_tracker = MagicMock(return_value=fresh_tracker)
        mock_config.evaluator = Evaluator.noop()

        mock_runner = MagicMock()

        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

        assert isinstance(result, ManagedResult)
        assert result.content == "Fresh tracker response"
        mock_config.create_tracker.assert_called_once()
        fresh_tracker.track_metrics_of_async.assert_called_once()
        if result.evaluations is not None:
            await result.evaluations

    def test_get_agent_runner_returns_runner(self):
        """Should return the underlying AgentRunner."""
        mock_runner = MagicMock()
        agent = ManagedAgent(MagicMock(), mock_runner)

        assert agent.get_agent_runner() is mock_runner

    def test_get_config_returns_config(self):
        """Should return the AI agent config."""
        mock_config = MagicMock()
        agent = ManagedAgent(mock_config, MagicMock())

        assert agent.get_config() is mock_config


class TestManagedAgentEvaluations:
    """Tests for ManagedAgent evaluations chain (PR 12)."""

    @pytest.mark.asyncio
    async def test_run_returns_before_evaluations_resolve(self):
        """run() should return before evaluations complete."""
        barrier = asyncio.Event()

        async def _slow_evaluate(input_text: str, output_text: str) -> List[JudgeResult]:
            await barrier.wait()
            return []

        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_slow_evaluate(i, o))
        )

        mock_config = MagicMock(spec=AIAgentConfig)
        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(
            return_value=RunnerResult(content="resp", raw=None, metrics=LDAIMetrics(success=True))
        )
        mock_tracker.get_summary = MagicMock(return_value=_make_summary(True))
        mock_config.create_tracker = MagicMock(return_value=mock_tracker)
        mock_config.evaluator = mock_evaluator

        mock_runner = MagicMock()
        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

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
            reasoning='Good agent response',
        )

        async def _evaluate_coro(input_text: str, output_text: str) -> List[JudgeResult]:
            return [judge_result]

        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_evaluate_coro(i, o))
        )

        mock_config = MagicMock(spec=AIAgentConfig)
        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(
            return_value=RunnerResult(content="resp", raw=None, metrics=LDAIMetrics(success=True))
        )
        mock_tracker.get_summary = MagicMock(return_value=_make_summary(True))
        mock_tracker.track_judge_result = MagicMock()
        mock_config.create_tracker = MagicMock(return_value=mock_tracker)
        mock_config.evaluator = mock_evaluator

        mock_runner = MagicMock()
        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

        results = await result.evaluations  # type: ignore[misc]
        assert results == [judge_result]

    @pytest.mark.asyncio
    async def test_tracking_fires_inside_awaited_chain(self):
        """tracker.track_judge_result() must be called when evaluations are awaited."""
        judge_result = JudgeResult(
            judge_config_key='agent-judge',
            success=True,
            sampled=True,
            metric_key='$ld:ai:judge:relevance',
            score=0.85,
        )

        async def _evaluate_coro(input_text: str, output_text: str) -> List[JudgeResult]:
            return [judge_result]

        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_evaluate_coro(i, o))
        )

        mock_config = MagicMock(spec=AIAgentConfig)
        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(
            return_value=RunnerResult(content="resp", raw=None, metrics=LDAIMetrics(success=True))
        )
        mock_tracker.get_summary = MagicMock(return_value=_make_summary(True))
        mock_tracker.track_judge_result = MagicMock()
        mock_config.create_tracker = MagicMock(return_value=mock_tracker)
        mock_config.evaluator = mock_evaluator

        mock_runner = MagicMock()
        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

        # Tracking should NOT have fired yet (before we await evaluations)
        mock_tracker.track_judge_result.assert_not_called()

        # Now await the evaluations task — tracking fires inside the chain
        await result.evaluations  # type: ignore[misc]

        mock_tracker.track_judge_result.assert_called_once_with(judge_result)

    @pytest.mark.asyncio
    async def test_noop_evaluator_returns_empty_list(self):
        """With a noop evaluator, awaiting evaluations should return an empty list."""
        mock_config = _make_noop_evaluator_config()
        mock_runner = MagicMock()
        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

        results = await result.evaluations  # type: ignore[misc]
        assert results == []

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

        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.evaluate = MagicMock(
            side_effect=lambda i, o: asyncio.create_task(_evaluate_coro(i, o))
        )

        mock_config = MagicMock(spec=AIAgentConfig)
        mock_tracker = MagicMock(spec=LDAIConfigTracker)
        mock_tracker.track_metrics_of_async = AsyncMock(
            return_value=RunnerResult(content="resp", raw=None, metrics=LDAIMetrics(success=True))
        )
        mock_tracker.get_summary = MagicMock(return_value=_make_summary(True))
        mock_tracker.track_judge_result = MagicMock()
        mock_config.create_tracker = MagicMock(return_value=mock_tracker)
        mock_config.evaluator = mock_evaluator

        mock_runner = MagicMock()
        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")
        await result.evaluations  # type: ignore[misc]

        mock_tracker.track_judge_result.assert_not_called()


class TestLDAIClientCreateAgent:
    """Tests for LDAIClient.create_agent."""

    @pytest.mark.asyncio
    async def test_returns_none_when_agent_is_disabled(self, ldai_client: LDAIClient):
        """Should return None when agent config is disabled."""
        context = Context.create('user-key')
        result = await ldai_client.create_agent('disabled-agent', context)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_provider_unavailable(self, ldai_client: LDAIClient):
        """Should return None when no AI provider is available."""
        import ldai.providers.runner_factory as rf
        context = Context.create('user-key')

        original = rf.RunnerFactory.create_agent
        rf.RunnerFactory.create_agent = MagicMock(return_value=None)
        try:
            result = await ldai_client.create_agent('customer-support-agent', context)
            assert result is None
        finally:
            rf.RunnerFactory.create_agent = original

    @pytest.mark.asyncio
    async def test_returns_managed_agent_when_runner_available(self, ldai_client: LDAIClient):
        """Should return ManagedAgent when runner is successfully created."""
        import ldai.providers.runner_factory as rf
        context = Context.create('user-key')

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=RunnerResult(content="Hello!", metrics=LDAIMetrics(success=True, usage=None), raw=None)
        )

        original = rf.RunnerFactory.create_agent
        rf.RunnerFactory.create_agent = MagicMock(return_value=mock_runner)
        try:
            result = await ldai_client.create_agent('customer-support-agent', context)
            assert isinstance(result, ManagedAgent)
            assert result.get_agent_runner() is mock_runner
        finally:
            rf.RunnerFactory.create_agent = original
