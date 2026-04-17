"""Tests for ManagedAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ldai import LDAIClient, ManagedAgent
from ldai.managed_agent import ManagedAgent
from ldai.models import AIAgentConfig, AIAgentConfigDefault, ModelConfig, ProviderConfig
from ldai.providers import AgentResult
from ldai.providers.types import LDAIMetrics

from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData


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
        """Should delegate run() to the underlying AgentRunner."""
        mock_config = MagicMock(spec=AIAgentConfig)
        mock_tracker = MagicMock()
        mock_tracker.track_metrics_of_async = AsyncMock(
            return_value=AgentResult(
                output="Test response",
                raw=None,
                metrics=LDAIMetrics(success=True, usage=None),
            )
        )
        mock_config.create_tracker = MagicMock(return_value=mock_tracker)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=AgentResult(
                output="Test response",
                raw=None,
                metrics=LDAIMetrics(success=True, usage=None),
            )
        )

        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

        assert result.output == "Test response"
        assert result.metrics.success is True
        mock_config.create_tracker.assert_called_once()
        mock_tracker.track_metrics_of_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_uses_create_tracker_for_fresh_tracker(self):
        """Should use create_tracker() factory for a fresh tracker per invocation."""
        mock_config = MagicMock(spec=AIAgentConfig)
        fresh_tracker = MagicMock()
        fresh_tracker.track_metrics_of_async = AsyncMock(
            return_value=AgentResult(
                output="Fresh tracker response",
                raw=None,
                metrics=LDAIMetrics(success=True, usage=None),
            )
        )
        mock_config.create_tracker = MagicMock(return_value=fresh_tracker)

        mock_runner = MagicMock()

        agent = ManagedAgent(mock_config, mock_runner)
        result = await agent.run("Hello")

        assert result.output == "Fresh tracker response"
        mock_config.create_tracker.assert_called_once()
        fresh_tracker.track_metrics_of_async.assert_called_once()

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
            return_value=AgentResult(output="Hello!", raw=None, metrics=LDAIMetrics(success=True, usage=None))
        )

        original = rf.RunnerFactory.create_agent
        rf.RunnerFactory.create_agent = MagicMock(return_value=mock_runner)
        try:
            result = await ldai_client.create_agent('customer-support-agent', context)
            assert isinstance(result, ManagedAgent)
            assert result.get_agent_runner() is mock_runner
        finally:
            rf.RunnerFactory.create_agent = original
