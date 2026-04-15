"""Tests for Judge functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai.judge import Judge
from ldai.judge.evaluation_schema_builder import EvaluationSchemaBuilder
from ldai.models import AIJudgeConfig, AIJudgeConfigDefault, LDMessage, ModelConfig, ProviderConfig
from ldai.providers.types import JudgeResult, LDAIMetrics, StructuredResponse
from ldai.tracker import LDAIConfigTracker


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('judge-config')
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.3}},
                'provider': {'name': 'openai'},
                'messages': [{'role': 'system', 'content': 'You are a judge.'}],
                'evaluationMetricKey': '$ld:ai:judge:relevance',
                '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
            }
        )
        .variation_for_all(0)
    )
    return td


@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config('sdk-key', update_processor_class=td, send_events=False)
    return LDClient(config=config)


@pytest.fixture
def mock_runner():
    """Create a mock AI provider."""
    provider = MagicMock()
    provider.invoke_structured_model = AsyncMock()
    return provider


@pytest.fixture
def context() -> Context:
    return Context.create('user-key')


@pytest.fixture
def tracker(client: LDClient, context: Context) -> LDAIConfigTracker:
    return LDAIConfigTracker(
        client, 'judge-v1', 'judge-config', 1, 'gpt-4', 'openai', context
    )


@pytest.fixture
def judge_config_with_key() -> AIJudgeConfig:
    """Create a judge config with evaluation_metric_key."""
    return AIJudgeConfig(
        key='judge-config',
        enabled=True,
        evaluation_metric_key='$ld:ai:judge:relevance',
        messages=[LDMessage(role='system', content='You are a judge.')],
        model=ModelConfig('gpt-4'),
        provider=ProviderConfig('openai'),
    )


@pytest.fixture
def judge_config_without_key() -> AIJudgeConfig:
    """Create a judge config without evaluation_metric_key."""
    return AIJudgeConfig(
        key='judge-config',
        enabled=True,
        evaluation_metric_key=None,
        messages=[LDMessage(role='system', content='You are a judge.')],
        model=ModelConfig('gpt-4'),
        provider=ProviderConfig('openai'),
    )


@pytest.fixture
def judge_config_without_messages() -> AIJudgeConfig:
    """Create a judge config without messages."""
    return AIJudgeConfig(
        key='judge-config',
        enabled=True,
        evaluation_metric_key='$ld:ai:judge:relevance',
        messages=None,
        model=ModelConfig('gpt-4'),
        provider=ProviderConfig('openai'),
    )


class TestJudgeInitialization:
    """Tests for Judge initialization."""

    def test_judge_initializes_with_evaluation_metric_key(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Judge should initialize successfully with evaluation_metric_key."""
        judge = Judge(judge_config_with_key, tracker, mock_runner)
        
        assert judge._ai_config == judge_config_with_key
        assert judge._evaluation_response_structure is not None
        assert judge._evaluation_response_structure['title'] == 'EvaluationResponse'
        assert judge._evaluation_response_structure['required'] == ['score', 'reasoning']
        assert 'score' in judge._evaluation_response_structure['properties']
        assert 'reasoning' in judge._evaluation_response_structure['properties']


class TestJudgeEvaluate:
    """Tests for Judge.evaluate() method."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_failure_when_evaluation_metric_key_missing(
        self, judge_config_without_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should return a failed JudgeResult when evaluation_metric_key is missing."""
        judge = Judge(judge_config_without_key, tracker, mock_runner)

        result = await judge.evaluate("input text", "output text")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.sampled is False
        mock_runner.invoke_structured_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_returns_failure_when_messages_missing(
        self, judge_config_without_messages: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should return a failed JudgeResult when messages are missing."""
        judge = Judge(judge_config_without_messages, tracker, mock_runner)

        result = await judge.evaluate("input text", "output text")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.sampled is False
        mock_runner.invoke_structured_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_success_with_valid_response(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should return JudgeResponse with valid evaluation."""
        mock_response = StructuredResponse(
            data={
                'score': 0.85,
                'reasoning': 'The response is highly relevant to the input.'
            },
            raw_response='{"score": 0.85, "reasoning": "..."}',
            metrics=LDAIMetrics(success=True)
        )
        
        mock_runner.invoke_structured_model.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)
        
        judge = Judge(judge_config_with_key, tracker, mock_runner)
        
        result = await judge.evaluate("What is AI?", "AI is artificial intelligence.")
        
        assert isinstance(result, JudgeResult)
        assert result.success is True
        assert result.metric_key == '$ld:ai:judge:relevance'
        assert result.score == 0.85
        assert result.reasoning is not None
        assert 'relevant' in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_evaluate_success_with_evaluation_response_shape(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should accept shape { score, reasoning } and key by metric."""
        mock_response = StructuredResponse(
            data={
                'score': 0.9,
                'reasoning': 'The response is accurate and complete.',
            },
            raw_response='{"score": 0.9, "reasoning": "..."}',
            metrics=LDAIMetrics(success=True),
        )
        mock_runner.invoke_structured_model.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        judge = Judge(judge_config_with_key, tracker, mock_runner)
        result = await judge.evaluate("What is feature flagging?", "Feature flagging is...")

        assert isinstance(result, JudgeResult)
        assert result.success is True
        assert result.metric_key == '$ld:ai:judge:relevance'
        assert result.score == 0.9
        assert result.reasoning is not None
        assert 'accurate' in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_evaluate_handles_missing_evaluation_in_response(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle missing score/reasoning in response."""
        mock_response = StructuredResponse(
            data={},
            raw_response='{}',
            metrics=LDAIMetrics(success=True)
        )
        
        mock_runner.invoke_structured_model.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)
        
        judge = Judge(judge_config_with_key, tracker, mock_runner)
        
        result = await judge.evaluate("input", "output")
        
        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.score is None

    @pytest.mark.asyncio
    async def test_evaluate_handles_invalid_score(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle invalid score values."""
        mock_response = StructuredResponse(
            data={
                'score': 1.5,
                'reasoning': 'Some reasoning'
            },
            raw_response='{"score": 1.5, "reasoning": "..."}',
            metrics=LDAIMetrics(success=True)
        )
        
        mock_runner.invoke_structured_model.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)
        
        judge = Judge(judge_config_with_key, tracker, mock_runner)
        
        result = await judge.evaluate("input", "output")
        
        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.score is None

    @pytest.mark.asyncio
    async def test_evaluate_handles_missing_reasoning(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle missing reasoning."""
        mock_response = StructuredResponse(
            data={'score': 0.8},
            raw_response='{"score": 0.8}',
            metrics=LDAIMetrics(success=True)
        )
        
        mock_runner.invoke_structured_model.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)
        
        judge = Judge(judge_config_with_key, tracker, mock_runner)
        
        result = await judge.evaluate("input", "output")
        
        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.score is None

    @pytest.mark.asyncio
    async def test_evaluate_handles_exception(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle exceptions gracefully."""
        mock_runner.invoke_structured_model.side_effect = Exception("Provider error")
        tracker.track_metrics_of_async = AsyncMock(side_effect=Exception("Provider error"))
        
        judge = Judge(judge_config_with_key, tracker, mock_runner)
        
        result = await judge.evaluate("input", "output")
        
        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_evaluate_respects_sampling_rate(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should return sampled=True when skipped due to sampling rate."""
        judge = Judge(judge_config_with_key, tracker, mock_runner)

        result = await judge.evaluate("input", "output", sampling_rate=0.0)

        assert isinstance(result, JudgeResult)
        assert result.sampled is True
        assert result.success is False
        mock_runner.invoke_structured_model.assert_not_called()


class TestJudgeEvaluateMessages:
    """Tests for Judge.evaluate_messages() method."""

    @pytest.mark.asyncio
    async def test_evaluate_messages_calls_evaluate(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """evaluate_messages should call evaluate with constructed input/output."""
        from ldai.providers.types import ModelResponse
        
        mock_response = StructuredResponse(
            data={'score': 0.9, 'reasoning': 'Very relevant'},
            raw_response='{"score": 0.9, "reasoning": "..."}',
            metrics=LDAIMetrics(success=True)
        )
        
        mock_runner.invoke_structured_model.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)
        
        judge = Judge(judge_config_with_key, tracker, mock_runner)
        
        messages = [
            LDMessage(role='user', content='Question 1'),
            LDMessage(role='assistant', content='Answer 1'),
        ]
        chat_response = ModelResponse(
            message=LDMessage(role='assistant', content='Answer 2'),
            metrics=LDAIMetrics(success=True)
        )
        
        result = await judge.evaluate_messages(messages, chat_response)
        
        assert result is not None
        assert result.success is True
        assert tracker.track_metrics_of_async.called


class TestEvaluationSchemaBuilder:
    """Tests for EvaluationSchemaBuilder."""

    def test_build_creates_correct_schema(self):
        """Schema builder should create fixed schema (top-level score + reasoning, no key param)."""
        schema = EvaluationSchemaBuilder.build()

        assert schema['title'] == 'EvaluationResponse'
        assert schema['type'] == 'object'
        assert schema['required'] == ['score', 'reasoning']
        assert 'score' in schema['properties']
        assert 'reasoning' in schema['properties']
        assert schema['properties']['score']['type'] == 'number'
        assert schema['properties']['score']['minimum'] == 0
        assert schema['properties']['score']['maximum'] == 1


class TestJudgeConfigSerialization:
    """Tests for AIJudgeConfig serialization."""

    def test_to_dict_includes_evaluation_metric_key(self):
        """to_dict should include evaluationMetricKey."""
        config = AIJudgeConfig(
            key='test-judge',
            enabled=True,
            evaluation_metric_key='$ld:ai:judge:relevance',
            messages=[LDMessage(role='system', content='You are a judge.')],
        )
        
        result = config.to_dict()
        
        assert result['evaluationMetricKey'] == '$ld:ai:judge:relevance'
        assert 'evaluationMetricKeys' not in result

    def test_to_dict_handles_none_evaluation_metric_key(self):
        """to_dict should handle None evaluation_metric_key."""
        config = AIJudgeConfig(
            key='test-judge',
            enabled=True,
            evaluation_metric_key=None,
            messages=[LDMessage(role='system', content='You are a judge.')],
        )
        
        result = config.to_dict()
        
        assert result['evaluationMetricKey'] is None

    def test_judge_config_default_to_dict(self):
        """AIJudgeConfigDefault.to_dict should work correctly."""
        config = AIJudgeConfigDefault(
            enabled=True,
            evaluation_metric_key='$ld:ai:judge:relevance',
            messages=[LDMessage(role='system', content='You are a judge.')],
        )
        
        result = config.to_dict()
        
        assert result['evaluationMetricKey'] == '$ld:ai:judge:relevance'
        assert 'evaluationMetricKeys' not in result


class TestClientJudgeConfig:
    """Tests for LDAIClient.judge_config() method."""

    def test_judge_config_extracts_evaluation_metric_key(
        self, client: LDClient, context: Context
    ):
        """judge_config should extract evaluationMetricKey from variation."""
        from ldai import LDAIClient
        
        ldai_client = LDAIClient(client)
        
        default = AIJudgeConfigDefault(
            enabled=True,
            evaluation_metric_key='$ld:ai:judge:relevance',
            messages=[LDMessage(role='system', content='You are a judge.')],
            model=ModelConfig('gpt-4'),
            provider=ProviderConfig('openai'),
        )
        
        config = ldai_client.judge_config('judge-config', context, default)
        
        assert config is not None
        assert config.evaluation_metric_key == '$ld:ai:judge:relevance'
        assert config.enabled is True
        assert config.messages is not None
        assert len(config.messages) > 0

    def test_judge_config_uses_default_when_flag_does_not_exist(
        self, client: LDClient, context: Context
    ):
        """judge_config should use default evaluation_metric_key when flag does not exist."""
        from ldai import LDAIClient
        from ldclient import Config, LDClient
        from ldclient.integrations.test_data import TestData
        
        td = TestData.data_source()
        
        test_client = LDClient(Config('sdk-key', update_processor_class=td, send_events=False))
        ldai_client = LDAIClient(test_client)
        
        default = AIJudgeConfigDefault(
            enabled=True,
            evaluation_metric_key='$ld:ai:judge:default',
            messages=[LDMessage(role='system', content='You are a judge.')],
            model=ModelConfig('gpt-4'),
            provider=ProviderConfig('openai'),
        )
        
        config = ldai_client.judge_config('judge-no-key', context, default)
        
        assert config is not None
        assert config.evaluation_metric_key == '$ld:ai:judge:default'

    def test_judge_config_uses_first_evaluation_metric_keys_from_variation(
        self, context: Context
    ):
        """judge_config should use first value from evaluationMetricKeys when evaluationMetricKey is None."""
        from ldai import LDAIClient
        from ldclient import Config, LDClient
        from ldclient.integrations.test_data import TestData
        
        td = TestData.data_source()
        td.update(
            td.flag('judge-with-keys')
            .variations(
                {
                    'model': {'name': 'gpt-4'},
                    'provider': {'name': 'openai'},
                    'messages': [{'role': 'system', 'content': 'You are a judge.'}],
                    'evaluationMetricKeys': ['$ld:ai:judge:relevance', '$ld:ai:judge:quality'],
                    '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
                }
            )
            .variation_for_all(0)
        )
        
        test_client = LDClient(Config('sdk-key', update_processor_class=td, send_events=False))
        ldai_client = LDAIClient(test_client)
        
        default = AIJudgeConfigDefault(
            enabled=True,
            evaluation_metric_key=None,
            messages=[LDMessage(role='system', content='You are a judge.')],
            model=ModelConfig('gpt-4'),
            provider=ProviderConfig('openai'),
        )
        
        config = ldai_client.judge_config('judge-with-keys', context, default)
        
        assert config is not None
        assert config.evaluation_metric_key == '$ld:ai:judge:relevance'

    def test_judge_config_uses_first_evaluation_metric_keys_from_default(
        self, context: Context
    ):
        """judge_config should use first value from default evaluation_metric_keys when flag does not exist."""
        from ldai import LDAIClient
        from ldclient import Config, LDClient
        from ldclient.integrations.test_data import TestData
        
        td = TestData.data_source()
        
        test_client = LDClient(Config('sdk-key', update_processor_class=td, send_events=False))
        ldai_client = LDAIClient(test_client)
        
        default = AIJudgeConfigDefault(
            enabled=True,
            evaluation_metric_key=None,
            evaluation_metric_keys=['$ld:ai:judge:default-key', '$ld:ai:judge:secondary'],
            messages=[LDMessage(role='system', content='You are a judge.')],
            model=ModelConfig('gpt-4'),
            provider=ProviderConfig('openai'),
        )
        
        config = ldai_client.judge_config('judge-fallback-keys', context, default)
        
        assert config is not None
        assert config.evaluation_metric_key == '$ld:ai:judge:default-key'

    def test_judge_config_prefers_evaluation_metric_key_over_keys(
        self, context: Context
    ):
        """judge_config should prefer evaluationMetricKey over evaluationMetricKeys when both are present."""
        from ldai import LDAIClient
        from ldclient import Config, LDClient
        from ldclient.integrations.test_data import TestData
        
        td = TestData.data_source()
        td.update(
            td.flag('judge-both-present')
            .variations(
                {
                    'model': {'name': 'gpt-4'},
                    'provider': {'name': 'openai'},
                    'messages': [{'role': 'system', 'content': 'You are a judge.'}],
                    'evaluationMetricKey': '$ld:ai:judge:preferred',
                    'evaluationMetricKeys': ['$ld:ai:judge:relevance', '$ld:ai:judge:quality'],
                    '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
                }
            )
            .variation_for_all(0)
        )
        
        test_client = LDClient(Config('sdk-key', update_processor_class=td, send_events=False))
        ldai_client = LDAIClient(test_client)
        
        default = AIJudgeConfigDefault(
            enabled=True,
            evaluation_metric_key=None,
            messages=[LDMessage(role='system', content='You are a judge.')],
            model=ModelConfig('gpt-4'),
            provider=ProviderConfig('openai'),
        )
        
        config = ldai_client.judge_config('judge-both-present', context, default)
        
        assert config is not None
        assert config.evaluation_metric_key == '$ld:ai:judge:preferred'

    def test_judge_config_without_default_uses_disabled(
        self, context: Context
    ):
        """judge_config should use a disabled config when no default is provided."""
        from ldai import LDAIClient
        from ldclient import Config, LDClient
        from ldclient.integrations.test_data import TestData

        td = TestData.data_source()
        test_client = LDClient(Config('sdk-key', update_processor_class=td, send_events=False))
        ldai_client = LDAIClient(test_client)

        config = ldai_client.judge_config('missing-judge', context)

        assert config is not None
        assert config.enabled is False

    def test_judge_config_uses_same_variation_for_consistency(
        self, context: Context
    ):
        """judge_config should use the same variation from __evaluate to avoid race conditions."""
        from ldai import LDAIClient
        from ldclient import Config, LDClient
        from ldclient.integrations.test_data import TestData
        from unittest.mock import patch

        td = TestData.data_source()
        td.update(
            td.flag('judge-consistency-test')
            .variations(
                {
                    'model': {'name': 'gpt-4'},
                    'provider': {'name': 'openai'},
                    'messages': [{'role': 'system', 'content': 'You are a judge.'}],
                    'evaluationMetricKey': '$ld:ai:judge:from-flag',
                    '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
                }
            )
            .variation_for_all(0)
        )

        test_client = LDClient(Config('sdk-key', update_processor_class=td, send_events=False))
        ldai_client = LDAIClient(test_client)

        default = AIJudgeConfigDefault(
            enabled=True,
            evaluation_metric_key='$ld:ai:judge:from-default',
            messages=[LDMessage(role='system', content='You are a judge.')],
            model=ModelConfig('gpt-4'),
            provider=ProviderConfig('openai'),
        )

        variation_calls = []
        original_variation = test_client.variation

        def tracked_variation(key, context, default):
            result = original_variation(key, context, default)
            variation_calls.append((key, result.get('evaluationMetricKey')))
            return result

        with patch.object(test_client, 'variation', side_effect=tracked_variation):
            config = ldai_client.judge_config('judge-consistency-test', context, default)

        assert len(variation_calls) == 1, f"Expected 1 variation call, got {len(variation_calls)}"
        assert config is not None
        assert config.evaluation_metric_key == '$ld:ai:judge:from-flag'
