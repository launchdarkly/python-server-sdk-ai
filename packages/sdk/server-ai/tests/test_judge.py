"""Tests for Judge functionality."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai.judge import Judge, _strip_legacy_judge_messages
from ldai.judge.evaluation_schema_builder import EvaluationSchemaBuilder
from ldai.models import (
    AIJudgeConfig,
    AIJudgeConfigDefault,
    LDMessage,
    ModelConfig,
    ProviderConfig,
)
from ldai.providers.types import JudgeResult, LDAIMetrics, RunnerResult
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
    """Create a mock AI runner."""
    provider = MagicMock()
    provider.run = AsyncMock()
    return provider


@pytest.fixture
def context() -> Context:
    return Context.create('user-key')


@pytest.fixture
def tracker(client: LDClient, context: Context) -> LDAIConfigTracker:
    return LDAIConfigTracker(
        ld_client=client, run_id='test-run-id', config_key='judge-config',
        variation_key='judge-v1', version=1, model_name='gpt-4',
        provider_name='openai', context=context,
    )


_SENTINEL = object()


def _make_judge_config(
    key='judge-config',
    enabled=True,
    evaluation_metric_key='$ld:ai:judge:relevance',
    messages=_SENTINEL,
    model=None,
    provider=None,
    tracker=None,
):
    """Create a judge config with create_tracker wired up."""
    if messages is _SENTINEL:
        messages = [LDMessage(role='system', content='You are a judge.')]
    kwargs = dict(
        key=key,
        enabled=enabled,
        evaluation_metric_key=evaluation_metric_key,
        messages=messages,
        model=model or ModelConfig('gpt-4'),
        provider=provider or ProviderConfig('openai'),
    )
    kwargs['create_tracker'] = (lambda: tracker) if tracker is not None else MagicMock()
    return AIJudgeConfig(**kwargs)


@pytest.fixture
def judge_config_with_key(tracker) -> AIJudgeConfig:
    """Create a judge config with evaluation_metric_key."""
    return _make_judge_config(tracker=tracker)


@pytest.fixture
def judge_config_without_key(tracker) -> AIJudgeConfig:
    """Create a judge config without evaluation_metric_key."""
    return _make_judge_config(evaluation_metric_key=None, tracker=tracker)


@pytest.fixture
def judge_config_without_messages(tracker) -> AIJudgeConfig:
    """Create a judge config without messages (None)."""
    return _make_judge_config(messages=None, tracker=tracker)


class TestStripLegacyJudgeMessages:
    """Tests for the _strip_legacy_judge_messages helper."""

    def test_strips_assistant_message_with_message_history(self):
        """Non-system messages containing {{message_history}} should be removed."""
        messages = [
            LDMessage(role='system', content='You are a judge.'),
            LDMessage(role='assistant', content='Here is the history: {{message_history}}'),
        ]
        result = _strip_legacy_judge_messages(messages)
        assert len(result) == 1
        assert result[0].role == 'system'

    def test_strips_user_message_with_response_to_evaluate(self):
        """Non-system messages containing {{response_to_evaluate}} should be removed."""
        messages = [
            LDMessage(role='system', content='You are a judge.'),
            LDMessage(role='user', content='Evaluate: {{response_to_evaluate}}'),
        ]
        result = _strip_legacy_judge_messages(messages)
        assert len(result) == 1
        assert result[0].role == 'system'

    def test_strips_all_legacy_messages(self):
        """All non-system template messages should be stripped from a typical legacy config."""
        messages = [
            LDMessage(role='system', content='You are a judge.'),
            LDMessage(role='assistant', content='{{message_history}}'),
            LDMessage(role='user', content='{{response_to_evaluate}}'),
        ]
        result = _strip_legacy_judge_messages(messages)
        assert len(result) == 1
        assert result[0].role == 'system'

    def test_does_not_strip_system_message_containing_template_vars(self):
        """System messages are never stripped, even if they contain template variable names."""
        messages = [
            LDMessage(role='system', content='Judge using {{message_history}} and {{response_to_evaluate}}.'),
        ]
        result = _strip_legacy_judge_messages(messages)
        assert len(result) == 1
        assert result[0].role == 'system'

    def test_does_not_strip_non_template_messages(self):
        """Non-system messages without template variables are left untouched."""
        messages = [
            LDMessage(role='system', content='You are a judge.'),
            LDMessage(role='user', content='This is a regular message.'),
        ]
        result = _strip_legacy_judge_messages(messages)
        assert len(result) == 2

    def test_returns_empty_list_for_empty_input(self):
        """An empty input list should return an empty list."""
        assert _strip_legacy_judge_messages([]) == []

    def test_new_style_config_system_only_unchanged(self):
        """A new-style config with only a system message passes through unchanged."""
        messages = [LDMessage(role='system', content='You are a judge.')]
        result = _strip_legacy_judge_messages(messages)
        assert result == messages


class TestJudgeInitialization:
    """Tests for Judge initialization."""

    def test_judge_initializes_with_evaluation_metric_key(
        self, judge_config_with_key: AIJudgeConfig, mock_runner
    ):
        """Judge should initialize successfully with evaluation_metric_key."""
        judge = Judge(judge_config_with_key, mock_runner)

        assert judge._ai_config == judge_config_with_key
        assert judge._evaluation_response_structure is not None
        assert judge._evaluation_response_structure['title'] == 'EvaluationResponse'
        assert judge._evaluation_response_structure['required'] == ['score', 'reasoning']
        assert 'score' in judge._evaluation_response_structure['properties']
        assert 'reasoning' in judge._evaluation_response_structure['properties']

    def test_judge_sample_rate_defaults_to_one(
        self, judge_config_with_key: AIJudgeConfig, mock_runner
    ):
        """sample_rate should default to 1.0 when not provided."""
        judge = Judge(judge_config_with_key, mock_runner)
        assert judge.sample_rate == 1.0

    def test_judge_sample_rate_can_be_set(
        self, judge_config_with_key: AIJudgeConfig, mock_runner
    ):
        """sample_rate should be settable via the constructor."""
        judge = Judge(judge_config_with_key, mock_runner, sample_rate=0.25)
        assert judge.sample_rate == 0.25


class TestJudgeEvaluate:
    """Tests for Judge.evaluate() method."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_failure_when_evaluation_metric_key_missing(
        self, judge_config_without_key: AIJudgeConfig, mock_runner
    ):
        """Evaluate should return a failed JudgeResult when evaluation_metric_key is missing."""
        judge = Judge(judge_config_without_key, mock_runner)

        result = await judge.evaluate("input text", "output text")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.sampled is False
        mock_runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_succeeds_when_messages_is_none(
        self, judge_config_without_messages: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should proceed (not error early) when messages is None."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.7, 'reasoning': 'Acceptable response.'},
        )
        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        config = _make_judge_config(messages=None, tracker=tracker)
        judge = Judge(config, mock_runner)

        result = await judge.evaluate("input text", "output text")

        assert isinstance(result, JudgeResult)
        assert result.sampled is True

    @pytest.mark.asyncio
    async def test_evaluate_passes_string_input_to_runner(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """runner.run() should receive the formatted string, NOT a message list."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.85, 'reasoning': 'Good answer.'},
        )
        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(
            side_effect=lambda _metric_fn, fn: fn()
        )

        judge = Judge(judge_config_with_key, mock_runner)
        await judge.evaluate("What is AI?", "AI is artificial intelligence.")

        mock_runner.run.assert_called_once()
        call_args = mock_runner.run.call_args
        input_arg = call_args[0][0] if call_args[0] else call_args[1].get('input')
        assert isinstance(input_arg, str)
        assert "MESSAGE HISTORY:\nWhat is AI?" in input_arg
        assert "RESPONSE TO EVALUATE:\nAI is artificial intelligence." in input_arg

    @pytest.mark.asyncio
    async def test_evaluate_string_input_format(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """runner.run() should receive the exact expected string format."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.9, 'reasoning': 'Correct.'},
        )
        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(
            side_effect=lambda _metric_fn, fn: fn()
        )

        judge = Judge(judge_config_with_key, mock_runner)
        await judge.evaluate("hello", "world")

        call_args = mock_runner.run.call_args
        input_arg = call_args[0][0] if call_args[0] else call_args[1].get('input')
        expected = "MESSAGE HISTORY:\nhello\n\nRESPONSE TO EVALUATE:\nworld"
        assert input_arg == expected

    @pytest.mark.asyncio
    async def test_evaluate_legacy_config_passes_string_input_to_runner(
        self, tracker: LDAIConfigTracker, mock_runner
    ):
        """
        Judge built directly with legacy messages (bypassing the client) still passes
        a formatted string to the runner.  Legacy message stripping is the client's
        responsibility; the Judge itself does not strip.
        """
        legacy_messages = [
            LDMessage(role='system', content='You are a strict judge.'),
            LDMessage(role='assistant', content='{{message_history}}'),
            LDMessage(role='user', content='Evaluate: {{response_to_evaluate}}'),
        ]
        config = _make_judge_config(messages=legacy_messages, tracker=tracker)

        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.75, 'reasoning': 'Mostly relevant.'},
        )
        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(
            side_effect=lambda _metric_fn, fn: fn()
        )

        judge = Judge(config, mock_runner)
        await judge.evaluate("input", "output")

        call_args = mock_runner.run.call_args
        input_arg = call_args[0][0] if call_args[0] else call_args[1].get('input')
        assert isinstance(input_arg, str)
        assert "MESSAGE HISTORY:\ninput" in input_arg
        assert "RESPONSE TO EVALUATE:\noutput" in input_arg

    @pytest.mark.asyncio
    async def test_evaluate_success_with_valid_response(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should return JudgeResponse with valid evaluation."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={
                'score': 0.85,
                'reasoning': 'The response is highly relevant to the input.'
            },
        )

        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        judge = Judge(judge_config_with_key, mock_runner)

        result = await judge.evaluate("What is AI?", "AI is artificial intelligence.")

        assert isinstance(result, JudgeResult)
        assert result.success is True
        assert result.sampled is True
        assert result.metric_key == '$ld:ai:judge:relevance'
        assert result.score == 0.85
        assert result.reasoning is not None
        assert 'relevant' in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_evaluate_success_with_evaluation_response_shape(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should accept shape { score, reasoning } and key by metric."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={
                'score': 0.9,
                'reasoning': 'The response is accurate and complete.',
            },
        )
        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        judge = Judge(judge_config_with_key, mock_runner)
        result = await judge.evaluate("What is feature flagging?", "Feature flagging is...")

        assert isinstance(result, JudgeResult)
        assert result.success is True
        assert result.sampled is True
        assert result.metric_key == '$ld:ai:judge:relevance'
        assert result.score == 0.9
        assert result.reasoning is not None
        assert 'accurate' in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_evaluate_handles_missing_evaluation_in_response(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle missing score/reasoning in response."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={},
        )

        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        judge = Judge(judge_config_with_key, mock_runner)

        result = await judge.evaluate("input", "output")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.score is None

    @pytest.mark.asyncio
    async def test_evaluate_handles_invalid_score(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle invalid score values."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={
                'score': 1.5,
                'reasoning': 'Some reasoning',
            },
        )

        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        judge = Judge(judge_config_with_key, mock_runner)

        result = await judge.evaluate("input", "output")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.score is None

    @pytest.mark.asyncio
    async def test_evaluate_handles_missing_reasoning(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle missing reasoning."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.8},
        )

        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        judge = Judge(judge_config_with_key, mock_runner)

        result = await judge.evaluate("input", "output")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.score is None

    @pytest.mark.asyncio
    async def test_evaluate_handles_exception(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """Evaluate should handle exceptions gracefully."""
        mock_runner.run.side_effect = Exception("Provider error")
        tracker.track_metrics_of_async = AsyncMock(side_effect=Exception("Provider error"))

        judge = Judge(judge_config_with_key, mock_runner)

        result = await judge.evaluate("input", "output")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_evaluate_respects_sampling_rate(
        self, judge_config_with_key: AIJudgeConfig, mock_runner
    ):
        """Evaluate should return sampled=False when skipped due to sampling rate."""
        judge = Judge(judge_config_with_key, mock_runner)

        result = await judge.evaluate("input", "output", sampling_rate=0.0)

        assert isinstance(result, JudgeResult)
        assert result.sampled is False
        assert result.success is False
        mock_runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_uses_instance_sample_rate_when_arg_omitted(
        self, judge_config_with_key: AIJudgeConfig, mock_runner
    ):
        """When sampling_rate arg is omitted, the instance's sample_rate is used."""
        judge = Judge(judge_config_with_key, mock_runner, sample_rate=0.0)

        result = await judge.evaluate("input", "output")

        assert isinstance(result, JudgeResult)
        assert result.sampled is False
        mock_runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_arg_overrides_instance_sample_rate(
        self, judge_config_with_key: AIJudgeConfig, mock_runner
    ):
        """An explicit sampling_rate=0.0 must override an instance sample_rate of 1.0."""
        judge = Judge(judge_config_with_key, mock_runner, sample_rate=1.0)

        result = await judge.evaluate("input", "output", sampling_rate=0.0)

        assert isinstance(result, JudgeResult)
        assert result.sampled is False
        mock_runner.run.assert_not_called()


class TestJudgeEvaluateMessages:
    """Tests for Judge.evaluate_messages() method."""

    @pytest.mark.asyncio
    async def test_evaluate_messages_calls_evaluate(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """evaluate_messages should call evaluate with constructed input/output."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.9, 'reasoning': 'Very relevant'},
        )

        mock_runner.run.return_value = mock_response
        tracker.track_metrics_of_async = AsyncMock(return_value=mock_response)

        judge = Judge(judge_config_with_key, mock_runner)

        messages = [
            LDMessage(role='user', content='Question 1'),
            LDMessage(role='assistant', content='Answer 1'),
        ]
        chat_response = RunnerResult(
            content='Answer 2',
            metrics=LDAIMetrics(success=True),
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
            create_tracker=MagicMock(),
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
            create_tracker=MagicMock(),
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
