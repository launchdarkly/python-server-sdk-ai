"""Tests for Judge functionality."""

from typing import List
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient
from ldai.judge import Judge
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
    async def test_evaluate_returns_failure_when_messages_is_none(
        self, judge_config_without_messages: AIJudgeConfig, mock_runner
    ):
        """Evaluate should return an error result when messages is None."""
        judge = Judge(judge_config_without_messages, mock_runner)

        result = await judge.evaluate("input text", "output text")

        assert isinstance(result, JudgeResult)
        assert result.success is False
        assert result.sampled is False
        assert result.error_message is not None
        mock_runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_passes_list_of_messages_to_runner(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker, mock_runner
    ):
        """runner.run() should receive a List[LDMessage], not a plain string."""
        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.85, 'reasoning': 'Good answer.'},
        )
        mock_runner.run.return_value = mock_response

        async def _await_fn(_metric_fn, fn):
            return await fn()

        tracker.track_metrics_of_async = AsyncMock(side_effect=_await_fn)

        judge = Judge(judge_config_with_key, mock_runner)
        await judge.evaluate("What is AI?", "AI is artificial intelligence.")

        mock_runner.run.assert_called_once()
        call_args = mock_runner.run.call_args
        input_arg = call_args[0][0] if call_args[0] else call_args[1].get('input')
        assert isinstance(input_arg, list)
        assert all(isinstance(m, LDMessage) for m in input_arg)

    @pytest.mark.asyncio
    async def test_evaluate_interpolates_message_history_and_response(
        self, tracker: LDAIConfigTracker, mock_runner
    ):
        """Config messages with {{message_history}} and {{response_to_evaluate}} are interpolated."""
        config = _make_judge_config(
            messages=[
                LDMessage(role='system', content='You are a judge.'),
                LDMessage(role='assistant', content='{{message_history}}'),
                LDMessage(role='user', content='Evaluate: {{response_to_evaluate}}'),
            ],
            tracker=tracker,
        )

        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.75, 'reasoning': 'Mostly relevant.'},
        )
        mock_runner.run.return_value = mock_response

        async def _await_fn(_metric_fn, fn):
            return await fn()

        tracker.track_metrics_of_async = AsyncMock(side_effect=_await_fn)

        judge = Judge(config, mock_runner)
        await judge.evaluate("user asked this", "AI said that")

        call_args = mock_runner.run.call_args
        messages_arg: List[LDMessage] = call_args[0][0] if call_args[0] else call_args[1].get('input')

        assert len(messages_arg) == 3
        assert messages_arg[0].role == 'system'
        assert messages_arg[0].content == 'You are a judge.'
        assert messages_arg[1].role == 'assistant'
        assert messages_arg[1].content == 'user asked this'
        assert messages_arg[2].role == 'user'
        assert messages_arg[2].content == 'Evaluate: AI said that'

    @pytest.mark.asyncio
    async def test_evaluate_sends_all_config_messages_to_runner(
        self, tracker: LDAIConfigTracker, mock_runner
    ):
        """All config messages (including non-system) must be passed to the runner."""
        config = _make_judge_config(
            messages=[
                LDMessage(role='system', content='You are a judge.'),
                LDMessage(role='assistant', content='{{message_history}}'),
                LDMessage(role='user', content='{{response_to_evaluate}}'),
            ],
            tracker=tracker,
        )

        mock_response = RunnerResult(
            content='',
            metrics=LDAIMetrics(success=True),
            parsed={'score': 0.8, 'reasoning': 'Good.'},
        )
        mock_runner.run.return_value = mock_response

        async def _await_fn(_metric_fn, fn):
            return await fn()

        tracker.track_metrics_of_async = AsyncMock(side_effect=_await_fn)

        judge = Judge(config, mock_runner)
        await judge.evaluate("input", "output")

        call_args = mock_runner.run.call_args
        messages_arg: List[LDMessage] = call_args[0][0] if call_args[0] else call_args[1].get('input')

        assert len(messages_arg) == 3

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

    @pytest.mark.asyncio
    async def test_evaluate_messages_joins_content_without_role(
        self, judge_config_with_key: AIJudgeConfig, mock_runner
    ):
        """evaluate_messages must forward message content (joined by CRLF) to evaluate()."""
        messages = [
            LDMessage(role='user', content='hi'),
            LDMessage(role='assistant', content='hello'),
        ]
        chat_response = RunnerResult(content='reply', metrics=LDAIMetrics(success=True))

        judge = Judge(judge_config_with_key, mock_runner)
        with patch.object(judge, 'evaluate', new=AsyncMock(return_value=JudgeResult(judge_config_key='judge-config'))) as mock_evaluate:
            await judge.evaluate_messages(messages, chat_response)

        mock_evaluate.assert_called_once()
        args, _ = mock_evaluate.call_args
        assert args[0] == 'hi\r\nhello'
        assert args[1] == 'reply'


class TestJudgeRunnerNonMultiTurn:
    """Successive evaluate() calls must not contaminate each other.

    The Judge shares one runner across evaluations, so the runner must be
    stateless across calls — RunnerFactory.create_model(..., multi_turn=False)
    is what guarantees that at the client layer. These tests verify the Judge
    itself does not accidentally mutate the runner's history and that two
    evaluations see the same baseline.
    """

    @pytest.mark.asyncio
    async def test_two_evaluations_do_not_contaminate_history(
        self, judge_config_with_key: AIJudgeConfig, tracker: LDAIConfigTracker
    ):
        """A judge bound to a non-multi-turn runner must run the same baseline twice."""
        seen_baselines: List[List[LDMessage]] = []

        class _FakeRunner:
            def __init__(self):
                self._history: List[LDMessage] = []
                self._multi_turn = False

            async def run(self, input, output_type=None):  # type: ignore[no-untyped-def]
                seen_baselines.append(list(self._history))
                return RunnerResult(
                    content='ok',
                    metrics=LDAIMetrics(success=True),
                    parsed={'score': 0.9, 'reasoning': 'fine'},
                )

        runner = _FakeRunner()

        async def _await_fn(_metric_fn, fn):
            return await fn()

        tracker.track_metrics_of_async = AsyncMock(side_effect=_await_fn)
        judge = Judge(judge_config_with_key, runner)  # type: ignore[arg-type]

        await judge.evaluate('first input', 'first output')
        await judge.evaluate('second input', 'second output')

        assert len(seen_baselines) == 2
        assert seen_baselines[0] == seen_baselines[1]
        assert runner._history == []


class TestJudgeConfigMessages:
    """Tests for LDAIClient.judge_config() preserving all config messages.

    judge_config() must NOT strip legacy template messages — the judge itself
    is responsible for interpolating {{message_history}} and
    {{response_to_evaluate}} when it constructs the evaluation message list.
    """

    @pytest.fixture
    def context(self) -> Context:
        return Context.create('user-key')

    def _make_client(self, td: TestData) -> LDAIClient:
        config = Config('sdk-key', update_processor_class=td, send_events=False)
        return LDAIClient(LDClient(config=config))

    def test_judge_config_preserves_legacy_messages(self, context):
        """Calling judge_config() must return all config messages, including legacy template ones."""
        td = TestData.data_source()
        td.update(
            td.flag('legacy-judge')
            .variations({
                'model': {'name': 'gpt-4'},
                'provider': {'name': 'openai'},
                'messages': [
                    {'role': 'system', 'content': 'You are a judge.'},
                    {'role': 'assistant', 'content': '{{message_history}}'},
                    {'role': 'user', 'content': 'Evaluate: {{response_to_evaluate}}'},
                ],
                'evaluationMetricKey': '$ld:ai:judge:relevance',
                '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
            })
            .variation_for_all(0)
        )
        client = self._make_client(td)

        result = client.judge_config('legacy-judge', context)

        assert result.enabled is True
        assert result.messages is not None
        assert len(result.messages) == 3
        assert result.messages[0].role == 'system'
        assert result.messages[1].role == 'assistant'
        assert result.messages[2].role == 'user'

    def test_judge_config_passes_user_variables_to_template(self, context):
        """User variables are still interpolated into the system message."""
        td = TestData.data_source()
        td.update(
            td.flag('parametric-judge')
            .variations({
                'model': {'name': 'gpt-4'},
                'provider': {'name': 'openai'},
                'messages': [
                    {'role': 'system', 'content': 'You are a {{tone}} judge.'},
                ],
                'evaluationMetricKey': '$ld:ai:judge:relevance',
                '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
            })
            .variation_for_all(0)
        )
        client = self._make_client(td)

        result = client.judge_config(
            'parametric-judge', context, variables={'tone': 'strict'}
        )

        assert result.messages is not None
        assert result.messages[0].content == 'You are a strict judge.'

    def test_judge_config_warns_on_reserved_variables(self, context):
        """_judge_config warns when callers pass reserved variable names."""
        td = TestData.data_source()
        td.update(
            td.flag('judge-config')
            .variations({
                'model': {'name': 'gpt-4'},
                'provider': {'name': 'openai'},
                'messages': [{'role': 'system', 'content': 'You are a judge.'}],
                'evaluationMetricKey': '$ld:ai:judge:relevance',
                '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
            })
            .variation_for_all(0)
        )
        client = self._make_client(td)

        with patch('ldai.client.log') as mock_log:
            client.judge_config(
                'judge-config',
                context,
                variables={
                    'message_history': 'should be ignored',
                    'response_to_evaluate': 'should be ignored',
                },
            )

        warning_messages = [c.args[0] for c in mock_log.warning.call_args_list]
        assert any("'message_history' is reserved" in m for m in warning_messages)
        assert any("'response_to_evaluate' is reserved" in m for m in warning_messages)

    def test_judge_config_does_not_warn_without_reserved_variables(self, context):
        """No warnings should be emitted when callers pass non-reserved variables."""
        td = TestData.data_source()
        td.update(
            td.flag('judge-config')
            .variations({
                'model': {'name': 'gpt-4'},
                'provider': {'name': 'openai'},
                'messages': [{'role': 'system', 'content': 'You are a judge.'}],
                'evaluationMetricKey': '$ld:ai:judge:relevance',
                '_ldMeta': {'enabled': True, 'variationKey': 'judge-v1', 'version': 1},
            })
            .variation_for_all(0)
        )
        client = self._make_client(td)

        with patch('ldai.client.log') as mock_log:
            client.judge_config('judge-config', context, variables={'tone': 'strict'})

        warning_messages = [c.args[0] for c in mock_log.warning.call_args_list]
        assert not any("'message_history' is reserved" in m for m in warning_messages)
        assert not any("'response_to_evaluate' is reserved" in m for m in warning_messages)


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
