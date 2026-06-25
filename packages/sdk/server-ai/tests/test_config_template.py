"""Tests for *_template config methods (un-interpolated variants)."""

from unittest.mock import Mock

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import (
    LDAIClient,
    LDMessage,
    ModelConfig,
    ProviderConfig,
)
from ldai.models import (
    AIAgentConfigDefault,
    AICompletionConfigDefault,
    AIJudgeConfigDefault,
)


# ---------------------------------------------------------------------------
# Shared flag data
# ---------------------------------------------------------------------------

_COMPLETION_FLAG_KEY = 'completion-template-flag'
_AGENT_FLAG_KEY = 'agent-template-flag'
_JUDGE_FLAG_KEY = 'judge-template-flag'


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()

    td.update(
        td.flag(_COMPLETION_FLAG_KEY)
        .variations(
            {
                'model': {'name': 'fakeModel', 'parameters': {'temperature': 0.5}},
                'provider': {'name': 'fakeProvider'},
                'messages': [
                    {'role': 'system', 'content': 'Hello, {{name}}!'},
                    {'role': 'user', 'content': 'Today is {{day}}.'},
                ],
                '_ldMeta': {'enabled': True, 'variationKey': 'abcd', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag(_AGENT_FLAG_KEY)
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.3}},
                'provider': {'name': 'openai'},
                'instructions': 'You are a helpful assistant for {{company}}.',
                '_ldMeta': {'enabled': True, 'variationKey': 'agent-v1', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag(_JUDGE_FLAG_KEY)
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.1}},
                'provider': {'name': 'openai'},
                'messages': [
                    {'role': 'system', 'content': 'Evaluate the following: {{topic}}'},
                ],
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
def ldai_client(client: LDClient) -> LDAIClient:
    return LDAIClient(client)


@pytest.fixture
def context() -> Context:
    return Context.create('user-key')


# ---------------------------------------------------------------------------
# completion_config_template
# ---------------------------------------------------------------------------

def test_completion_config_template_preserves_placeholders(ldai_client: LDAIClient, context: Context):
    config = ldai_client.completion_config_template(_COMPLETION_FLAG_KEY, context)

    assert config.messages is not None
    assert config.messages[0].content == 'Hello, {{name}}!'
    assert config.messages[1].content == 'Today is {{day}}.'


def test_completion_config_template_returns_typed_config(ldai_client: LDAIClient, context: Context):
    from ldai.models import AICompletionConfig
    config = ldai_client.completion_config_template(_COMPLETION_FLAG_KEY, context)

    assert isinstance(config, AICompletionConfig)
    assert config.enabled is True
    assert config.model is not None
    assert config.model.name == 'fakeModel'
    assert config.provider is not None
    assert config.provider.name == 'fakeProvider'


def test_completion_config_template_differs_from_rendered(ldai_client: LDAIClient, context: Context):
    """Template variant should return raw placeholders; regular variant renders them."""
    template = ldai_client.completion_config_template(_COMPLETION_FLAG_KEY, context)
    rendered = ldai_client.completion_config(_COMPLETION_FLAG_KEY, context, variables={'name': 'World', 'day': 'Monday'})

    assert template.messages is not None
    assert rendered.messages is not None
    assert template.messages[0].content == 'Hello, {{name}}!'
    assert rendered.messages[0].content == 'Hello, World!'


def test_completion_config_template_tracking(context: Context):
    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
        'model': {'name': 'test-model'},
        'messages': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
    }

    ldai = LDAIClient(mock_client)
    ldai.completion_config_template('my-flag', context)

    mock_client.track.assert_any_call(
        '$ld:ai:usage:completion-config-template',
        context,
        'my-flag',
        1,
    )


def test_completion_config_template_uses_default_when_flag_missing(ldai_client: LDAIClient, context: Context):
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('default-model'),
        messages=[LDMessage(role='system', content='Fallback: {{placeholder}}')],
    )
    config = ldai_client.completion_config_template('missing-flag', context, default)

    assert config.messages is not None
    assert config.messages[0].content == 'Fallback: {{placeholder}}'


# ---------------------------------------------------------------------------
# agent_config_template
# ---------------------------------------------------------------------------

def test_agent_config_template_preserves_placeholders(ldai_client: LDAIClient, context: Context):
    config = ldai_client.agent_config_template(_AGENT_FLAG_KEY, context)

    assert config.instructions == 'You are a helpful assistant for {{company}}.'


def test_agent_config_template_returns_typed_config(ldai_client: LDAIClient, context: Context):
    from ldai.models import AIAgentConfig
    config = ldai_client.agent_config_template(_AGENT_FLAG_KEY, context)

    assert isinstance(config, AIAgentConfig)
    assert config.enabled is True
    assert config.model is not None
    assert config.model.name == 'gpt-4'
    assert config.provider is not None
    assert config.provider.name == 'openai'


def test_agent_config_template_differs_from_rendered(ldai_client: LDAIClient, context: Context):
    """Template variant should return raw placeholders; regular variant renders them."""
    template = ldai_client.agent_config_template(_AGENT_FLAG_KEY, context)
    rendered = ldai_client.agent_config(_AGENT_FLAG_KEY, context, variables={'company': 'Acme'})

    assert template.instructions == 'You are a helpful assistant for {{company}}.'
    assert rendered.instructions == 'You are a helpful assistant for Acme.'


def test_agent_config_template_tracking(context: Context):
    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
        'model': {'name': 'test-model'},
        'instructions': 'Do something for {{entity}}.',
    }

    ldai = LDAIClient(mock_client)
    ldai.agent_config_template('my-agent-flag', context)

    mock_client.track.assert_any_call(
        '$ld:ai:usage:agent-config-template',
        context,
        'my-agent-flag',
        1,
    )


def test_agent_config_template_uses_default_when_flag_missing(ldai_client: LDAIClient, context: Context):
    default = AIAgentConfigDefault(
        enabled=True,
        model=ModelConfig('default-model'),
        instructions='Default: {{placeholder}}',
    )
    config = ldai_client.agent_config_template('missing-flag', context, default)

    assert config.instructions == 'Default: {{placeholder}}'


# ---------------------------------------------------------------------------
# judge_config_template
# ---------------------------------------------------------------------------

def test_judge_config_template_preserves_placeholders(ldai_client: LDAIClient, context: Context):
    config = ldai_client.judge_config_template(_JUDGE_FLAG_KEY, context)

    assert config.messages is not None
    assert config.messages[0].content == 'Evaluate the following: {{topic}}'


def test_judge_config_template_returns_typed_config(ldai_client: LDAIClient, context: Context):
    from ldai.models import AIJudgeConfig
    config = ldai_client.judge_config_template(_JUDGE_FLAG_KEY, context)

    assert isinstance(config, AIJudgeConfig)
    assert config.enabled is True
    assert config.model is not None
    assert config.model.name == 'gpt-4'
    assert config.evaluation_metric_key == '$ld:ai:judge:relevance'


def test_judge_config_template_preserves_reserved_placeholders(context: Context):
    """Reserved judge placeholders must survive even in the template variant."""
    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
        'model': {'name': 'gpt-4'},
        'messages': [
            {'role': 'system', 'content': 'History: {{message_history}} Response: {{response_to_evaluate}}'},
        ],
        'evaluationMetricKey': '$ld:ai:judge:relevance',
    }

    ldai = LDAIClient(mock_client)
    config = ldai.judge_config_template('judge-flag', context)

    assert config.messages is not None
    assert '{{message_history}}' in config.messages[0].content
    assert '{{response_to_evaluate}}' in config.messages[0].content


def test_judge_config_template_tracking(context: Context):
    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
        'model': {'name': 'gpt-4'},
        'messages': [{'role': 'system', 'content': 'You are a judge.'}],
        'evaluationMetricKey': '$ld:ai:judge:relevance',
    }

    ldai = LDAIClient(mock_client)
    ldai.judge_config_template('my-judge-flag', context)

    mock_client.track.assert_any_call(
        '$ld:ai:usage:judge-config-template',
        context,
        'my-judge-flag',
        1,
    )


def test_judge_config_template_uses_default_when_flag_missing(ldai_client: LDAIClient, context: Context):
    default = AIJudgeConfigDefault(
        enabled=True,
        model=ModelConfig('default-model'),
        messages=[LDMessage(role='system', content='Judge: {{placeholder}}')],
        evaluation_metric_key='$ld:ai:judge:test',
    )
    config = ldai_client.judge_config_template('missing-flag', context, default)

    assert config.messages is not None
    assert config.messages[0].content == 'Judge: {{placeholder}}'
