import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, LDMessage, ModelConfig, ToolDefinition
from ldai.models import (AIAgentConfigDefault, AICompletionConfigDefault,
                         AIConfigDefault, AIJudgeConfigDefault)


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('model-config')
        .variations(
            {
                'model': {'name': 'fakeModel', 'parameters': {'temperature': 0.5, 'maxTokens': 4096}, 'custom': {'extra-attribute': 'value'}},
                'provider': {'name': 'fakeProvider'},
                'messages': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'abcd', 'version': 1},
            },
            "green",
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('multiple-messages')
        .variations(
            {
                'model': {'name': 'fakeModel', 'parameters': {'temperature': 0.7, 'maxTokens': 8192}},
                'messages': [
                    {'role': 'system', 'content': 'Hello, {{name}}!'},
                    {'role': 'user', 'content': 'The day is, {{day}}!'},
                ],
                '_ldMeta': {'enabled': True, 'variationKey': 'abcd', 'version': 1},
            },
            "green",
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('ctx-interpolation')
        .variations(
            {
                'model': {'name': 'fakeModel', 'parameters': {'extra-attribute': 'I can be anything I set my mind/type to'}},
                'messages': [{'role': 'system', 'content': 'Hello, {{ldctx.name}}! Is your last name {{ldctx.last}}?'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'abcd', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('multi-ctx-interpolation')
        .variations(
            {
                'model': {'name': 'fakeModel', 'parameters': {'extra-attribute': 'I can be anything I set my mind/type to'}},
                'messages': [{'role': 'system', 'content': 'Hello, {{ldctx.user.name}}! Do you work for {{ldctx.org.shortname}}?'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'abcd', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('off-config')
        .variations(
            {
                'model': {'name': 'fakeModel', 'parameters': {'temperature': 0.1}},
                'messages': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': False, 'variationKey': 'abcd', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('initial-config-disabled')
        .variations(
            {
                '_ldMeta': {'enabled': False},
            },
            {
                '_ldMeta': {'enabled': True},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('initial-config-enabled')
        .variations(
            {
                '_ldMeta': {'enabled': False},
            },
            {
                '_ldMeta': {'enabled': True},
            }
        )
        .variation_for_all(1)
    )

    td.update(
        td.flag('config-with-tools')
        .variations(
            {
                'model': {
                    'name': 'gpt-4',
                    'parameters': {
                        'temperature': 0.7,
                        'tools': [
                            {'name': 'web_search', 'customParameters': {'maxResults': 10, 'region': 'us'}},
                            {'name': 'get_weather', 'customParameters': {'units': 'celsius'}},
                            {'name': 'calculator'},
                        ],
                    },
                },
                'provider': {'name': 'openai'},
                'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'tools-v1', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('config-no-tools')
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.5}},
                'provider': {'name': 'openai'},
                'messages': [{'role': 'system', 'content': 'Hello'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'no-tools-v1', 'version': 1},
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


def test_model_config_delegates_to_properties():
    model = ModelConfig('fakeModel', parameters={'extra-attribute': 'value'})
    assert model.name == 'fakeModel'
    assert model.get_parameter('extra-attribute') == 'value'
    assert model.get_parameter('non-existent') is None

    assert model.name == model.get_parameter('name')


def test_model_config_handles_custom():
    model = ModelConfig('fakeModel', custom={'extra-attribute': 'value'})
    assert model.name == 'fakeModel'
    assert model.get_parameter('extra-attribute') is None
    assert model.get_custom('non-existent') is None
    assert model.get_custom('name') is None


def test_uses_default_on_invalid_flag(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fakeModel', parameters={'temperature': 0.5, 'maxTokens': 4096}),
        messages=[LDMessage(role='system', content='Hello, {{name}}!')],
    )
    variables = {'name': 'World'}

    config = ldai_client.config('missing-flag', context, default, variables)

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, World!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.5
    assert config.model.get_parameter('maxTokens') == 4096


def test_model_config_interpolation(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fakeModel'),
        messages=[LDMessage(role='system', content='Hello, {{name}}!')],
    )
    variables = {'name': 'World'}

    config = ldai_client.config('model-config', context, default, variables)

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, World!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.5
    assert config.model.get_parameter('maxTokens') == 4096


def test_model_config_no_variables(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('model-config', context, default, {})

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, !'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.5
    assert config.model.get_parameter('maxTokens') == 4096


def test_provider_config_handling(ldai_client: LDAIClient):
    context = Context.builder('user-key').name("Sandy").build()
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World'}

    config = ldai_client.config('model-config', context, default, variables)

    assert config.provider is not None
    assert config.provider.name == 'fakeProvider'


def test_context_interpolation(ldai_client: LDAIClient):
    context = Context.builder('user-key').name("Sandy").set('last', 'Beaches').build()
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World'}

    config = ldai_client.config(
        'ctx-interpolation', context, default, variables
    )

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, Sandy! Is your last name Beaches?'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') is None
    assert config.model.get_parameter('maxTokens') is None
    assert config.model.get_parameter('extra-attribute') == 'I can be anything I set my mind/type to'


def test_multi_context_interpolation(ldai_client: LDAIClient):
    user_context = Context.builder('user-key').name("Sandy").build()
    org_context = Context.builder('org-key').kind('org').name("LaunchDarkly").set('shortname', 'LD').build()
    context = Context.multi_builder().add(user_context).add(org_context).build()
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World'}

    config = ldai_client.config(
        'multi-ctx-interpolation', context, default, variables
    )

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, Sandy! Do you work for LD?'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') is None
    assert config.model.get_parameter('maxTokens') is None
    assert config.model.get_parameter('extra-attribute') == 'I can be anything I set my mind/type to'


def test_model_config_multiple(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World', 'day': 'Monday'}

    config = ldai_client.config(
        'multiple-messages', context, default, variables
    )

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, World!'
    assert config.messages[1].content == 'The day is, Monday!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.7
    assert config.model.get_parameter('maxTokens') == 8192


def test_model_config_disabled(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('off-config', context, default, {})

    assert config.model is not None
    assert config.enabled is False
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.1
    assert config.model.get_parameter('maxTokens') is None


def test_model_initial_config_disabled(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('initial-config-disabled', context, default, {})

    assert config.enabled is False
    assert config.model is None
    assert config.messages is None
    assert config.provider is None


def test_model_initial_config_enabled(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('initial-config-enabled', context, default, {})

    assert config.enabled is True
    assert config.model is None
    assert config.messages is None
    assert config.provider is None


def test_config_method_tracking(ldai_client: LDAIClient):
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {'enabled': True, 'variationKey': 'test-variation', 'version': 1},
        'model': {'name': 'test-model'},
        'provider': {'name': 'test-provider'},
        'messages': []
    }

    client = LDAIClient(mock_client)
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = client.config('test-config-key', context, default)

    mock_client.track.assert_any_call(
        '$ld:ai:usage:completion-config',
        context,
        'test-config-key',
        1
    )


def test_sdk_info_tracked_on_init():
    from unittest.mock import Mock

    from ldai.client import _INIT_TRACK_CONTEXT
    from ldai.sdk_info import AI_SDK_LANGUAGE, AI_SDK_NAME, AI_SDK_VERSION

    mock_client = Mock()

    client = LDAIClient(mock_client)

    mock_client.track.assert_called_once_with(
        '$ld:ai:sdk:info',
        _INIT_TRACK_CONTEXT,
        {
            'aiSdkName': AI_SDK_NAME,
            'aiSdkVersion': AI_SDK_VERSION,
            'aiSdkLanguage': AI_SDK_LANGUAGE,
        },
        1,
    )


# ============================================================================
# disabled() classmethod tests
# ============================================================================

def test_ai_config_default_disabled_returns_disabled_instance():
    result = AIConfigDefault.disabled()
    assert isinstance(result, AIConfigDefault)
    assert result.enabled is False


def test_completion_config_default_disabled_returns_correct_type():
    result = AICompletionConfigDefault.disabled()
    assert isinstance(result, AICompletionConfigDefault)
    assert result.enabled is False
    assert result.messages is None
    assert result.model is None


def test_agent_config_default_disabled_returns_correct_type():
    result = AIAgentConfigDefault.disabled()
    assert isinstance(result, AIAgentConfigDefault)
    assert result.enabled is False
    assert result.instructions is None
    assert result.model is None


def test_judge_config_default_disabled_returns_correct_type():
    result = AIJudgeConfigDefault.disabled()
    assert isinstance(result, AIJudgeConfigDefault)
    assert result.enabled is False
    assert result.messages is None
    assert result.evaluation_metric_key is None


def test_disabled_returns_new_instance_each_call():
    first = AICompletionConfigDefault.disabled()
    second = AICompletionConfigDefault.disabled()
    assert first is not second


# ============================================================================
# Optional default value tests
# ============================================================================

def test_completion_config_without_default_uses_disabled(ldai_client: LDAIClient):
    context = Context.create('user-key')

    config = ldai_client.completion_config('missing-flag', context)

    assert config.enabled is False


# ============================================================================
# ToolDefinition tests
# ============================================================================

def test_tool_definition_basic():
    tool = ToolDefinition('web_search')
    assert tool.name == 'web_search'
    assert tool.get_custom_parameter('anything') is None


def test_tool_definition_with_custom_parameters():
    tool = ToolDefinition('web_search', custom_parameters={'maxResults': 10, 'region': 'us'})
    assert tool.name == 'web_search'
    assert tool.get_custom_parameter('maxResults') == 10
    assert tool.get_custom_parameter('region') == 'us'
    assert tool.get_custom_parameter('nonexistent') is None


def test_tool_definition_to_dict():
    tool = ToolDefinition('web_search', custom_parameters={'maxResults': 10})
    d = tool.to_dict()
    assert d == {'name': 'web_search', 'customParameters': {'maxResults': 10}}


def test_tool_definition_to_dict_no_custom_parameters():
    tool = ToolDefinition('calculator')
    d = tool.to_dict()
    assert d == {'name': 'calculator'}


def test_completion_config_has_tools(ldai_client: LDAIClient):
    """Test that tools with custom parameters are parsed from flag variations."""
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fallback'), messages=[])

    config = ldai_client.completion_config('config-with-tools', context, default)

    assert config.tools is not None
    assert len(config.tools) == 3

    web_search = config.tools[0]
    assert web_search.name == 'web_search'
    assert web_search.get_custom_parameter('maxResults') == 10
    assert web_search.get_custom_parameter('region') == 'us'

    get_weather = config.tools[1]
    assert get_weather.name == 'get_weather'
    assert get_weather.get_custom_parameter('units') == 'celsius'

    calculator = config.tools[2]
    assert calculator.name == 'calculator'
    assert calculator.get_custom_parameter('anything') is None


def test_completion_config_no_tools(ldai_client: LDAIClient):
    """Test that tools is None when no tools are defined."""
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fallback'), messages=[])

    config = ldai_client.completion_config('config-no-tools', context, default)

    assert config.tools is None


def test_completion_config_tools_missing_flag(ldai_client: LDAIClient):
    """Test that tools from default are not used for completion configs."""
    context = Context.create('user-key')
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fallback'),
        messages=[],
        tools=[ToolDefinition('default_tool', custom_parameters={'key': 'value'})],
    )

    config = ldai_client.completion_config('missing-flag', context, default)

    # The default is serialized into the variation dict, so the SDK evaluates
    # against it; completion_config does not fall back to default.tools
    # separately — the variation itself carries the tool definitions.
    assert config.tools is not None
    assert len(config.tools) == 1
    assert config.tools[0].name == 'default_tool'
    assert config.tools[0].get_custom_parameter('key') == 'value'
