import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, LDMessage, ModelConfig
from ldai.models import (
    AIAgentConfigDefault,
    AICompletionConfigDefault,
    AIJudgeConfigDefault,
)


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('model-config')
        .variations(
            {
                'model': {
                    'name': 'fakeModel',
                    'parameters': {'temperature': 0.5, 'maxTokens': 4096},
                    'custom': {'extra-attribute': 'value'},
                },
                'provider': {'name': 'fakeProvider'},
                'messages': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'abcd', 'version': 1},
            },
            "green",
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('model-config-with-region')
        .variations(
            {
                'model': {
                    'name': 'anthropic.claude-opus-4-7',
                    'parameters': {},
                    'region': 'us',
                },
                'provider': {'name': 'Bedrock'},
                'messages': [{'role': 'system', 'content': 'Hello!'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'us-variation', 'version': 1},
            },
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
                'model': {'name': 'fakeModel', 'parameters': {
                    'extra-attribute': 'I can be anything I set my mind/type to',
                }},
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
                'model': {'name': 'fakeModel', 'parameters': {
                    'extra-attribute': 'I can be anything I set my mind/type to',
                }},
                'messages': [{'role': 'system', 'content': (
                    'Hello, {{ldctx.user.name}}! Do you work for {{ldctx.org.shortname}}?'
                )}],
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
        td.flag('model-config-with-key-version')
        .variations(
            {
                'model': {
                    'name': 'gpt-4',
                },
                'provider': {'name': 'openai'},
                'messages': [],
                '_ldMeta': {
                    'enabled': True, 'variationKey': 'v1', 'version': 1,
                    'modelKey': 'my-model', 'modelVersion': 2,
                },
            },
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


def test_model_config_to_dict_omits_model_key_and_version_when_unset():
    model = ModelConfig('fakeModel', parameters={'temperature': 0.5})
    assert model.model_key is None
    assert model.model_version is None
    result = model.to_dict()
    assert 'modelKey' not in result
    assert 'modelVersion' not in result


def test_model_config_to_dict_includes_model_key_and_version_when_set():
    model = ModelConfig(
        'fakeModel',
        model_key='my-model',
        model_version=2,
    )
    result = model.to_dict()
    assert result['modelKey'] == 'my-model'
    assert result['modelVersion'] == 2


def test_model_config_to_dict_omits_empty_model_key():
    model = ModelConfig('fakeModel', model_key='')
    result = model.to_dict()
    assert 'modelKey' not in result


def test_uses_default_on_invalid_flag(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fakeModel', parameters={'temperature': 0.5, 'maxTokens': 4096}),
        messages=[LDMessage(role='system', content='Hello, {{name}}!')],
    )
    variables = {'name': 'World'}

    config = ldai_client.completion_config('missing-flag', context, default, variables)

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

    config = ldai_client.completion_config('model-config', context, default, variables)

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

    config = ldai_client.completion_config('model-config', context, default, {})

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

    config = ldai_client.completion_config('model-config', context, default, variables)

    assert config.provider is not None
    assert config.provider.name == 'fakeProvider'


def test_context_interpolation(ldai_client: LDAIClient):
    context = Context.builder('user-key').name("Sandy").set('last', 'Beaches').build()
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World'}

    config = ldai_client.completion_config(
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

    config = ldai_client.completion_config(
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

    config = ldai_client.completion_config(
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

    config = ldai_client.completion_config('off-config', context, default, {})

    assert config.model is not None
    assert config.enabled is False
    assert config.model.name == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.1
    assert config.model.get_parameter('maxTokens') is None


def test_model_initial_config_disabled(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.completion_config('initial-config-disabled', context, default, {})

    assert config.enabled is False
    assert config.model is None
    assert config.messages is None
    assert config.provider is None


def test_model_initial_config_enabled(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.completion_config('initial-config-enabled', context, default, {})

    assert config.enabled is True
    assert config.model is None
    assert config.messages is None
    assert config.provider is None


def test_completion_config_method_tracking(ldai_client: LDAIClient):
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

    client.completion_config('test-config-key', context, default)

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
# Optional default value tests
# ============================================================================

def test_completion_config_without_default_uses_disabled(ldai_client: LDAIClient):
    context = Context.create('user-key')

    config = ldai_client.completion_config('missing-flag', context)

    assert config.enabled is False


# ============================================================================
# create_tracker factory tests
# ============================================================================

def test_enabled_config_has_create_tracker(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fakeModel'),
        messages=[LDMessage(role='system', content='Hello!')],
    )

    config = ldai_client.completion_config('model-config', context, default)

    assert config.enabled is True
    assert config.create_tracker is not None
    assert callable(config.create_tracker)


def test_disabled_config_has_working_create_tracker(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.completion_config('off-config', context, default)

    assert config.enabled is False
    assert callable(config.create_tracker)
    tracker = config.create_tracker()
    assert tracker is not None


def test_create_tracker_returns_new_tracker_each_call(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fakeModel'),
        messages=[LDMessage(role='system', content='Hello!')],
    )

    config = ldai_client.completion_config('model-config', context, default)

    assert config.create_tracker is not None
    tracker1 = config.create_tracker()
    tracker2 = config.create_tracker()

    assert tracker1 is not tracker2


def test_create_tracker_produces_fresh_run_id_each_call(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fakeModel'),
        messages=[LDMessage(role='system', content='Hello!')],
    )

    config = ldai_client.completion_config('model-config', context, default)

    assert config.create_tracker is not None
    tracker1 = config.create_tracker()
    tracker2 = config.create_tracker()

    # Each tracker should have a unique runId
    tracker1.track_success()
    tracker2.track_success()


def test_create_tracker_preserves_config_metadata():
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {'enabled': True, 'variationKey': 'var-abc', 'version': 7},
        'model': {'name': 'gpt-4'},
        'provider': {'name': 'openai'},
        'messages': []
    }

    client = LDAIClient(mock_client)
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fake'), messages=[])

    config = client.completion_config('my-config-key', context, default)

    assert config.create_tracker is not None
    tracker = config.create_tracker()
    tracker.track_success()

    # Find the track_success call (skip the sdk:info and usage calls)
    success_calls = [
        c for c in mock_client.track.call_args_list
        if c.args[0] == '$ld:ai:generation:success'
    ]
    assert len(success_calls) == 1
    track_data = success_calls[0].args[2]
    assert track_data['configKey'] == 'my-config-key'
    assert track_data['variationKey'] == 'var-abc'
    assert track_data['version'] == 7
    assert track_data['modelName'] == 'gpt-4'
    assert track_data['providerName'] == 'openai'
    assert 'runId' in track_data


def test_model_config_region():
    model = ModelConfig('fakeModel', region='us')
    assert model.region == 'us'


def test_model_config_region_defaults_to_none():
    model = ModelConfig('fakeModel')
    assert model.region is None


def test_model_config_region_from_flag(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.completion_config('model-config-with-region', context, default)

    assert config.model is not None
    assert config.model.region == 'us'


def test_model_config_no_region_is_none(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=True, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.completion_config('model-config', context, default)

    assert config.model is not None
    assert config.model.region is None


def test_create_tracker_each_call_has_different_run_id():
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
        'model': {'name': 'test-model'},
        'provider': {'name': 'test-provider'},
        'messages': []
    }

    client = LDAIClient(mock_client)
    context = Context.create('user-key')

    config = client.completion_config('key', context)

    assert config.create_tracker is not None
    tracker1 = config.create_tracker()
    tracker2 = config.create_tracker()

    tracker1.track_success()
    tracker2.track_success()

    success_calls = [
        c for c in mock_client.track.call_args_list
        if c.args[0] == '$ld:ai:generation:success'
    ]
    assert len(success_calls) == 2
    run_id_1 = success_calls[0].args[2]['runId']
    run_id_2 = success_calls[1].args[2]['runId']
    assert run_id_1 != run_id_2


def test_model_config_reads_model_key_and_version_from_flag(ldai_client: LDAIClient):
    context = Context.create('user-key')
    result = ldai_client.completion_config('model-config-with-key-version', context)

    assert result.model is not None
    assert result.model.model_key == 'my-model'
    assert result.model.model_version == 2


def test_create_tracker_stamps_model_key_and_version_on_track_data():
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {
            'enabled': True, 'variationKey': 'var-abc', 'version': 7,
            'modelKey': 'my-model', 'modelVersion': 2,
        },
        'model': {
            'name': 'gpt-4',
        },
        'provider': {'name': 'openai'},
        'messages': []
    }

    client = LDAIClient(mock_client)
    context = Context.create('user-key')

    config = client.completion_config('my-config-key', context)
    tracker = config.create_tracker()
    tracker.track_success()

    success_calls = [
        c for c in mock_client.track.call_args_list
        if c.args[0] == '$ld:ai:generation:success'
    ]
    assert len(success_calls) == 1
    track_data = success_calls[0].args[2]
    assert track_data['modelKey'] == 'my-model'
    assert track_data['modelVersion'] == 2


@pytest.mark.parametrize(
    'ld_meta_overrides,expected_model_key,expected_model_version',
    [
        pytest.param({}, None, None, id='omits_model_version_when_absent'),
        pytest.param({'modelVersion': 3}, None, 3, id='omits_model_key_when_absent'),
    ],
)
def test_create_tracker_model_key_and_version_defaults(
    ld_meta_overrides, expected_model_key, expected_model_version,
):
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.variation.return_value = {
        '_ldMeta': {
            'enabled': True, 'variationKey': 'var-abc', 'version': 7,
            **ld_meta_overrides,
        },
        'model': {'name': 'gpt-4'},
        'provider': {'name': 'openai'},
        'messages': []
    }

    client = LDAIClient(mock_client)
    context = Context.create('user-key')

    config = client.completion_config('my-config-key', context)
    tracker = config.create_tracker()
    tracker.track_success()

    success_calls = [
        c for c in mock_client.track.call_args_list
        if c.args[0] == '$ld:ai:generation:success'
    ]
    assert len(success_calls) == 1
    track_data = success_calls[0].args[2]
    if expected_model_key is None:
        assert 'modelKey' not in track_data
    else:
        assert track_data['modelKey'] == expected_model_key
    if expected_model_version is None:
        assert 'modelVersion' not in track_data
    else:
        assert track_data['modelVersion'] == expected_model_version
