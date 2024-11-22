import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai.client import AIConfig, LDAIClient, LDMessage, ModelConfig
from ldai.tracker import LDAIConfigTracker


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('model-config')
        .variations(
            {
                'model': {'id': 'fakeModel', 'parameters': {'temperature': 0.5, 'maxTokens': 4096}, 'custom': {'extra-attribute': 'value'}},
                'provider': {'id': 'fakeProvider'},
                'messages': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': True, 'versionKey': 'abcd'},
            },
            "green",
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('multiple-messages')
        .variations(
            {
                'model': {'id': 'fakeModel', 'parameters': {'temperature': 0.7, 'maxTokens': 8192}},
                'messages': [
                    {'role': 'system', 'content': 'Hello, {{name}}!'},
                    {'role': 'user', 'content': 'The day is, {{day}}!'},
                ],
                '_ldMeta': {'enabled': True, 'versionKey': 'abcd'},
            },
            "green",
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('ctx-interpolation')
        .variations(
            {
                'model': {'id': 'fakeModel', 'parameters': {'extra-attribute': 'I can be anything I set my mind/type to'}},
                'messages': [{'role': 'system', 'content': 'Hello, {{ldctx.name}}!'}],
                '_ldMeta': {'enabled': True, 'versionKey': 'abcd'},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('off-config')
        .variations(
            {
                'model': {'id': 'fakeModel', 'parameters': {'temperature': 0.1}},
                'messages': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': False, 'versionKey': 'abcd'},
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

    return td


@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config('sdk-key', update_processor_class=td, send_events=False)
    return LDClient(config=config)


@pytest.fixture
def tracker(client: LDClient) -> LDAIConfigTracker:
    return LDAIConfigTracker(client, 'abcd', 'model-config', Context.create('user-key'))


@pytest.fixture
def ldai_client(client: LDClient) -> LDAIClient:
    return LDAIClient(client)


def test_model_config_delegates_to_properties():
    model = ModelConfig('fakeModel', parameters={'extra-attribute': 'value'})
    assert model.id == 'fakeModel'
    assert model.get_parameter('extra-attribute') == 'value'
    assert model.get_parameter('non-existent') is None

    assert model.id == model.get_parameter('id')


def test_model_config_handles_custom():
    model = ModelConfig('fakeModel', custom={'extra-attribute': 'value'})
    assert model.id == 'fakeModel'
    assert model.get_parameter('extra-attribute') is None
    assert model.get_custom('non-existent') is None
    assert model.get_custom('id') is None


def test_model_config_interpolation(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(
        tracker=tracker,
        enabled=True,
        model=ModelConfig('fakeModel'),
        messages=[LDMessage(role='system', content='Hello, {{name}}!')],
    )
    variables = {'name': 'World'}

    config = ldai_client.config('model-config', context, default_value, variables)

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, World!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.5
    assert config.model.get_parameter('maxTokens') == 4096


def test_model_config_no_variables(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=True, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('model-config', context, default_value, {})

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, !'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.5
    assert config.model.get_parameter('maxTokens') == 4096


def test_provider_config_handling(ldai_client: LDAIClient, tracker):
    context = Context.builder('user-key').name("Sandy").build()
    default_value = AIConfig(tracker=tracker, enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World'}

    config = ldai_client.config('model-config', context, default_value, variables)

    assert config.provider is not None
    assert config.provider.id == 'fakeProvider'


def test_context_interpolation(ldai_client: LDAIClient, tracker):
    context = Context.builder('user-key').name("Sandy").build()
    default_value = AIConfig(tracker=tracker, enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World'}

    config = ldai_client.config(
        'ctx-interpolation', context, default_value, variables
    )

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, Sandy!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.get_parameter('temperature') is None
    assert config.model.get_parameter('maxTokens') is None
    assert config.model.get_parameter('extra-attribute') == 'I can be anything I set my mind/type to'


def test_model_config_multiple(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=True, model=ModelConfig('fake-model'), messages=[])
    variables = {'name': 'World', 'day': 'Monday'}

    config = ldai_client.config(
        'multiple-messages', context, default_value, variables
    )

    assert config.messages is not None
    assert len(config.messages) > 0
    assert config.messages[0].content == 'Hello, World!'
    assert config.messages[1].content == 'The day is, Monday!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.7
    assert config.model.get_parameter('maxTokens') == 8192


def test_model_config_disabled(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('off-config', context, default_value, {})

    assert config.model is not None
    assert config.enabled is False
    assert config.model.id == 'fakeModel'
    assert config.model.get_parameter('temperature') == 0.1
    assert config.model.get_parameter('maxTokens') is None


def test_model_initial_config_disabled(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('initial-config-disabled', context, default_value, {})

    assert config.enabled is False
    assert config.model is None
    assert config.messages is None
    assert config.provider is None


def test_model_initial_config_enabled(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=False, model=ModelConfig('fake-model'), messages=[])

    config = ldai_client.config('initial-config-enabled', context, default_value, {})

    assert config.enabled is True
    assert config.model is None
    assert config.messages is None
    assert config.provider is None
