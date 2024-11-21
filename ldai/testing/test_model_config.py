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
                'model': {'modelId': 'fakeModel', 'temperature': 0.5, 'maxTokens': 4096},
                'prompt': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': True, 'versionKey': 'abcd'},
            },
            "green",
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('multiple-prompt')
        .variations(
            {
                'model': {'modelId': 'fakeModel', 'temperature': 0.7, 'maxTokens': 8192},
                'prompt': [
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
                'model': {'modelId': 'fakeModel', 'extra-attribute': 'I can be anything I set my mind/type to'},
                'prompt': [{'role': 'system', 'content': 'Hello, {{ldctx.name}}!'}],
                '_ldMeta': {'enabled': True, 'versionKey': 'abcd'},
            }
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('off-config')
        .variations(
            {
                'model': {'modelId': 'fakeModel', 'temperature': 0.1},
                'prompt': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
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
    model = ModelConfig('fakeModel', temperature=0.5, max_tokens=4096, attributes={'extra-attribute': 'value'})
    assert model.id == 'fakeModel'
    assert model.temperature == 0.5
    assert model.max_tokens == 4096
    assert model.get_attribute('extra-attribute') == 'value'
    assert model.get_attribute('non-existent') is None

    assert model.id == model.get_attribute('id')
    assert model.temperature == model.get_attribute('temperature')
    assert model.max_tokens == model.get_attribute('maxTokens')
    assert model.max_tokens != model.get_attribute('max_tokens')


def test_model_config_interpolation(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(
        tracker=tracker,
        enabled=True,
        model=ModelConfig('fakeModel'),
        prompt=[LDMessage(role='system', content='Hello, {{name}}!')],
    )
    variables = {'name': 'World'}

    config = ldai_client.model_config('model-config', context, default_value, variables)

    assert config.prompt is not None
    assert len(config.prompt) > 0
    assert config.prompt[0].content == 'Hello, World!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.temperature == 0.5
    assert config.model.max_tokens == 4096


def test_model_config_no_variables(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=True, model=ModelConfig('fake-model'), prompt=[])

    config = ldai_client.model_config('model-config', context, default_value, {})

    assert config.prompt is not None
    assert len(config.prompt) > 0
    assert config.prompt[0].content == 'Hello, !'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.temperature == 0.5
    assert config.model.max_tokens == 4096


def test_context_interpolation(ldai_client: LDAIClient, tracker):
    context = Context.builder('user-key').name("Sandy").build()
    default_value = AIConfig(tracker=tracker, enabled=True, model=ModelConfig('fake-model'), prompt=[])
    variables = {'name': 'World'}

    config = ldai_client.model_config(
        'ctx-interpolation', context, default_value, variables
    )

    assert config.prompt is not None
    assert len(config.prompt) > 0
    assert config.prompt[0].content == 'Hello, Sandy!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.temperature is None
    assert config.model.max_tokens is None
    assert config.model.get_attribute('extra-attribute') == 'I can be anything I set my mind/type to'


def test_model_config_multiple(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=True, model=ModelConfig('fake-model'), prompt=[])
    variables = {'name': 'World', 'day': 'Monday'}

    config = ldai_client.model_config(
        'multiple-prompt', context, default_value, variables
    )

    assert config.prompt is not None
    assert len(config.prompt) > 0
    assert config.prompt[0].content == 'Hello, World!'
    assert config.prompt[1].content == 'The day is, Monday!'
    assert config.enabled is True

    assert config.model is not None
    assert config.model.id == 'fakeModel'
    assert config.model.temperature == 0.7
    assert config.model.max_tokens == 8192


def test_model_config_disabled(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=False, model=ModelConfig('fake-model'), prompt=[])

    config = ldai_client.model_config('off-config', context, default_value, {})

    assert config.model is not None
    assert config.enabled is False
    assert config.model.id == 'fakeModel'
    assert config.model.temperature == 0.1
    assert config.model.max_tokens is None


def test_model_initial_config_disabled(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=False, model=ModelConfig('fake-model'), prompt=[])

    config = ldai_client.model_config('initial-config-disabled', context, default_value, {})

    assert config.enabled is False
    assert config.model is None
    assert config.prompt is None


def test_model_initial_config_enabled(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(tracker=tracker, enabled=False, model=ModelConfig('fake-model'), prompt=[])

    config = ldai_client.model_config('initial-config-enabled', context, default_value, {})

    assert config.enabled is True
    assert config.model is None
    assert config.prompt is None
