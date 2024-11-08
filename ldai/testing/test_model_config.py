import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData
from ldclient.testing.builders import *

from ldai.client import AIConfig, AIConfigData, LDAIClient, LDMessage
from ldai.tracker import LDAIConfigTracker


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('model-config')
        .variations(
            {
                'model': {'modelId': 'fakeModel'},
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
                'model': {'modelId': 'fakeModel'},
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
                'model': {'modelId': 'fakeModel'},
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
                'model': {'modelId': 'fakeModel'},
                'prompt': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': False, 'versionKey': 'abcd'},
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
def tracker(client: LDClient) -> LDAIConfigTracker:
    return LDAIConfigTracker(client, 'abcd', 'model-config', Context.create('user-key'))


@pytest.fixture
def ldai_client(client: LDClient) -> LDAIClient:
    return LDAIClient(client)


def test_model_config_interpolation(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(
        config=AIConfigData(
            model={'modelId': 'fakeModel'},
            prompt=[LDMessage(role='system', content='Hello, {{name}}!')],
        ),
        tracker=tracker,
        enabled=True,
    )
    variables = {'name': 'World'}

    config = ldai_client.model_config('model-config', context, default_value, variables)

    assert config.config.prompt is not None
    assert len(config.config.prompt) > 0
    assert config.config.prompt[0].content == 'Hello, World!'
    assert config.enabled is True


def test_model_config_no_variables(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(
        config=AIConfigData(model={}, prompt=[]), tracker=tracker, enabled=True
    )

    config = ldai_client.model_config('model-config', context, default_value, {})

    assert config.config.prompt is not None
    assert len(config.config.prompt) > 0
    assert config.config.prompt[0].content == 'Hello, !'
    assert config.enabled is True


def test_context_interpolation(ldai_client: LDAIClient, tracker):
    context = Context.builder('user-key').name("Sandy").build()
    default_value = AIConfig(
        config=AIConfigData(model={}, prompt=[]), tracker=tracker, enabled=True
    )
    variables = {'name': 'World'}

    config = ldai_client.model_config(
        'ctx-interpolation', context, default_value, variables
    )

    assert config.config.prompt is not None
    assert len(config.config.prompt) > 0
    assert config.config.prompt[0].content == 'Hello, Sandy!'
    assert config.enabled is True


def test_model_config_multiple(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(
        config=AIConfigData(model={}, prompt=[]), tracker=tracker, enabled=True
    )
    variables = {'name': 'World', 'day': 'Monday'}

    config = ldai_client.model_config(
        'multiple-prompt', context, default_value, variables
    )

    assert config.config.prompt is not None
    assert len(config.config.prompt) > 0
    assert config.config.prompt[0].content == 'Hello, World!'
    assert config.config.prompt[1].content == 'The day is, Monday!'
    assert config.enabled is True


def test_model_config_disabled(ldai_client: LDAIClient, tracker):
    context = Context.create('user-key')
    default_value = AIConfig(
        config=AIConfigData(model={}, prompt=[]), tracker=tracker, enabled=False
    )

    config = ldai_client.model_config('off-config', context, default_value, {})

    assert config.enabled is False
