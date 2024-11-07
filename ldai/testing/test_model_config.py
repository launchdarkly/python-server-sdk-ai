import pytest
from ldclient import LDClient, Context, Config
from ldclient.integrations.test_data import TestData
from ldai.types import AIConfig
from ldai.client import LDAIClient
from ldclient.testing.builders import *

@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(td.flag('model-config').variations({
        'model': { 'modelId': 'fakeModel'},
        'prompt': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
        '_ldMeta': {'enabled': True, 'versionKey': 'abcd'}
    }, "green").variation_for_all(0))

    td.update(td.flag('multiple-prompt').variations({
        'model': { 'modelId': 'fakeModel'},
        'prompt': [{'role': 'system', 'content': 'Hello, {{name}}!'}, {'role': 'user', 'content': 'The day is, {{day}}!'}],
        '_ldMeta': {'enabled': True, 'versionKey': 'abcd'}
    }, "green").variation_for_all(0))

    td.update(td.flag('ctx-interpolation').variations({
        'model': { 'modelId': 'fakeModel'},
        'prompt': [{'role': 'system', 'content': 'Hello, {{ldctx.name}}!'}],
        '_ldMeta': {'enabled': True, 'versionKey': 'abcd'}
    }).variation_for_all(0))

    td.update(td.flag('off-config').variations({
        'model': { 'modelId': 'fakeModel'},
        'prompt': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
        '_ldMeta': {'enabled': False, 'versionKey': 'abcd'}
    }).variation_for_all(0))

    return td

@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config('sdk-key', update_processor_class=td, send_events=False)
    return LDClient(config=config)

@pytest.fixture
def ldai_client(client: LDClient) -> LDAIClient:
    return LDAIClient(client)

def test_model_config_interpolation(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default_value = AIConfig(config={
        'model': { 'modelId': 'fakeModel'},
        'prompt': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
        '_ldMeta': {'enabled': True, 'versionKey': 'abcd'}
    }, tracker=None, enabled=True)
    variables = {'name': 'World'}

    config = ldai_client.model_config('model-config', context, default_value, variables)

    assert config.config['prompt'][0]['content'] == 'Hello, World!'
    assert config.enabled is True
    assert config.tracker.version_key == 'abcd'

def test_model_config_no_variables(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default_value = AIConfig(config={}, tracker=None, enabled=True)

    config = ldai_client.model_config('model-config', context, default_value, {})

    assert config.config['prompt'][0]['content'] == 'Hello, !'
    assert config.enabled is True
    assert config.tracker.version_key == 'abcd'

def test_context_interpolation(ldai_client: LDAIClient):
    context = Context.builder('user-key').name("Sandy").build()
    default_value = AIConfig(config={}, tracker=None, enabled=True)
    variables = {'name': 'World'}

    config = ldai_client.model_config('ctx-interpolation', context, default_value, variables)

    assert config.config['prompt'][0]['content'] == 'Hello, Sandy!'
    assert config.enabled is True
    assert config.tracker.version_key == 'abcd'
    
def test_model_config_disabled(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default_value = AIConfig(config={}, tracker=None, enabled=True)

    config = ldai_client.model_config('off-config', context, default_value, {})

    assert config.enabled is False
    assert config.tracker.version_key == 'abcd'

def test_model_config_multiple(ldai_client: LDAIClient):
    context = Context.create('user-key')
    default_value = AIConfig(config={}, tracker=None, enabled=True)
    variables = {'name': 'World', 'day': 'Monday'}

    config = ldai_client.model_config('multiple-prompt', context, default_value, variables)

    assert config.config['prompt'][0]['content'] == 'Hello, World!'
    assert config.config['prompt'][1]['content'] == 'The day is, Monday!'
    assert config.enabled is True
    assert config.tracker.version_key == 'abcd'