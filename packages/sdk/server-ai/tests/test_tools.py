import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDTool, LDAIClient
from ldai.models import AIAgentConfigDefault, AICompletionConfigDefault


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('completion-with-tools')
        .variations(
            {
                'model': {'name': 'gpt-5', 'parameters': {'temperature': 0.7}},
                'messages': [{'role': 'user', 'content': 'Hello'}],
                'tools': {
                    'web-search-tool': {
                        'name': 'web-search-tool',
                        'type': 'function',
                        'parameters': {'type': 'object', 'properties': {}, 'required': []},
                        'customParameters': {'some-custom-parameter': 'some-custom-value'},
                    }
                },
                '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
            },
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('completion-no-tools')
        .variations(
            {
                'model': {'name': 'gpt-5'},
                'messages': [{'role': 'user', 'content': 'Hello'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
            },
        )
        .variation_for_all(0)
    )

    td.update(
        td.flag('agent-with-tools')
        .variations(
            {
                'model': {'name': 'gpt-5'},
                'instructions': 'You are a helpful agent.',
                'tools': {
                    'search-tool': {
                        'name': 'search-tool',
                        'type': 'function',
                        'customParameters': {'maxResults': 10},
                    }
                },
                '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1, 'mode': 'agent'},
            },
        )
        .variation_for_all(0)
    )

    return td


@pytest.fixture
def client(td) -> LDAIClient:
    config = Config('fake-sdk-key', update_processor_class=td, send_events=False)
    ld_client = LDClient(config=config)
    return LDAIClient(ld_client)


@pytest.fixture
def context() -> Context:
    return Context.builder('test-user').name('Test User').build()


def test_completion_config_includes_tools_from_variation(client, context):
    result = client.completion_config('completion-with-tools', context, AICompletionConfigDefault())

    assert result.tools is not None
    assert 'web-search-tool' in result.tools
    tool = result.tools['web-search-tool']
    assert tool.name == 'web-search-tool'
    assert tool.type == 'function'
    assert tool.custom_parameters == {'some-custom-parameter': 'some-custom-value'}


def test_completion_config_tools_none_when_not_in_variation(client, context):
    result = client.completion_config('completion-no-tools', context, AICompletionConfigDefault())

    assert result.tools is None


def test_completion_config_falls_back_to_default_tools(client, context):
    default_tool = LDTool(name='default-tool', type='function', custom_parameters={'priority': 'high'})
    default = AICompletionConfigDefault(tools={'default-tool': default_tool})

    result = client.completion_config('completion-no-tools', context, default)

    assert result.tools is not None
    assert 'default-tool' in result.tools
    assert result.tools['default-tool'].custom_parameters == {'priority': 'high'}


def test_agent_config_includes_tools_from_variation(client, context):
    result = client.agent_config('agent-with-tools', context, AIAgentConfigDefault())

    assert result.tools is not None
    assert 'search-tool' in result.tools
    tool = result.tools['search-tool']
    assert tool.name == 'search-tool'
    assert tool.custom_parameters == {'maxResults': 10}


def test_aitool_to_dict_serializes_custom_parameters_as_camel_case():
    tool = LDTool(
        name='my-tool',
        type='function',
        parameters={'type': 'object'},
        custom_parameters={'someKey': 'someValue'},
    )
    d = tool.to_dict()

    assert d['name'] == 'my-tool'
    assert d['type'] == 'function'
    assert d['parameters'] == {'type': 'object'}
    assert 'customParameters' in d
    assert d['customParameters'] == {'someKey': 'someValue'}
    assert 'custom_parameters' not in d


def test_aitool_to_dict_omits_none_fields():
    tool = LDTool(name='bare-tool')
    d = tool.to_dict()

    assert d == {'name': 'bare-tool'}
