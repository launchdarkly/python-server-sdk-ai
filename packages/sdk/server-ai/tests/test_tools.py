import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import LDAIClient, ModelConfig
from ldai.models import (AIAgentConfigDefault, AICompletionConfigDefault,
                         AITool)


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()

    # Completion config with tools
    td.update(
        td.flag('completion-with-tools')
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.5}},
                'provider': {'name': 'openai'},
                'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}],
                'tools': [
                    {
                        'key': 'get-customer',
                        'version': 1,
                        'instructions': 'Use this tool to look up customer details by email.',
                        'examples': 'Input: {"email": "jane@example.com"}\nOutput: {"name": "Jane Doe"}',
                        'customParameters': {'endpoint': '/api/customers'},
                    },
                    {
                        'key': 'search-orders',
                        'version': 2,
                        'instructions': 'Search for orders by customer ID.',
                    },
                ],
                '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    # Agent config with tools
    td.update(
        td.flag('agent-with-tools')
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.3}},
                'provider': {'name': 'openai'},
                'instructions': 'You are a customer support agent for {{company_name}}.',
                'tools': [
                    {
                        'key': 'crm-lookup',
                        'version': 1,
                        'instructions': 'Look up customer info in the CRM.',
                        'examples': 'Input: {"id": "123"}\nOutput: {"name": "John", "plan": "Enterprise"}',
                        'customParameters': {'timeout': 30},
                    },
                ],
                '_ldMeta': {'enabled': True, 'variationKey': 'agent-v1', 'version': 1, 'mode': 'agent'},
            }
        )
        .variation_for_all(0)
    )

    # Config with no tools
    td.update(
        td.flag('no-tools-config')
        .variations(
            {
                'model': {'name': 'gpt-4'},
                'messages': [{'role': 'system', 'content': 'Hello'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    # Config with empty tools array
    td.update(
        td.flag('empty-tools-config')
        .variations(
            {
                'model': {'name': 'gpt-4'},
                'messages': [{'role': 'system', 'content': 'Hello'}],
                'tools': [],
                '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1},
            }
        )
        .variation_for_all(0)
    )

    # Agent config with tools that have minimal fields
    td.update(
        td.flag('agent-minimal-tools')
        .variations(
            {
                'instructions': 'Minimal agent.',
                'tools': [
                    {'key': 'basic-tool', 'version': 1},
                ],
                '_ldMeta': {'enabled': True, 'variationKey': 'v1', 'version': 1, 'mode': 'agent'},
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


# ============================================================================
# AITool dataclass tests
# ============================================================================

def test_ai_tool_to_dict_full():
    """Test AITool.to_dict() with all fields populated."""
    tool = AITool(
        key='get-customer',
        version=1,
        instructions='Look up customer by email.',
        examples='Input: {"email": "a@b.com"}\nOutput: {"name": "Alice"}',
        custom_parameters={'endpoint': '/api/customers'},
    )

    result = tool.to_dict()

    assert result == {
        'key': 'get-customer',
        'version': 1,
        'instructions': 'Look up customer by email.',
        'examples': 'Input: {"email": "a@b.com"}\nOutput: {"name": "Alice"}',
        'customParameters': {'endpoint': '/api/customers'},
    }


def test_ai_tool_to_dict_minimal():
    """Test AITool.to_dict() with only required fields."""
    tool = AITool(key='basic-tool', version=1)

    result = tool.to_dict()

    assert result == {
        'key': 'basic-tool',
        'version': 1,
    }
    assert 'instructions' not in result
    assert 'examples' not in result
    assert 'customParameters' not in result


def test_ai_tool_is_frozen():
    """Test that AITool is immutable."""
    tool = AITool(key='test', version=1)
    with pytest.raises(AttributeError):
        tool.key = 'changed'  # type: ignore[misc]


# ============================================================================
# Completion config with tools tests
# ============================================================================

def test_completion_config_with_tools(ldai_client: LDAIClient):
    """Test that completion config correctly parses tools from variation."""
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fallback'))

    config = ldai_client.completion_config('completion-with-tools', context, default)

    assert config.enabled is True
    assert config.tools is not None
    assert len(config.tools) == 2

    tool1 = config.tools[0]
    assert tool1.key == 'get-customer'
    assert tool1.version == 1
    assert tool1.instructions == 'Use this tool to look up customer details by email.'
    assert tool1.examples == 'Input: {"email": "jane@example.com"}\nOutput: {"name": "Jane Doe"}'
    assert tool1.custom_parameters == {'endpoint': '/api/customers'}

    tool2 = config.tools[1]
    assert tool2.key == 'search-orders'
    assert tool2.version == 2
    assert tool2.instructions == 'Search for orders by customer ID.'
    assert tool2.examples is None
    assert tool2.custom_parameters is None


def test_completion_config_without_tools(ldai_client: LDAIClient):
    """Test that completion config has no tools when variation has none."""
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fallback'))

    config = ldai_client.completion_config('no-tools-config', context, default)

    assert config.enabled is True
    assert config.tools is None


def test_completion_config_empty_tools(ldai_client: LDAIClient):
    """Test that completion config handles empty tools array."""
    context = Context.create('user-key')
    default = AICompletionConfigDefault(enabled=False, model=ModelConfig('fallback'))

    config = ldai_client.completion_config('empty-tools-config', context, default)

    assert config.enabled is True
    assert config.tools is None


def test_completion_config_uses_default_tools(ldai_client: LDAIClient):
    """Test that completion config falls back to default tools when variation has none."""
    context = Context.create('user-key')
    default_tools = [AITool(key='default-tool', version=1, instructions='Default tool')]
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('fallback'),
        tools=default_tools,
    )

    config = ldai_client.completion_config('no-tools-config', context, default)

    assert config.tools is not None
    assert len(config.tools) == 1
    assert config.tools[0].key == 'default-tool'


def test_completion_config_default_to_dict_with_tools():
    """Test AICompletionConfigDefault.to_dict() includes tools."""
    tools = [
        AITool(key='tool-a', version=1, instructions='Do A'),
        AITool(key='tool-b', version=2),
    ]
    default = AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig('gpt-4'),
        tools=tools,
    )

    result = default.to_dict()

    assert 'tools' in result
    assert len(result['tools']) == 2
    assert result['tools'][0]['key'] == 'tool-a'
    assert result['tools'][0]['instructions'] == 'Do A'
    assert result['tools'][1]['key'] == 'tool-b'


def test_completion_config_to_dict_with_tools():
    """Test AICompletionConfig.to_dict() includes tools."""
    from ldai.models import AICompletionConfig

    tools = [AITool(key='my-tool', version=3, examples='example text')]
    config = AICompletionConfig(
        key='test',
        enabled=True,
        tools=tools,
    )

    result = config.to_dict()

    assert 'tools' in result
    assert len(result['tools']) == 1
    assert result['tools'][0]['key'] == 'my-tool'
    assert result['tools'][0]['version'] == 3
    assert result['tools'][0]['examples'] == 'example text'


# ============================================================================
# Agent config with tools tests
# ============================================================================

def test_agent_config_with_tools(ldai_client: LDAIClient):
    """Test that agent config correctly parses tools from variation."""
    context = Context.create('user-key')
    default = AIAgentConfigDefault(enabled=False, model=ModelConfig('fallback'))

    agent = ldai_client.agent_config(
        'agent-with-tools', context, default, {'company_name': 'Acme Corp'}
    )

    assert agent.enabled is True
    assert agent.instructions == 'You are a customer support agent for Acme Corp.'
    assert agent.tools is not None
    assert len(agent.tools) == 1

    tool = agent.tools[0]
    assert tool.key == 'crm-lookup'
    assert tool.version == 1
    assert tool.instructions == 'Look up customer info in the CRM.'
    assert tool.examples == 'Input: {"id": "123"}\nOutput: {"name": "John", "plan": "Enterprise"}'
    assert tool.custom_parameters == {'timeout': 30}


def test_agent_config_minimal_tools(ldai_client: LDAIClient):
    """Test agent config with tools that have only required fields."""
    context = Context.create('user-key')
    default = AIAgentConfigDefault(enabled=False)

    agent = ldai_client.agent_config('agent-minimal-tools', context, default)

    assert agent.enabled is True
    assert agent.tools is not None
    assert len(agent.tools) == 1
    assert agent.tools[0].key == 'basic-tool'
    assert agent.tools[0].version == 1
    assert agent.tools[0].instructions is None
    assert agent.tools[0].examples is None
    assert agent.tools[0].custom_parameters is None


def test_agent_config_uses_default_tools(ldai_client: LDAIClient):
    """Test that agent config falls back to default tools when variation has none."""
    context = Context.create('user-key')
    default_tools = [AITool(key='default-agent-tool', version=1)]
    default = AIAgentConfigDefault(
        enabled=True,
        model=ModelConfig('fallback'),
        tools=default_tools,
        instructions='Default instructions',
    )

    agent = ldai_client.agent_config('non-existent-agent', context, default)

    assert agent.tools is not None
    assert len(agent.tools) == 1
    assert agent.tools[0].key == 'default-agent-tool'


def test_agent_config_default_to_dict_with_tools():
    """Test AIAgentConfigDefault.to_dict() includes tools."""
    tools = [AITool(key='agent-tool', version=1, instructions='Do something')]
    default = AIAgentConfigDefault(
        enabled=True,
        instructions='Be helpful.',
        tools=tools,
    )

    result = default.to_dict()

    assert 'tools' in result
    assert len(result['tools']) == 1
    assert result['tools'][0]['key'] == 'agent-tool'


def test_agent_config_to_dict_with_tools():
    """Test AIAgentConfig.to_dict() includes tools."""
    from ldai.models import AIAgentConfig

    tools = [AITool(key='my-tool', version=2)]
    config = AIAgentConfig(
        key='test',
        enabled=True,
        instructions='Test instructions',
        tools=tools,
    )

    result = config.to_dict()

    assert 'tools' in result
    assert len(result['tools']) == 1
    assert result['tools'][0]['key'] == 'my-tool'
    assert result['tools'][0]['version'] == 2
