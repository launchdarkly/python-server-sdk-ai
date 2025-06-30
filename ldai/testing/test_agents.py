import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai.client import (LDAIAgentDefaults, LDAIClient, ModelConfig,
                         ProviderConfig)


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()

    # Single agent with instructions
    td.update(
        td.flag('customer-support-agent')
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.3, 'maxTokens': 2048}},
                'provider': {'name': 'openai'},
                'instructions': 'You are a helpful customer support agent for {{company_name}}. Always be polite and professional.',
                '_ldMeta': {'enabled': True, 'variationKey': 'agent-v1', 'version': 1, 'mode': 'agent'},
            }
        )
        .variation_for_all(0)
    )

    # Agent with context interpolation
    td.update(
        td.flag('personalized-agent')
        .variations(
            {
                'model': {'name': 'claude-3', 'parameters': {'temperature': 0.5}},
                'instructions': 'Hello {{ldctx.name}}! I am your personal assistant. Your user key is {{ldctx.key}}.',
                '_ldMeta': {'enabled': True, 'variationKey': 'personal-v1', 'version': 2, 'mode': 'agent'},
            }
        )
        .variation_for_all(0)
    )

    # Agent with multi-context interpolation
    td.update(
        td.flag('multi-context-agent')
        .variations(
            {
                'model': {'name': 'gpt-3.5-turbo'},
                'instructions': 'Welcome {{ldctx.user.name}} from {{ldctx.org.name}}! Your organization tier is {{ldctx.org.tier}}.',
                '_ldMeta': {'enabled': True, 'variationKey': 'multi-v1', 'version': 1, 'mode': 'agent'},
            }
        )
        .variation_for_all(0)
    )

    # Disabled agent
    td.update(
        td.flag('disabled-agent')
        .variations(
            {
                'model': {'name': 'gpt-4'},
                'instructions': 'This agent is disabled.',
                '_ldMeta': {'enabled': False, 'variationKey': 'disabled-v1', 'version': 1, 'mode': 'agent'},
            }
        )
        .variation_for_all(0)
    )

    # Agent with minimal metadata
    td.update(
        td.flag('minimal-agent')
        .variations(
            {
                'instructions': 'Minimal agent configuration.',
                '_ldMeta': {'enabled': True},
            }
        )
        .variation_for_all(0)
    )

    # Sales assistant agent
    td.update(
        td.flag('sales-assistant')
        .variations(
            {
                'model': {'name': 'gpt-4', 'parameters': {'temperature': 0.7}},
                'provider': {'name': 'openai'},
                'instructions': 'You are a sales assistant for {{company_name}}. Help customers find the right products.',
                '_ldMeta': {'enabled': True, 'variationKey': 'sales-v1', 'version': 1, 'mode': 'agent'},
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


def test_single_agent_basic_functionality(ldai_client: LDAIClient):
    """Test basic agent retrieval and configuration."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(
        enabled=False,
        model=ModelConfig('fallback-model'),
        instructions="Default instructions"
    )
    variables = {'company_name': 'Acme Corp'}

    agents = ldai_client.agents(['customer-support-agent'], context, defaults, variables)

    assert len(agents) == 1
    assert 'customer-support-agent' in agents

    agent = agents['customer-support-agent']
    assert agent.enabled is True
    assert agent.model is not None
    assert agent.model.name == 'gpt-4'
    assert agent.model.get_parameter('temperature') == 0.3
    assert agent.model.get_parameter('maxTokens') == 2048
    assert agent.provider is not None
    assert agent.provider.name == 'openai'
    assert agent.instructions == 'You are a helpful customer support agent for Acme Corp. Always be polite and professional.'
    assert agent.tracker is not None


def test_agent_instructions_interpolation(ldai_client: LDAIClient):
    """Test that agent instructions are properly interpolated with variables."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(enabled=True, instructions="Default")
    variables = {'company_name': 'TechStart Inc'}

    agents = ldai_client.agents(['customer-support-agent'], context, defaults, variables)
    agent = agents['customer-support-agent']

    expected_instructions = 'You are a helpful customer support agent for TechStart Inc. Always be polite and professional.'
    assert agent.instructions == expected_instructions


def test_agent_context_interpolation(ldai_client: LDAIClient):
    """Test that agent instructions can access context data via ldctx."""
    context = Context.builder('user-123').name('Alice').build()
    defaults = LDAIAgentDefaults(enabled=True, instructions="Default")

    agents = ldai_client.agents(['personalized-agent'], context, defaults)
    agent = agents['personalized-agent']

    expected_instructions = 'Hello Alice! I am your personal assistant. Your user key is user-123.'
    assert agent.instructions == expected_instructions


def test_agent_multi_context_interpolation(ldai_client: LDAIClient):
    """Test agent instructions with multi-context interpolation."""
    user_context = Context.builder('user-key').name('Bob').build()
    org_context = Context.builder('org-key').kind('org').name('LaunchDarkly').set('tier', 'Enterprise').build()
    context = Context.multi_builder().add(user_context).add(org_context).build()

    defaults = LDAIAgentDefaults(enabled=True, instructions="Default")

    agents = ldai_client.agents(['multi-context-agent'], context, defaults)
    agent = agents['multi-context-agent']

    expected_instructions = 'Welcome Bob from LaunchDarkly! Your organization tier is Enterprise.'
    assert agent.instructions == expected_instructions


def test_multiple_agents_retrieval(ldai_client: LDAIClient):
    """Test retrieving multiple agents in a single call."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(
        enabled=False,
        model=ModelConfig('fallback'),
        instructions="Default"
    )
    variables = {'company_name': 'MultiCorp'}

    agents = ldai_client.agents(
        ['customer-support-agent', 'sales-assistant'],
        context,
        defaults,
        variables
    )

    assert len(agents) == 2
    assert 'customer-support-agent' in agents
    assert 'sales-assistant' in agents

    support_agent = agents['customer-support-agent']
    assert support_agent.enabled is True
    assert support_agent.instructions is not None and 'MultiCorp' in support_agent.instructions

    sales_agent = agents['sales-assistant']
    assert sales_agent.enabled is True
    assert sales_agent.instructions is not None and 'MultiCorp' in sales_agent.instructions
    assert sales_agent.model is not None and sales_agent.model.get_parameter('temperature') == 0.7


def test_disabled_agent(ldai_client: LDAIClient):
    """Test that disabled agents are properly handled."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(enabled=True, instructions="Default")

    agents = ldai_client.agents(['disabled-agent'], context, defaults)
    agent = agents['disabled-agent']

    assert agent.enabled is False
    assert agent.instructions == 'This agent is disabled.'


def test_agent_with_missing_metadata(ldai_client: LDAIClient):
    """Test agent handling when metadata is minimal or missing."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(
        enabled=False,
        model=ModelConfig('default-model'),
        instructions="Default instructions"
    )

    agents = ldai_client.agents(['minimal-agent'], context, defaults)
    agent = agents['minimal-agent']

    assert agent.enabled is True  # From flag
    assert agent.instructions == 'Minimal agent configuration.'
    assert agent.model == defaults.model  # Falls back to default
    assert agent.tracker is not None


def test_agent_uses_defaults_on_missing_flag(ldai_client: LDAIClient):
    """Test that default values are used when agent flag doesn't exist."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(
        enabled=True,
        model=ModelConfig('default-gpt', parameters={'temp': 0.5}),
        provider=ProviderConfig('default-provider'),
        instructions="You are a default assistant."
    )

    agents = ldai_client.agents(['non-existent-agent'], context, defaults)
    agent = agents['non-existent-agent']

    assert agent.enabled == defaults.enabled
    assert agent.model is not None and agent.model.name == 'default-gpt'
    assert agent.model is not None and agent.model.get_parameter('temp') == 0.5
    assert agent.provider is not None and agent.provider.name == 'default-provider'
    assert agent.instructions == defaults.instructions
    # Tracker should still be created for non-existent flags
    assert agent.tracker is not None


def test_agent_error_handling(ldai_client: LDAIClient):
    """Test that agent errors are handled gracefully."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(
        enabled=True,
        model=ModelConfig('fallback-model'),
        instructions="Fallback instructions"
    )

    # Test with a mix of valid and invalid keys
    agents = ldai_client.agents(
        ['customer-support-agent', 'invalid-flag'],
        context,
        defaults
    )

    assert len(agents) == 2

    # Valid agent should work normally
    valid_agent = agents['customer-support-agent']
    assert valid_agent.enabled is True
    assert valid_agent.tracker is not None

    # Invalid agent should use defaults but still be created
    invalid_agent = agents['invalid-flag']
    assert invalid_agent.enabled == defaults.enabled
    assert invalid_agent.model is not None and invalid_agent.model.name == 'fallback-model'
    assert invalid_agent.instructions == defaults.instructions


def test_agent_no_variables_interpolation(ldai_client: LDAIClient):
    """Test agent instructions with no variables provided."""
    context = Context.builder('user-456').name('Charlie').build()
    defaults = LDAIAgentDefaults(enabled=True, instructions="Default")

    agents = ldai_client.agents(['personalized-agent'], context, defaults)
    agent = agents['personalized-agent']

    # Should still interpolate context but not variables
    expected_instructions = 'Hello Charlie! I am your personal assistant. Your user key is user-456.'
    assert agent.instructions == expected_instructions


def test_agent_empty_agent_list(ldai_client: LDAIClient):
    """Test agents method with empty agent list."""
    context = Context.create('user-key')
    defaults = LDAIAgentDefaults(enabled=True, instructions="Default")

    agents = ldai_client.agents([], context, defaults)

    assert len(agents) == 0
    assert agents == {}


def test_agents_backwards_compatibility_with_config(ldai_client: LDAIClient):
    """Test that the existing config method still works after agent additions."""
    from ldai.client import AIConfig, LDMessage

    context = Context.create('user-key')
    default_value = AIConfig(
        enabled=True,
        model=ModelConfig('test-model'),
        messages=[LDMessage(role='system', content='Test message')]
    )

    # This should still work as before
    config, tracker = ldai_client.config('customer-support-agent', context, default_value)

    assert config.enabled is True
    assert config.model is not None
    assert tracker is not None
