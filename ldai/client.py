from typing import Any, Dict, List, Optional, Tuple

import chevron
from ldclient import Context
from ldclient.client import LDClient

from ldai.models import (
    AIConfig,
    LDAIAgent,
    LDAIAgentConfig,
    LDAIAgentDefaults,
    LDAIAgents,
    LDMessage,
    ModelConfig,
    ProviderConfig,
)
from ldai.tracker import LDAIConfigTracker


class LDAIClient:
    """The LaunchDarkly AI SDK client object."""

    def __init__(self, client: LDClient):
        self._client = client

    def config(
        self,
        key: str,
        context: Context,
        default_value: AIConfig,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AIConfig, LDAIConfigTracker]:
        """
        Get the value of a model configuration.

        :param key: The key of the model configuration.
        :param context: The context to evaluate the model configuration in.
        :param default_value: The default value of the model configuration.
        :param variables: Additional variables for the model configuration.
        :return: The value of the model configuration along with a tracker used for gathering metrics.
        """
        self._client.track('$ld:ai:config:function:single', context, key, 1)

        model, provider, messages, instructions, tracker, enabled = self.__evaluate(key, context, default_value.to_dict(), variables)

        config = AIConfig(
            enabled=bool(enabled),
            model=model,
            messages=messages,
            provider=provider,
        )

        return config, tracker

    def agent(
        self,
        config: LDAIAgentConfig,
        context: Context,
    ) -> LDAIAgent:
        """
        Retrieve a single AI Config agent.

        This method retrieves a single agent configuration with instructions
        dynamically interpolated using the provided variables and context data.

        Example::

            agent = client.agent(LDAIAgentConfig(
                key='research_agent',
                default_value=LDAIAgentDefaults(
                    enabled=True,
                    model=ModelConfig('gpt-4'),
                    instructions="You are a research assistant specializing in {{topic}}."
                ),
                variables={'topic': 'climate change'}
            ), context)

            if agent.enabled:
                research_result = agent.instructions  # Interpolated instructions
                agent.tracker.track_success()

        :param config: The agent configuration to use.
        :param context: The context to evaluate the agent configuration in.
        :return: Configured LDAIAgent instance.
        """
        # Track single agent usage
        self._client.track(
            "$ld:ai:agent:function:single",
            context,
            config.key,
            1
        )

        return self.__evaluate_agent(config.key, context, config.default_value, config.variables)

    def agents(
        self,
        agent_configs: List[LDAIAgentConfig],
        context: Context,
    ) -> LDAIAgents:
        """
        Retrieve multiple AI agent configurations.

        This method allows you to retrieve multiple agent configurations in a single call,
        with each agent having its own default configuration and variables for instruction
        interpolation.

        Example::

            agents = client.agents([
                LDAIAgentConfig(
                    key='research_agent',
                    default_value=LDAIAgentDefaults(
                        enabled=True,
                        instructions='You are a research assistant.'
                    ),
                    variables={'topic': 'climate change'}
                ),
                LDAIAgentConfig(
                    key='writing_agent',
                    default_value=LDAIAgentDefaults(
                        enabled=True,
                        instructions='You are a writing assistant.'
                    ),
                    variables={'style': 'academic'}
                )
            ], context)

            research_result = agents["research_agent"].instructions
            agents["research_agent"].tracker.track_success()

        :param agent_configs: List of agent configurations to retrieve.
        :param context: The context to evaluate the agent configurations in.
        :return: Dictionary mapping agent keys to their LDAIAgent configurations.
        """
        # Track multiple agents usage
        agent_count = len(agent_configs)
        self._client.track(
            "$ld:ai:agent:function:multiple",
            context,
            agent_count,
            agent_count
        )

        result: LDAIAgents = {}

        for config in agent_configs:
            agent = self.__evaluate_agent(
                config.key,
                context,
                config.default_value,
                config.variables
            )
            result[config.key] = agent

        return result

    def __evaluate(
        self,
        key: str,
        context: Context,
        default_dict: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ModelConfig], Optional[ProviderConfig], Optional[List[LDMessage]], Optional[str], LDAIConfigTracker, bool]:
        """
        Internal method to evaluate a configuration and extract components.

        :param key: The configuration key.
        :param context: The evaluation context.
        :param default_dict: Default configuration as dictionary.
        :param variables: Variables for interpolation.
        :return: Tuple of (model, provider, messages, instructions, tracker, enabled).
        """
        variation = self._client.variation(key, context, default_dict)

        all_variables = {}
        if variables:
            all_variables.update(variables)
        all_variables['ldctx'] = context.to_dict()

        # Extract messages
        messages = None
        if 'messages' in variation and isinstance(variation['messages'], list) and all(
            isinstance(entry, dict) for entry in variation['messages']
        ):
            messages = [
                LDMessage(
                    role=entry['role'],
                    content=self.__interpolate_template(
                        entry['content'], all_variables
                    ),
                )
                for entry in variation['messages']
            ]

        # Extract instructions
        instructions = None
        if 'instructions' in variation and isinstance(variation['instructions'], str):
            instructions = self.__interpolate_template(variation['instructions'], all_variables)

        # Extract provider config
        provider_config = None
        if 'provider' in variation and isinstance(variation['provider'], dict):
            provider = variation['provider']
            provider_config = ProviderConfig(provider.get('name', ''))

        # Extract model config
        model = None
        if 'model' in variation and isinstance(variation['model'], dict):
            parameters = variation['model'].get('parameters', None)
            custom = variation['model'].get('custom', None)
            model = ModelConfig(
                name=variation['model']['name'],
                parameters=parameters,
                custom=custom
            )

        # Create tracker
        tracker = LDAIConfigTracker(
            self._client,
            variation.get('_ldMeta', {}).get('variationKey', ''),
            key,
            int(variation.get('_ldMeta', {}).get('version', 1)),
            model.name if model else '',
            provider_config.name if provider_config else '',
            context,
        )

        enabled = variation.get('_ldMeta', {}).get('enabled', False)

        return model, provider_config, messages, instructions, tracker, enabled

    def __evaluate_agent(
        self,
        key: str,
        context: Context,
        default_value: LDAIAgentDefaults,
        variables: Optional[Dict[str, Any]] = None,
    ) -> LDAIAgent:
        """
        Internal method to evaluate an agent configuration.

        :param key: The agent configuration key.
        :param context: The evaluation context.
        :param default_value: Default agent values.
        :param variables: Variables for interpolation.
        :return: Configured LDAIAgent instance.
        """
        model, provider, messages, instructions, tracker, enabled = self.__evaluate(
            key, context, default_value.to_dict(), variables
        )

        # For agents, prioritize instructions over messages
        final_instructions = instructions if instructions is not None else default_value.instructions

        return LDAIAgent(
            enabled=bool(enabled) if enabled is not None else default_value.enabled,
            model=model or default_value.model,
            provider=provider or default_value.provider,
            instructions=final_instructions,
            tracker=tracker,
        )

    def __interpolate_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Interpolate the template with the given variables using Mustache format.

        :param template: The template string.
        :param variables: The variables to interpolate into the template.
        :return: The interpolated string.
        """
        return chevron.render(template, variables)
