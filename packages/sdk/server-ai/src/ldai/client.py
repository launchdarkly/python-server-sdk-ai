import logging
from typing import Any, Dict, List, Optional, Tuple

import chevron
from ldclient import Context
from ldclient.client import LDClient

from ldai.chat import Chat
from ldai.judge import Judge
from ldai.models import (AIAgentConfig, AIAgentConfigDefault,
                         AIAgentConfigRequest, AIAgents, AICompletionConfig,
                         AICompletionConfigDefault, AIJudgeConfig,
                         AIJudgeConfigDefault, JudgeConfiguration, LDMessage,
                         ModelConfig, ProviderConfig)
from ldai.providers.ai_provider_factory import (AIProviderFactory,
                                                SupportedAIProvider)
from ldai.tracker import LDAIConfigTracker


class LDAIClient:
    """The LaunchDarkly AI SDK client object."""

    def __init__(self, client: LDClient):
        self._client = client
        self._logger = logging.getLogger('ldclient.ai')

    def completion_config(
        self,
        key: str,
        context: Context,
        default_value: AICompletionConfigDefault,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AICompletionConfig:
        """
        Get the value of a completion configuration.

        :param key: The key of the completion configuration.
        :param context: The context to evaluate the completion configuration in.
        :param default_value: The default value of the completion configuration.
        :param variables: Additional variables for the completion configuration.
        :return: The completion configuration with a tracker used for gathering metrics.
        """
        self._client.track('$ld:ai:config:function:single', context, key, 1)

        model, provider, messages, instructions, tracker, enabled, judge_configuration = self.__evaluate(
            key, context, default_value.to_dict(), variables
        )

        config = AICompletionConfig(
            key=key,
            enabled=bool(enabled),
            model=model,
            messages=messages,
            provider=provider,
            tracker=tracker,
            judge_configuration=judge_configuration,
        )

        return config

    def config(
        self,
        key: str,
        context: Context,
        default_value: AICompletionConfigDefault,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AICompletionConfig:
        """
        Get the value of a model configuration.

        .. deprecated:: Use :meth:`completion_config` instead. This method will be removed in a future version.

        :param key: The key of the model configuration.
        :param context: The context to evaluate the model configuration in.
        :param default_value: The default value of the model configuration.
        :param variables: Additional variables for the model configuration.
        :return: The value of the model configuration along with a tracker used for gathering metrics.
        """
        return self.completion_config(key, context, default_value, variables)

    def judge_config(
        self,
        key: str,
        context: Context,
        default_value: AIJudgeConfigDefault,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AIJudgeConfig:
        """
        Get the value of a judge configuration.

        :param key: The key of the judge configuration.
        :param context: The context to evaluate the judge configuration in.
        :param default_value: The default value of the judge configuration.
        :param variables: Additional variables for the judge configuration.
        :return: The judge configuration with a tracker used for gathering metrics.
        """
        self._client.track('$ld:ai:judge:function:single', context, key, 1)

        model, provider, messages, instructions, tracker, enabled, judge_configuration = self.__evaluate(
            key, context, default_value.to_dict(), variables
        )

        # Extract evaluation_metric_keys from the variation
        variation = self._client.variation(key, context, default_value.to_dict())
        evaluation_metric_keys = variation.get('evaluationMetricKeys', default_value.evaluation_metric_keys or [])

        config = AIJudgeConfig(
            key=key,
            enabled=bool(enabled),
            evaluation_metric_keys=evaluation_metric_keys,
            model=model,
            messages=messages,
            provider=provider,
            tracker=tracker,
        )

        return config

    async def create_judge(
        self,
        key: str,
        context: Context,
        default_value: AIJudgeConfigDefault,
        variables: Optional[Dict[str, Any]] = None,
        default_ai_provider: Optional[SupportedAIProvider] = None,
    ) -> Optional[Judge]:
        """
        Creates and returns a new Judge instance for AI evaluation.

        :param key: The key identifying the AI judge configuration to use
        :param context: Standard Context used when evaluating flags
        :param default_value: A default value representing a standard AI config result
        :param variables: Dictionary of values for instruction interpolation.
            The variables `message_history` and `response_to_evaluate` are reserved for the judge and will be ignored.
        :param default_ai_provider: Optional default AI provider to use.
        :return: Judge instance or None if disabled/unsupported

        Example::

            judge = client.create_judge(
                "relevance-judge",
                context,
                AIJudgeConfigDefault(
                    enabled=True,
                    model=ModelConfig("gpt-4"),
                    provider=ProviderConfig("openai"),
                    evaluation_metric_keys=['$ld:ai:judge:relevance'],
                    messages=[LDMessage(role='system', content='You are a relevance judge.')]
                ),
                variables={'metric': "relevance"}
            )

            if judge:
                result = await judge.evaluate("User question", "AI response")
                if result and result.evals:
                    relevance_eval = result.evals.get('$ld:ai:judge:relevance')
                    if relevance_eval:
                        print('Relevance score:', relevance_eval.score)
        """
        self._client.track('$ld:ai:judge:function:createJudge', context, key, 1)

        try:
            # Warn if reserved variables are provided
            if variables:
                if 'message_history' in variables:
                    # Note: Python doesn't have a logger on the client, but we could add one
                    pass  # Would log warning if logger available
                if 'response_to_evaluate' in variables:
                    pass  # Would log warning if logger available

            # Overwrite reserved variables to ensure they remain as placeholders for judge evaluation
            extended_variables = dict(variables) if variables else {}
            extended_variables['message_history'] = '{{message_history}}'
            extended_variables['response_to_evaluate'] = '{{response_to_evaluate}}'

            judge_config = self.judge_config(key, context, default_value, extended_variables)

            if not judge_config.enabled or not judge_config.tracker:
                # Would log info if logger available
                return None

            # Create AI provider for the judge
            provider = await AIProviderFactory.create(judge_config, self._logger, default_ai_provider)
            if not provider:
                return None

            return Judge(judge_config, judge_config.tracker, provider, self._logger)
        except Exception as error:
            # Would log error if logger available
            return None

    async def _initialize_judges(
        self,
        judge_configs: List[JudgeConfiguration.Judge],
        context: Context,
        variables: Optional[Dict[str, Any]] = None,
        default_ai_provider: Optional[SupportedAIProvider] = None,
    ) -> Dict[str, Judge]:
        """
        Initialize judges from judge configurations.

        :param judge_configs: List of judge configurations
        :param context: Standard Context used when evaluating flags
        :param variables: Dictionary of values for instruction interpolation
        :param default_ai_provider: Optional default AI provider to use
        :return: Dictionary of judge instances keyed by their configuration keys
        """
        judges: Dict[str, Judge] = {}

        async def create_judge_for_config(judge_key: str):
            judge = await self.create_judge(
                judge_key,
                context,
                AIJudgeConfigDefault(enabled=False),
                variables,
                default_ai_provider,
            )
            return judge_key, judge

        judge_promises = [
            create_judge_for_config(judge_config.key)
            for judge_config in judge_configs
        ]

        import asyncio
        results = await asyncio.gather(*judge_promises, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            judge_key, judge = result  # type: ignore[misc]
            if judge:
                judges[judge_key] = judge

        return judges

    async def create_chat(
        self,
        key: str,
        context: Context,
        default_value: AICompletionConfigDefault,
        variables: Optional[Dict[str, Any]] = None,
        default_ai_provider: Optional[SupportedAIProvider] = None,
    ) -> Optional[Chat]:
        """
        Creates and returns a new Chat instance for AI conversations.

        :param key: The key identifying the AI completion configuration to use
        :param context: Standard Context used when evaluating flags
        :param default_value: A default value representing a standard AI config result
        :param variables: Dictionary of values for instruction interpolation
        :param default_ai_provider: Optional default AI provider to use
        :return: Chat instance or None if disabled/unsupported

        Example::

            chat = await client.create_chat(
                "customer-support-chat",
                context,
                AICompletionConfigDefault(
                    enabled=True,
                    model=ModelConfig("gpt-4"),
                    provider=ProviderConfig("openai"),
                    messages=[LDMessage(role='system', content='You are a helpful assistant.')]
                ),
                variables={'customerName': 'John'}
            )

            if chat:
                response = await chat.invoke("I need help with my order")
                print(response.message.content)

                # Access conversation history
                messages = chat.get_messages()
                print(f"Conversation has {len(messages)} messages")
        """
        self._client.track('$ld:ai:config:function:createChat', context, key, 1)
        if self._logger:
            self._logger.debug(f"Creating chat for key: {key}")
        config = self.completion_config(key, context, default_value, variables)

        if not config.enabled or not config.tracker:
            # Would log info if logger available
            return None

        provider = await AIProviderFactory.create(config, self._logger, default_ai_provider)
        if not provider:
            return None

        judges = {}
        if config.judge_configuration and config.judge_configuration.judges:
            judges = await self._initialize_judges(
                config.judge_configuration.judges,
                context,
                variables,
                default_ai_provider,
            )

        return Chat(config, config.tracker, provider, judges, self._logger)

    def agent_config(
        self,
        key: str,
        context: Context,
        default_value: AIAgentConfigDefault,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AIAgentConfig:
        """
        Retrieve a single AI Config agent.

        This method retrieves a single agent configuration with instructions
        dynamically interpolated using the provided variables and context data.

        Example::

            agent = client.agent_config(
                'research_agent',
                context,
                AIAgentConfigDefault(
                    enabled=True,
                    model=ModelConfig('gpt-4'),
                    instructions="You are a research assistant specializing in {{topic}}."
                ),
                variables={'topic': 'climate change'}
            )

            if agent.enabled:
                research_result = agent.instructions  # Interpolated instructions
                agent.tracker.track_success()

        :param key: The agent configuration key.
        :param context: The context to evaluate the agent configuration in.
        :param default_value: Default agent values.
        :param variables: Variables for interpolation.
        :return: Configured AIAgentConfig instance.
        """
        # Track single agent usage
        self._client.track(
            "$ld:ai:agent:function:single",
            context,
            key,
            1
        )

        return self.__evaluate_agent(key, context, default_value, variables)

    def agent(
        self,
        config: AIAgentConfigRequest,
        context: Context,
    ) -> AIAgentConfig:
        """
        Retrieve a single AI Config agent.

        .. deprecated:: Use :meth:`agent_config` instead. This method will be removed in a future version.

        :param config: The agent configuration to use.
        :param context: The context to evaluate the agent configuration in.
        :return: Configured AIAgentConfig instance.
        """
        return self.agent_config(config.key, context, config.default_value, config.variables)

    def agent_configs(
        self,
        agent_configs: List[AIAgentConfigRequest],
        context: Context,
    ) -> AIAgents:
        """
        Retrieve multiple AI agent configurations.

        This method allows you to retrieve multiple agent configurations in a single call,
        with each agent having its own default configuration and variables for instruction
        interpolation.

        Example::

            agents = client.agent_configs([
                AIAgentConfigRequest(
                    key='research_agent',
                    default_value=AIAgentConfigDefault(
                        enabled=True,
                        instructions='You are a research assistant.'
                    ),
                    variables={'topic': 'climate change'}
                ),
                AIAgentConfigRequest(
                    key='writing_agent',
                    default_value=AIAgentConfigDefault(
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
        :return: Dictionary mapping agent keys to their AIAgentConfig configurations.
        """
        # Track multiple agents usage
        agent_count = len(agent_configs)
        self._client.track(
            "$ld:ai:agent:function:multiple",
            context,
            agent_count,
            agent_count
        )

        result: AIAgents = {}

        for config in agent_configs:
            agent = self.__evaluate_agent(
                config.key,
                context,
                config.default_value,
                config.variables
            )
            result[config.key] = agent

        return result

    def agents(
        self,
        agent_configs: List[AIAgentConfigRequest],
        context: Context,
    ) -> AIAgents:
        """
        Retrieve multiple AI agent configurations.

        .. deprecated:: Use :meth:`agent_configs` instead. This method will be removed in a future version.

        :param agent_configs: List of agent configurations to retrieve.
        :param context: The context to evaluate the agent configurations in.
        :return: Dictionary mapping agent keys to their AIAgentConfig configurations.
        """
        return self.agent_configs(agent_configs, context)

    def __evaluate(
        self,
        key: str,
        context: Context,
        default_dict: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        Optional[ModelConfig], Optional[ProviderConfig], Optional[List[LDMessage]],
        Optional[str], LDAIConfigTracker, bool, Optional[Any]
    ]:
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

        # Extract judge configuration
        judge_configuration = None
        if 'judgeConfiguration' in variation and isinstance(variation['judgeConfiguration'], dict):
            judge_config = variation['judgeConfiguration']
            if 'judges' in judge_config and isinstance(judge_config['judges'], list):
                judges = [
                    JudgeConfiguration.Judge(
                        key=judge['key'],
                        sampling_rate=judge['samplingRate']
                    )
                    for judge in judge_config['judges']
                    if isinstance(judge, dict) and 'key' in judge and 'samplingRate' in judge
                ]
                if judges:
                    judge_configuration = JudgeConfiguration(judges=judges)

        return model, provider_config, messages, instructions, tracker, enabled, judge_configuration

    def __evaluate_agent(
        self,
        key: str,
        context: Context,
        default_value: AIAgentConfigDefault,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AIAgentConfig:
        """
        Internal method to evaluate an agent configuration.

        :param key: The agent configuration key.
        :param context: The evaluation context.
        :param default_value: Default agent values.
        :param variables: Variables for interpolation.
        :return: Configured AIAgentConfig instance.
        """
        model, provider, messages, instructions, tracker, enabled, judge_configuration = self.__evaluate(
            key, context, default_value.to_dict(), variables
        )

        # For agents, prioritize instructions over messages
        final_instructions = instructions if instructions is not None else default_value.instructions

        return AIAgentConfig(
            key=key,
            enabled=bool(enabled) if enabled is not None else (default_value.enabled or False),
            model=model or default_value.model,
            provider=provider or default_value.provider,
            instructions=final_instructions,
            tracker=tracker,
            judge_configuration=judge_configuration or default_value.judge_configuration,
        )

    def __interpolate_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Interpolate the template with the given variables using Mustache format.

        :param template: The template string.
        :param variables: The variables to interpolate into the template.
        :return: The interpolated string.
        """
        return chevron.render(template, variables)
