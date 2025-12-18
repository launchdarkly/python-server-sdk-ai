"""Factory for creating AIProvider instances based on the provider configuration."""

import importlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from ldai.models import AIConfigKind
from ldai.providers.ai_provider import AIProvider

# List of supported AI providers
SUPPORTED_AI_PROVIDERS: List[str] = [
    # Multi-provider packages should be last in the list
    # 'langchain',  # TODO: Uncomment when langchain provider package is introduced
]

# Type representing the supported AI providers
# TODO: Update this type when provider packages are introduced
# SupportedAIProvider = Literal['langchain']
SupportedAIProvider = Literal['none']  # Placeholder until providers are added


class AIProviderFactory:
    """
    Factory for creating AIProvider instances based on the provider configuration.
    """

    @staticmethod
    async def create(
        ai_config: AIConfigKind,
        logger: Optional[Any] = None,
        default_ai_provider: Optional[SupportedAIProvider] = None,
    ) -> Optional[AIProvider]:
        """
        Create an AIProvider instance based on the AI configuration.

        This method attempts to load provider-specific implementations dynamically.
        Returns None if the provider is not supported.

        :param ai_config: The AI configuration
        :param logger: Optional logger for logging provider initialization
        :param default_ai_provider: Optional default AI provider to use
        :return: AIProvider instance or None if not supported
        """
        provider_name = ai_config.provider.name.lower() if ai_config.provider else None
        # Determine which providers to try based on default_ai_provider
        providers_to_try = AIProviderFactory._get_providers_to_try(default_ai_provider, provider_name)

        # Try each provider in order
        for provider_type in providers_to_try:
            provider = await AIProviderFactory._try_create_provider(provider_type, ai_config, logger)
            if provider:
                return provider

        # If no provider was successfully created, log a warning
        if logger:
            logger.warn(
                f"Provider is not supported or failed to initialize: {provider_name or 'unknown'}"
            )
        return None

    @staticmethod
    def _get_providers_to_try(
        default_ai_provider: Optional[SupportedAIProvider],
        provider_name: Optional[str],
    ) -> List[SupportedAIProvider]:
        """
        Determine which providers to try based on default_ai_provider and provider_name.

        :param default_ai_provider: Optional default provider to use
        :param provider_name: Optional provider name from config
        :return: List of providers to try in order
        """
        # If default_ai_provider is set, only try that specific provider
        if default_ai_provider:
            return [default_ai_provider]

        # If no default_ai_provider is set, try all providers in order
        provider_set = set()

        # First try the specific provider if it's supported
        if provider_name and provider_name in SUPPORTED_AI_PROVIDERS:
            provider_set.add(provider_name)  # type: ignore

        # Then try multi-provider packages, but avoid duplicates
        # TODO: Uncomment when langchain provider package is introduced
        # multi_provider_packages: List[SupportedAIProvider] = ['langchain']
        # for provider in multi_provider_packages:
        #     provider_set.add(provider)

        # Return list of providers, converting from set
        # The set contains strings that should be valid SupportedAIProvider values
        return list(provider_set)  # type: ignore[arg-type]

    @staticmethod
    async def _try_create_provider(
        provider_type: SupportedAIProvider,
        ai_config: AIConfigKind,
        logger: Optional[Any] = None,
    ) -> Optional[AIProvider]:
        """
        Try to create a provider of the specified type.

        :param provider_type: Type of provider to create
        :param ai_config: AI configuration
        :param logger: Optional logger
        :return: AIProvider instance or None if creation failed
        """
        # Handle built-in providers (part of this package)
        # TODO: Uncomment when langchain provider package is introduced
        # if provider_type == 'langchain':
        #     try:
        #         from ldai.providers.langchain import LangChainProvider
        #         return await LangChainProvider.create(ai_config, logger)
        #     except ImportError as error:
        #         if logger:
        #             logger.warn(
        #                 f"Error creating LangChainProvider: {error}. "
        #                 f"Make sure langchain and langchain-core packages are installed."
        #             )
        #         return None

        # For future external providers, use dynamic import
        provider_mappings: Dict[str, Tuple[str, str]] = {
            # 'openai': ('launchdarkly_server_sdk_ai_openai', 'OpenAIProvider'),
            # 'vercel': ('launchdarkly_server_sdk_ai_vercel', 'VercelProvider'),
        }

        if provider_type not in provider_mappings:
            return None

        package_name, provider_class_name = provider_mappings[provider_type]
        return await AIProviderFactory._create_provider(
            package_name, provider_class_name, ai_config, logger
        )

    @staticmethod
    async def _create_provider(
        package_name: str,
        provider_class_name: str,
        ai_config: AIConfigKind,
        logger: Optional[Any] = None,
    ) -> Optional[AIProvider]:
        """
        Create a provider instance dynamically.

        :param package_name: Name of the package containing the provider
        :param provider_class_name: Name of the provider class
        :param ai_config: AI configuration
        :param logger: Optional logger
        :return: AIProvider instance or None if creation failed
        """
        try:
            # Try to dynamically import the provider
            # This will work if the package is installed
            module = importlib.import_module(package_name)
            provider_class: Type[AIProvider] = getattr(module, provider_class_name)

            provider = await provider_class.create(ai_config, logger)
            if logger:
                provider_name = ai_config.provider.name if ai_config.provider else 'unknown'
                logger.debug(
                    f"Successfully created AIProvider for: {provider_name} "
                    f"with package {package_name}"
                )
            return provider
        except (ImportError, AttributeError, Exception) as error:
            # If the provider is not available or creation fails, return None
            if logger:
                logger.warn(
                    f"Error creating AIProvider for: {ai_config.provider.name if ai_config.provider else 'unknown'} "
                    f"with package {package_name}: {error}"
                )
            return None
