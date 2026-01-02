"""Factory for creating AIProvider instances based on the provider configuration."""

from importlib import util
from typing import Any, Dict, List, Optional, Tuple, Type

from ldai import log
from ldai.models import AIConfigKind
from ldai.providers.ai_provider import AIProvider

# Supported AI providers
# Multi-provider packages should be last in the list
SUPPORTED_AI_PROVIDERS = ('openai', 'langchain')

class AIProviderFactory:
    """
    Factory for creating AIProvider instances based on the provider configuration.
    """

    @staticmethod
    async def create(
        ai_config: AIConfigKind,
        default_ai_provider: Optional[str] = None,
    ) -> Optional[AIProvider]:
        """
        Create an AIProvider instance based on the AI configuration.

        This method attempts to load provider-specific implementations dynamically.
        Returns None if the provider is not supported.

        :param ai_config: The AI configuration
        :param default_ai_provider: Optional default AI provider to use
        :return: AIProvider instance or None if not supported
        """
        provider_name = ai_config.provider.name.lower() if ai_config.provider else None
        providers_to_try = AIProviderFactory._get_providers_to_try(default_ai_provider, provider_name)

        for provider_type in providers_to_try:
            provider = await AIProviderFactory._try_create_provider(provider_type, ai_config)
            if provider:
                log.debug(
                    f"Successfully created AIProvider for: {provider_name} "
                    f"with provider type: {provider_type} for AIConfig: {ai_config.key}"
                )
                return provider

        log.warn(
            f"Provider is not supported or failed to initialize: {provider_name}"
        )
        return None

    @staticmethod
    def _get_providers_to_try(
        default_ai_provider: Optional[str],
        provider_name: Optional[str],
    ) -> List[str]:
        """
        Determine which providers to try based on default_ai_provider and provider_name.

        :param default_ai_provider: Optional default provider to use
        :param provider_name: Optional provider name from config
        :return: List of providers to try in order
        """
        if default_ai_provider:
            return [default_ai_provider]

        providers = []

        if provider_name and provider_name in SUPPORTED_AI_PROVIDERS:
            providers.append(provider_name)

        # Then try multi-provider packages, but avoid duplicates
        multi_provider_packages: List[str] = ['langchain']
        for provider in multi_provider_packages:
            if provider not in providers:
                providers.append(provider)

        return providers

    @staticmethod
    async def _try_create_provider(
        provider_type: str,
        ai_config: AIConfigKind,
    ) -> Optional[AIProvider]:
        """
        Try to create a provider of the specified type.

        :param provider_type: Type of provider to create
        :param ai_config: AI configuration
        :return: AIProvider instance or None if creation failed
        """
        try:
            if provider_type == 'langchain':
                AIProviderFactory._pkg_exists('ldai_langchain')
                from ldai_langchain import LangChainProvider  # pyright: ignore[reportMissingImports]
                return await LangChainProvider.create(ai_config)

            if provider_type == 'openai':
                AIProviderFactory._pkg_exists('ldai_openai')
                from ldai_openai import OpenAIProvider  # pyright: ignore[reportMissingImports]
                return await OpenAIProvider.create(ai_config)

            log.warn(
                f"Provider {provider_type} is not supported. "
                f"Supported providers are: {SUPPORTED_AI_PROVIDERS}"
            )

            return None
        except ImportError as error:
            log.warn(
                f"Error creating {provider_type} provider: {error}. "
                f"Make sure the {provider_type} package is installed."
            )
            return None

    @staticmethod
    def _pkg_exists(package_name: str) -> None:
        """
        Check if a package exists.

        :param package_name: Name of the package to check
        :return: None if the package exists, otherwise raises an ImportError
        """
        if util.find_spec(package_name) is None:
            raise ImportError(f"Package {package_name} not found")
