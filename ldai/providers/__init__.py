"""AI Provider interfaces and factory for LaunchDarkly AI SDK."""

from ldai.providers.ai_provider import AIProvider
from ldai.providers.ai_provider_factory import (AIProviderFactory,
                                                SupportedAIProvider)

# Export LangChain provider if available
try:
    from ldai.providers.langchain import LangChainProvider
    __all__ = [
        'AIProvider',
        'AIProviderFactory',
        'LangChainProvider',
        'SupportedAIProvider',
    ]
except ImportError:
    __all__ = [
        'AIProvider',
        'AIProviderFactory',
        'SupportedAIProvider',
    ]
