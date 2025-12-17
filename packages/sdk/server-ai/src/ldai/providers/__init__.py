"""AI Provider interfaces and factory for LaunchDarkly AI SDK."""

from ldai.providers.ai_provider import AIProvider
from ldai.providers.ai_provider_factory import (AIProviderFactory,
                                                SupportedAIProvider)

# Export LangChain provider if available
# TODO: Uncomment when langchain provider package is introduced
# try:
#     from ldai.providers.langchain import LangChainProvider
#     __all__ = [
#         'AIProvider',
#         'AIProviderFactory',
#         'LangChainProvider',
#         'SupportedAIProvider',
#     ]
# except ImportError:
#     __all__ = [
#         'AIProvider',
#         'AIProviderFactory',
#         'SupportedAIProvider',
#     ]

__all__ = [
    'AIProvider',
    'AIProviderFactory',
    'SupportedAIProvider',
]
