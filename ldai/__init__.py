__version__ = "0.10.1"  # x-release-please-version

# Export main client
from ldai.client import LDAIClient

# Export models for convenience
from ldai.models import (
    AIConfig,
    LDAIAgent,
    LDAIAgentConfig,
    LDAIAgentDefaults,
    LDMessage,
    ModelConfig,
    ProviderConfig,
)

__all__ = [
    'LDAIClient',
    'AIConfig',
    'LDAIAgent',
    'LDAIAgentConfig',
    'LDAIAgentDefaults',
    'LDMessage',
    'ModelConfig',
    'ProviderConfig',
]
