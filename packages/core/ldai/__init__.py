__version__ = "0.10.1"  # x-release-please-version

# Extend __path__ to support namespace packages at the ldai level
# This allows provider packages (like launchdarkly-server-sdk-ai-langchain)
# to extend ldai.providers.* even though ldai itself has an __init__.py
import sys
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Export main client
from ldai.client import LDAIClient

# Export models for convenience
from ldai.models import (
    AIAgentConfig,
    AIAgentConfigDefault,
    AIAgentConfigRequest,
    AIAgents,
    AICompletionConfig,
    AICompletionConfigDefault,
    AIJudgeConfig,
    AIJudgeConfigDefault,
    JudgeConfiguration,
    LDMessage,
    ModelConfig,
    ProviderConfig,
    # Deprecated aliases for backward compatibility
    AIConfig,
    LDAIAgent,
    LDAIAgentConfig,
    LDAIAgentDefaults,
)

# Export judge
from ldai.judge import AIJudge

# Export chat
from ldai.chat import TrackedChat

# Export judge types
from ldai.providers.types import EvalScore, JudgeResponse

__all__ = [
    'LDAIClient',
    'AIAgentConfig',
    'AIAgentConfigDefault',
    'AIAgentConfigRequest',
    'AIAgents',
    'AICompletionConfig',
    'AICompletionConfigDefault',
    'AIJudgeConfig',
    'AIJudgeConfigDefault',
    'AIJudge',
    'TrackedChat',
    'EvalScore',
    'JudgeConfiguration',
    'JudgeResponse',
    'LDMessage',
    'ModelConfig',
    'ProviderConfig',
    # Deprecated exports
    'AIConfig',
    'LDAIAgent',
    'LDAIAgentConfig',
    'LDAIAgentDefaults',
]
