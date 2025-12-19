__version__ = "0.11.0"  # x-release-please-version

# Export main client
# Export chat
from ldai.chat import Chat
from ldai.client import LDAIClient
# Export judge
from ldai.judge import Judge
# Export models for convenience
from ldai.models import (  # Deprecated aliases for backward compatibility
    AIAgentConfig, AIAgentConfigDefault, AIAgentConfigRequest, AIAgents,
    AICompletionConfig, AICompletionConfigDefault, AIConfig, AIJudgeConfig,
    AIJudgeConfigDefault, JudgeConfiguration, LDAIAgent, LDAIAgentConfig,
    LDAIAgentDefaults, LDMessage, ModelConfig, ProviderConfig)
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
    'Judge',
    'Chat',
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
