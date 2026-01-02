__version__ = "0.11.0"  # x-release-please-version

from ldclient import log

from ldai.chat import Chat
from ldai.client import LDAIClient
from ldai.judge import Judge
from ldai.models import (  # Deprecated aliases for backward compatibility
    AIAgentConfig, AIAgentConfigDefault, AIAgentConfigRequest, AIAgents,
    AICompletionConfig, AICompletionConfigDefault, AIConfig, AIJudgeConfig,
    AIJudgeConfigDefault, JudgeConfiguration, LDAIAgent, LDAIAgentConfig,
    LDAIAgentDefaults, LDMessage, ModelConfig, ProviderConfig)
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
    'Chat',
    'EvalScore',
    'Judge',
    'JudgeConfiguration',
    'JudgeResponse',
    'LDMessage',
    'ModelConfig',
    'ProviderConfig',
    'log',
    # Deprecated exports
    'AIConfig',
    'LDAIAgent',
    'LDAIAgentConfig',
    'LDAIAgentDefaults',
]
