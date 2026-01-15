__version__ = "0.12.0"  # x-release-please-version

from ldclient import log

from ldai.agent_graph import AgentGraphDefinition, AIAgentGraphResponse
from ldai.chat import Chat
from ldai.client import LDAIClient
from ldai.judge import Judge
from ldai.models import (  # Deprecated aliases for backward compatibility
    AIAgentConfig, AIAgentConfigDefault, AIAgentConfigRequest,
    AIAgentGraphConfig, AIAgents, AICompletionConfig,
    AICompletionConfigDefault, AIConfig, AIJudgeConfig, AIJudgeConfigDefault,
    Edge, JudgeConfiguration, LDAIAgent, LDAIAgentConfig, LDAIAgentDefaults,
    LDMessage, ModelConfig, ProviderConfig)
from ldai.providers.types import EvalScore, JudgeResponse

__all__ = [
    'LDAIClient',
    'AIAgentConfig',
    'AIAgentConfigDefault',
    'AIAgentConfigRequest',
    'AIAgents',
    'AIAgentGraphConfig',
    'AIAgentGraphResponse',
    'Edge',
    'AICompletionConfig',
    'AICompletionConfigDefault',
    'AIJudgeConfig',
    'AIJudgeConfigDefault',
    'Chat',
    'EvalScore',
    'AgentGraphDefinition',
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
