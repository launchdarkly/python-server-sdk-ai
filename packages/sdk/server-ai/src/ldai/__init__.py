__version__ = "0.12.0"  # x-release-please-version

from ldclient import log

from ldai.chat import Chat
from ldai.client import LDAIClient
from ldai.agent_graph import AgentGraph
from ldai.judge import Judge
from ldai.models import (  # Deprecated aliases for backward compatibility
    AIAgentConfig, AIAgentConfigDefault, AIAgentConfigRequest, AIAgents,
    AICompletionConfig, AICompletionConfigDefault, AIConfig, AIJudgeConfig,
    AIJudgeConfigDefault, JudgeConfiguration, LDAIAgent, LDAIAgentConfig,
    LDAIAgentDefaults, LDMessage, ModelConfig, ProviderConfig, AIAgentGraph, AIAgentGraphEdge)
from ldai.providers.types import EvalScore, JudgeResponse

__all__ = [
    'LDAIClient',
    'AIAgentConfig',
    'AIAgentConfigDefault',
    'AIAgentConfigRequest',
    'AIAgents',
    'AIAgentGraph',
    'AIAgentGraphEdge',
    'AICompletionConfig',
    'AICompletionConfigDefault',
    'AIJudgeConfig',
    'AIJudgeConfigDefault',
    'Chat',
    'EvalScore',
    'AgentGraph',
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
