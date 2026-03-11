__version__ = "0.16.0"  # x-release-please-version

from ldclient import log

from ldai.agent_graph import AgentGraphDefinition
from ldai.managed_model import ManagedModel
from ldai.chat import Chat  # Deprecated — use ManagedModel
from ldai.client import LDAIClient
from ldai.judge import Judge
from ldai.models import (  # Deprecated aliases for backward compatibility
    AIAgentConfig,
    AIAgentConfigDefault,
    AIAgentConfigRequest,
    AIAgentGraphConfig,
    AIAgents,
    AICompletionConfig,
    AICompletionConfigDefault,
    AIConfig,
    AIJudgeConfig,
    AIJudgeConfigDefault,
    Edge,
    JudgeConfiguration,
    LDAIAgent,
    LDAIAgentConfig,
    LDAIAgentDefaults,
    LDMessage,
    ModelConfig,
    ProviderConfig,
)
from ldai.providers.types import EvalScore, JudgeResponse
from ldai.runners import AgentGraphRunner, AgentRunner
from ldai.runners.types import AgentGraphResult, AgentResult, ToolRegistry
from ldai.tracker import AIGraphTracker

__all__ = [
    'LDAIClient',
    'AgentRunner',
    'AgentGraphRunner',
    'AgentResult',
    'AgentGraphResult',
    'ToolRegistry',
    'AIAgentConfig',
    'AIAgentConfigDefault',
    'AIAgentConfigRequest',
    'AIAgents',
    'AIAgentGraphConfig',
    'AIGraphTracker',
    'Edge',
    'AICompletionConfig',
    'AICompletionConfigDefault',
    'AIJudgeConfig',
    'AIJudgeConfigDefault',
    'ManagedModel',
    'EvalScore',
    # Deprecated — use ManagedModel
    'Chat',
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
