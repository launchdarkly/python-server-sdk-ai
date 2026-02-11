__version__ = "0.14.0"  # x-release-please-version

from ldclient import log

from ldai.agent_graph import AgentGraphDefinition
from ldai.chat import Chat
from ldai.client import LDAIClient
from ldai.judge import Judge
from ldai.optimization import (
    AutoCommitConfig,
    JudgeResult, Message, OptimizeContext, OptimizeJudgeContext,
    OptimizeOptions, OptimizationJudge, StructuredOutputTool)
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
    'JudgeResult',
    'LDMessage',
    'Message',
    'ModelConfig',
    'AutoCommitConfig',
    'OptimizeContext',
    'OptimizeJudgeContext',
    'OptimizeOptions',
    'OptimizationJudge',
    'ProviderConfig',
    'StructuredOutputTool',
    'log',
    # Deprecated exports
    'AIConfig',
    'LDAIAgent',
    'LDAIAgentConfig',
    'LDAIAgentDefaults',
]
