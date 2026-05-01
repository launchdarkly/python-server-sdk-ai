__version__ = "0.18.0"  # x-release-please-version

from ldclient import log

from ldai.agent_graph import AgentGraphDefinition
from ldai.chat import Chat  # Deprecated — use ManagedModel
from ldai.client import LDAIClient
from ldai.evaluator import Evaluator
from ldai.judge import Judge
from ldai.managed_agent import ManagedAgent
from ldai.managed_agent_graph import ManagedAgentGraph
from ldai.managed_model import ManagedModel
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
    LDTool,
    ModelConfig,
    ProviderConfig,
)
from ldai.providers import (
    AgentGraphResult,
    AgentGraphRunner,
    AgentGraphRunnerResult,
    AgentResult,
    AgentRunner,
    GraphMetrics,
    GraphMetricSummary,
    ManagedGraphResult,
    ManagedResult,
    Runner,
    RunnerResult,
    ToolRegistry,
)
from ldai.providers.types import JudgeResult
from ldai.tracker import AIGraphTracker, LDAIMetricSummary

__all__ = [
    'LDAIClient',
    'Evaluator',
    'AgentRunner',
    'AgentGraphRunner',
    'AgentResult',
    'AgentGraphResult',
    'AgentGraphRunnerResult',
    'GraphMetrics',
    'GraphMetricSummary',
    'ManagedGraphResult',
    'ManagedResult',
    'Runner',
    'RunnerResult',
    'LDAIMetricSummary',
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
    'ManagedAgent',
    'ManagedModel',
    'ManagedAgentGraph',
    'AgentGraphDefinition',
    'Judge',
    'JudgeConfiguration',
    'JudgeResult',
    'LDTool',
    'LDMessage',
    'ModelConfig',
    'ProviderConfig',
    'log',
    # Deprecated exports
    'AIConfig',
    'Chat',
    'LDAIAgent',
    'LDAIAgentConfig',
    'LDAIAgentDefaults',
]
