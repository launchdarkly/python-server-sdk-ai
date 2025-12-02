from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ldai.config.types import LDMessage, ModelConfig, ProviderConfig
from ldai.tracker import LDAIConfigTracker

# ============================================================================
# Judge Types
# ============================================================================


@dataclass(frozen=True)
class JudgeConfiguration:
    """
    Configuration for judge attachment to AI Configs.
    """

    @dataclass(frozen=True)
    class Judge:
        """
        Configuration for a single judge attachment.
        """
        key: str
        sampling_rate: float

        def to_dict(self) -> dict:
            """
            Render the judge as a dictionary object.
            """
            return {
                'key': self.key,
                'samplingRate': self.sampling_rate,
            }

    judges: List[JudgeConfiguration.Judge]

    def to_dict(self) -> dict:
        """
        Render the judge configuration as a dictionary object.
        """
        return {
            'judges': [judge.to_dict() for judge in self.judges],
        }


# ============================================================================
# Base AI Config Types
# ============================================================================

@dataclass(frozen=True)
class AIConfigDefault:
    """
    Base AI Config interface for default implementations with optional enabled property.
    """
    enabled: Optional[bool] = None
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None

    def _base_to_dict(self) -> Dict[str, Any]:
        """
        Render the base config fields as a dictionary object.
        """
        return {
            '_ldMeta': {
                'enabled': self.enabled or False,
            },
            'model': self.model.to_dict() if self.model else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }


@dataclass(frozen=True)
class AIConfig:
    """
    Base AI Config interface without mode-specific fields.
    """
    enabled: bool
    model: Optional[ModelConfig] = None
    provider: Optional[ProviderConfig] = None
    tracker: Optional[LDAIConfigTracker] = None

    def _base_to_dict(self) -> Dict[str, Any]:
        """
        Render the base config fields as a dictionary object.
        """
        return {
            '_ldMeta': {
                'enabled': self.enabled,
            },
            'model': self.model.to_dict() if self.model else None,
            'provider': self.provider.to_dict() if self.provider else None,
        }


# ============================================================================
# Completion Config Types
# ============================================================================

@dataclass(frozen=True)
class AICompletionConfigDefault(AIConfigDefault):
    """
    Default Completion AI Config (default mode).
    """
    messages: Optional[List[LDMessage]] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> dict:
        """
        Render the given default values as an AICompletionConfigDefault-compatible dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


@dataclass(frozen=True)
class AICompletionConfig(AIConfig):
    """
    Completion AI Config (default mode).
    """
    messages: Optional[List[LDMessage]] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> dict:
        """
        Render the given completion config as a dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


# ============================================================================
# Agent Config Types
# ============================================================================

@dataclass(frozen=True)
class AIAgentConfigDefault(AIConfigDefault):
    """
    Default Agent-specific AI Config with instructions.
    """
    instructions: Optional[str] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config default as a dictionary object.
        """
        result = self._base_to_dict()
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


@dataclass(frozen=True)
class AIAgentConfig(AIConfig):
    """
    Agent-specific AI Config with instructions.
    """
    instructions: Optional[str] = None
    judge_configuration: Optional[JudgeConfiguration] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the given agent config as a dictionary object.
        """
        result = self._base_to_dict()
        if self.instructions is not None:
            result['instructions'] = self.instructions
        if self.judge_configuration is not None:
            result['judgeConfiguration'] = self.judge_configuration.to_dict()
        return result


# ============================================================================
# Judge Config Types
# ============================================================================

@dataclass(frozen=True)
class AIJudgeConfigDefault(AIConfigDefault):
    """
    Default Judge-specific AI Config with required evaluation metric key.
    """
    messages: Optional[List[LDMessage]] = None
    evaluation_metric_keys: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """
        Render the given judge config default as a dictionary object.
        """
        result = self._base_to_dict()
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        if self.evaluation_metric_keys is not None:
            result['evaluationMetricKeys'] = self.evaluation_metric_keys
        return result


@dataclass(frozen=True)
class AIJudgeConfig(AIConfig):
    """
    Judge-specific AI Config with required evaluation metric key.
    """
    evaluation_metric_keys: List[str] = field(default_factory=list)
    messages: Optional[List[LDMessage]] = None

    def to_dict(self) -> dict:
        """
        Render the given judge config as a dictionary object.
        """
        result = self._base_to_dict()
        result['evaluationMetricKeys'] = self.evaluation_metric_keys
        result['messages'] = [message.to_dict() for message in self.messages] if self.messages else None
        return result


# ============================================================================
# Agent Request Config
# ============================================================================

@dataclass
class AIAgentConfigRequest:
    """
    Configuration for a single agent request.

    Combines agent key with its specific default configuration and variables.
    """
    key: str
    default_value: AIAgentConfigDefault
    variables: Optional[Dict[str, Any]] = None


# Type alias for multiple agents
AIAgents = Dict[str, AIAgentConfig]

# Type alias for all AI Config variants
AIConfigKind = Union[AIAgentConfig, AICompletionConfig, AIJudgeConfig]


# ============================================================================
# Deprecated Type Aliases for Backward Compatibility
# ============================================================================

# Note: These are type aliases that point to the new types.
# Since Python uses duck typing, these will work at runtime even if type checkers complain.
# The old AIConfig had optional enabled, so it maps to AICompletionConfigDefault
# The old AIConfig return type had required enabled, so it maps to AICompletionConfig

# Note: AIConfig is now the base class for all config types (defined above at line 169)
# For default configs (with optional enabled), use AICompletionConfigDefault instead
# For required configs (with required enabled), use AICompletionConfig instead

# Deprecated: Use AIAgentConfigDefault instead
LDAIAgentDefaults = AIAgentConfigDefault

# Deprecated: Use AIAgentConfigRequest instead
LDAIAgentConfig = AIAgentConfigRequest

# Deprecated: Use AIAgentConfig instead (note: this was the old return type)
LDAIAgent = AIAgentConfig
