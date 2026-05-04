"""LaunchDarkly AI SDK — optimization.

This package will provide helpers to run selected tools against the LaunchDarkly API from SDK-based workflows.
"""

from ldai.tracker import TokenUsage

from ldai_optimizer.client import OptimizationClient
from ldai_optimizer.dataclasses import (
    AIJudgeCallConfig,
    GroundTruthOptimizationOptions,
    GroundTruthSample,
    LLMCallConfig,
    LLMCallContext,
    OptimizationContext,
    OptimizationFromConfigOptions,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
    OptimizationResponse,
    ToolDefinition,
)
from ldai_optimizer.ld_api_client import LDApiError

__version__ = "0.0.0"

__all__ = [
    '__version__',
    'AIJudgeCallConfig',
    'GroundTruthOptimizationOptions',
    'GroundTruthSample',
    'LDApiError',
    'LLMCallConfig',
    'LLMCallContext',
    'OptimizationClient',
    'OptimizationContext',
    'OptimizationFromConfigOptions',
    'OptimizationJudge',
    'OptimizationJudgeContext',
    'OptimizationOptions',
    'OptimizationResponse',
    'TokenUsage',
    'ToolDefinition',
]
