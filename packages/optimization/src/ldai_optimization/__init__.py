"""LaunchDarkly AI SDK — optimization.

This package will provide helpers to run selected tools against the LaunchDarkly API from SDK-based workflows.
"""

from ldai_optimization.client import OptimizationClient
from ldai_optimization.dataclasses import (
    AIJudgeCallConfig,
    OptimizationContext,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
    ToolDefinition,
)

__version__ = "0.0.0"

__all__ = [
    '__version__',
    'AIJudgeCallConfig',
    'OptimizationClient',
    'OptimizationContext',
    'OptimizationJudge',
    'OptimizationJudgeContext',
    'OptimizationOptions',
    'ToolDefinition',
]
