"""LaunchDarkly AI SDK — optimization.

This package will provide helpers to run selected tools against the LaunchDarkly API from SDK-based workflows.
"""

from ldai_optimization.client import ApiAgentOptimizationClient

__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    '__version__',
    'ApiAgentOptimizationClient',
]
