"""Result types and type aliases for agent and agent graph runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from ldai.providers.types import LDAIMetrics

# Type alias for a registry of tools available to an agent.
# Keys are tool names; values are the callable implementations.
ToolRegistry = Dict[str, Callable]


@dataclass
class AgentResult:
    """
    Result from a single-agent run.
    """
    output: str
    raw: Any
    metrics: LDAIMetrics


@dataclass
class AgentGraphResult:
    """
    Result from an agent graph run.
    """
    output: str
    raw: Any
    metrics: LDAIMetrics
