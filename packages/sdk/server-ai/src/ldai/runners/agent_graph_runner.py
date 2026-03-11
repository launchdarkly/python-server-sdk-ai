"""Abstract base class for agent graph runners."""

from abc import ABC, abstractmethod
from typing import Any

from ldai.runners.types import AgentGraphResult


class AgentGraphRunner(ABC):
    """
    Abstract base class for agent graph runners.

    An AgentGraphRunner encapsulates multi-agent graph execution.
    Provider-specific implementations (e.g. OpenAIAgentGraphRunner) are
    returned by RunnerFactory.create_agent_graph() and hold all provider
    wiring internally.
    """

    @abstractmethod
    async def run(self, input: Any) -> AgentGraphResult:
        """
        Run the agent graph with the given input.

        :param input: The input to the agent graph (string prompt or structured input)
        :return: AgentGraphResult containing the output, raw response, and metrics
        """
