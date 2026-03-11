"""Abstract base class for agent runners."""

from abc import ABC, abstractmethod
from typing import Any

from ldai.runners.types import AgentResult


class AgentRunner(ABC):
    """
    Abstract base class for single-agent runners.

    An AgentRunner encapsulates the execution of a single AI agent.
    Provider-specific implementations (e.g. OpenAIAgentRunner) are returned
    by RunnerFactory.create_agent() and hold all provider wiring internally.
    """

    @abstractmethod
    async def run(self, input: Any) -> AgentResult:
        """
        Run the agent with the given input.

        :param input: The input to the agent (string prompt or structured input)
        :return: AgentResult containing the output, raw response, and metrics
        """
