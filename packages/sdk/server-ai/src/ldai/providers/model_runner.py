from typing import List, Protocol, runtime_checkable

from ldai.models import LDMessage
from ldai.providers.types import RunnerResult


@runtime_checkable
class ModelRunner(Protocol):
    """
    Runtime capability interface for model invocation.

    A ModelRunner is a focused, configured object returned by
    AIProvider.create_model(). It knows exactly which model to call
    and with what parameters — the caller just passes messages.
    """

    async def invoke_model(self, messages: List[LDMessage]) -> RunnerResult:
        """
        Invoke the model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: RunnerResult containing the model's response and metrics
        """
        ...
