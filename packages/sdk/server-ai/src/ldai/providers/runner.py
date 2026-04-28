"""Unified Runner protocol for AI providers."""

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ldai.providers.types import RunnerResult


@runtime_checkable
class Runner(Protocol):
    """
    Unified runtime capability interface for all AI provider runners.

    A :class:`Runner` is a focused, configured object that performs a single
    AI invocation.  Both model runners and agent runners implement this protocol.

    :param input: The input to the runner (string prompt, list of messages, or
        other provider-specific input type).
    :param output_type: Optional JSON schema dict that requests structured output.
        When provided, the runner populates :attr:`~RunnerResult.parsed` on the
        returned :class:`RunnerResult`.
    :return: :class:`RunnerResult` containing ``content``, ``metrics``, and
        optionally ``raw`` and ``parsed``.
    """

    async def run(
        self,
        input: Any,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Execute the runner with the given input.

        :param input: The input to the runner.
        :param output_type: Optional JSON schema for structured output.
        :return: RunnerResult containing content, metrics, raw, and parsed fields.
        """
        ...
