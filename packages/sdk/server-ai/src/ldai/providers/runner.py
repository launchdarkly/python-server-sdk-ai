"""Unified Runner protocol for AI providers."""

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ldai.providers.types import RunnerResult


@runtime_checkable
class Runner(Protocol):
    """
    Unified runtime capability interface for all AI provider runners.

    A :class:`Runner` is a focused, configured object that performs a single
    AI invocation.
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
