"""Abstract base class for AI providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ldai.models import AIConfigKind, LDMessage
from ldai.providers.types import ChatResponse, StructuredResponse


class AIProvider(ABC):
    """
    Abstract base class for AI providers that implement chat model functionality.

    This class provides the contract that all provider implementations must follow
    to integrate with LaunchDarkly's tracking and configuration capabilities.

    Following the AICHAT spec recommendation to use base classes with non-abstract methods
    for better extensibility and backwards compatibility.
    """

    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize the AI provider.

        :param logger: Optional logger for logging provider operations.
        """
        self.logger = logger

    async def invoke_model(self, messages: List[LDMessage]) -> ChatResponse:
        """
        Invoke the chat model with an array of messages.

        This method should convert messages to provider format, invoke the model,
        and return a ChatResponse with the result and metrics.

        Default implementation takes no action and returns a placeholder response.
        Provider implementations should override this method.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ChatResponse containing the model's response
        """
        if self.logger:
            self.logger.warn('invokeModel not implemented by this provider')

        from ldai.models import LDMessage
        from ldai.providers.types import LDAIMetrics

        return ChatResponse(
            message=LDMessage(role='assistant', content=''),
            metrics=LDAIMetrics(success=False, usage=None),
        )

    async def invoke_structured_model(
        self,
        messages: List[LDMessage],
        response_structure: Dict[str, Any],
    ) -> StructuredResponse:
        """
        Invoke the chat model with structured output support.

        This method should convert messages to provider format, invoke the model with
        structured output configuration, and return a structured response.

        Default implementation takes no action and returns a placeholder response.
        Provider implementations should override this method.

        :param messages: Array of LDMessage objects representing the conversation
        :param response_structure: Dictionary of output configurations keyed by output name
        :return: StructuredResponse containing the structured data
        """
        if self.logger:
            self.logger.warn('invokeStructuredModel not implemented by this provider')

        from ldai.providers.types import LDAIMetrics

        return StructuredResponse(
            data={},
            raw_response='',
            metrics=LDAIMetrics(success=False, usage=None),
        )

    @staticmethod
    @abstractmethod
    async def create(ai_config: AIConfigKind, logger: Optional[Any] = None) -> 'AIProvider':
        """
        Static method that constructs an instance of the provider.

        Each provider implementation must provide their own static create method
        that accepts an AIConfigKind and returns a configured instance.

        :param ai_config: The LaunchDarkly AI configuration
        :param logger: Optional logger for the provider
        :return: Configured provider instance
        """
        raise NotImplementedError('Provider implementations must override the static create method')
