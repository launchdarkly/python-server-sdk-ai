"""Chat implementation for managing AI chat conversations."""

import asyncio
from typing import Any, Dict, List, Optional

from ldai.judge import Judge
from ldai.models import AICompletionConfig, LDMessage
from ldai.providers.ai_provider import AIProvider
from ldai.providers.types import ChatResponse, JudgeResponse
from ldai.tracker import LDAIConfigTracker


class Chat:
    """
    Concrete implementation of Chat that provides chat functionality
    by delegating to an AIProvider implementation.

    This class handles conversation management and tracking, while delegating
    the actual model invocation to the provider.
    """

    def __init__(
        self,
        ai_config: AICompletionConfig,
        tracker: LDAIConfigTracker,
        provider: AIProvider,
        judges: Optional[Dict[str, Judge]] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the Chat.

        :param ai_config: The completion AI configuration
        :param tracker: The tracker for the completion configuration
        :param provider: The AI provider to use for chat
        :param judges: Optional dictionary of judge instances keyed by their configuration keys
        :param logger: Optional logger for logging
        """
        self._ai_config = ai_config
        self._tracker = tracker
        self._provider = provider
        self._judges = judges or {}
        self._logger = logger
        self._messages: List[LDMessage] = []

    async def invoke(self, prompt: str) -> ChatResponse:
        """
        Invoke the chat model with a prompt string.

        This method handles conversation management and tracking, delegating to the provider's invoke_model method.

        :param prompt: The user prompt to send to the chat model
        :return: ChatResponse containing the model's response and metrics
        """
        # Convert prompt string to LDMessage with role 'user' and add to conversation history
        user_message: LDMessage = LDMessage(role='user', content=prompt)
        self._messages.append(user_message)

        # Prepend config messages to conversation history for model invocation
        config_messages = self._ai_config.messages or []
        all_messages = config_messages + self._messages

        # Delegate to provider-specific implementation with tracking
        response = await self._tracker.track_metrics_of(
            lambda: self._provider.invoke_model(all_messages),
            lambda result: result.metrics,
        )

        # Start judge evaluations as async tasks (don't await them)
        if (
            self._ai_config.judge_configuration
            and self._ai_config.judge_configuration.judges
            and len(self._ai_config.judge_configuration.judges) > 0
        ):
            response.evaluations = self._start_judge_evaluations(self._messages, response)

        # Add the response message to conversation history
        self._messages.append(response.message)
        return response

    def _start_judge_evaluations(
        self,
        messages: List[LDMessage],
        response: ChatResponse,
    ) -> List[asyncio.Task[Optional[JudgeResponse]]]:
        """
        Start judge evaluations as async tasks without awaiting them.

        Returns a list of async tasks that can be awaited later.

        :param messages: Array of messages representing the conversation history
        :param response: The AI response to be evaluated
        :return: List of async tasks that will return judge evaluation results
        """
        if not self._ai_config.judge_configuration or not self._ai_config.judge_configuration.judges:
            return []

        judge_configs = self._ai_config.judge_configuration.judges

        # Start all judge evaluations as tasks
        async def evaluate_judge(judge_config):
            judge = self._judges.get(judge_config.key)
            if not judge:
                if self._logger:
                    self._logger.warn(
                        f"Judge configuration is not enabled: {judge_config.key}",
                    )
                return None

            eval_result = await judge.evaluate_messages(
                messages, response, judge_config.sampling_rate
            )

            if eval_result and eval_result.success:
                self._tracker.track_judge_response(eval_result)

            return eval_result

        # Create tasks for each judge evaluation
        tasks = [
            asyncio.create_task(evaluate_judge(judge_config))
            for judge_config in judge_configs
        ]

        return tasks

    def get_config(self) -> AICompletionConfig:
        """
        Get the underlying AI configuration used to initialize this Chat.

        :return: The AI completion configuration
        """
        return self._ai_config

    def get_tracker(self) -> LDAIConfigTracker:
        """
        Get the underlying AI configuration tracker used to initialize this Chat.

        :return: The tracker instance
        """
        return self._tracker

    def get_provider(self) -> AIProvider:
        """
        Get the underlying AI provider instance.

        This provides direct access to the provider for advanced use cases.

        :return: The AI provider instance
        """
        return self._provider

    def get_judges(self) -> Dict[str, Judge]:
        """
        Get the judges associated with this Chat.

        Returns a dictionary of judge instances keyed by their configuration keys.

        :return: Dictionary of judge instances
        """
        return self._judges

    def append_messages(self, messages: List[LDMessage]) -> None:
        """
        Append messages to the conversation history.

        Adds messages to the conversation history without invoking the model,
        which is useful for managing multi-turn conversations or injecting context.

        :param messages: Array of messages to append to the conversation history
        """
        self._messages.extend(messages)

    def get_messages(self, include_config_messages: bool = False) -> List[LDMessage]:
        """
        Get all messages in the conversation history.

        :param include_config_messages: Whether to include the config messages from the AIConfig.
                                       Defaults to False.
        :return: Array of messages. When include_config_messages is True, returns both config
                messages and conversation history with config messages prepended. When False,
                returns only the conversation history messages.
        """
        if include_config_messages:
            config_messages = self._ai_config.messages or []
            return config_messages + self._messages
        return list(self._messages)
