"""TrackedChat implementation for managing AI chat conversations."""

from typing import Any, Dict, List, Optional

from ldai.models import AICompletionConfig, LDMessage
from ldai.providers.ai_provider import AIProvider
from ldai.providers.types import ChatResponse, JudgeResponse
from ldai.judge import AIJudge
from ldai.tracker import LDAIConfigTracker


class TrackedChat:
    """
    Concrete implementation of TrackedChat that provides chat functionality
    by delegating to an AIProvider implementation.
    
    This class handles conversation management and tracking, while delegating
    the actual model invocation to the provider.
    """

    def __init__(
        self,
        ai_config: AICompletionConfig,
        tracker: LDAIConfigTracker,
        provider: AIProvider,
        judges: Optional[Dict[str, AIJudge]] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the TrackedChat.
        
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
            lambda result: result.metrics,
            lambda: self._provider.invoke_model(all_messages),
        )

        # Evaluate with judges if configured
        if (
            self._ai_config.judge_configuration
            and self._ai_config.judge_configuration.judges
            and len(self._ai_config.judge_configuration.judges) > 0
        ):
            evaluations = await self._evaluate_with_judges(self._messages, response)
            response.evaluations = evaluations

        # Add the response message to conversation history
        self._messages.append(response.message)
        return response

    async def _evaluate_with_judges(
        self,
        messages: List[LDMessage],
        response: ChatResponse,
    ) -> List[Optional[JudgeResponse]]:
        """
        Evaluates the response with all configured judges.
        
        Returns a list of evaluation results.
        
        :param messages: Array of messages representing the conversation history
        :param response: The AI response to be evaluated
        :return: List of judge evaluation results (may contain None for failed evaluations)
        """
        if not self._ai_config.judge_configuration or not self._ai_config.judge_configuration.judges:
            return []

        judge_configs = self._ai_config.judge_configuration.judges

        # Start all judge evaluations in parallel
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
                self._tracker.track_eval_scores(eval_result.evals)

            return eval_result

        # Ensure all evaluations complete even if some fail
        import asyncio
        evaluation_promises = [evaluate_judge(judge_config) for judge_config in judge_configs]
        results = await asyncio.gather(*evaluation_promises, return_exceptions=True)
        
        # Map exceptions to None
        return [
            None if isinstance(result, Exception) else result
            for result in results
        ]

    def get_config(self) -> AICompletionConfig:
        """
        Get the underlying AI configuration used to initialize this TrackedChat.
        
        :return: The AI completion configuration
        """
        return self._ai_config

    def get_tracker(self) -> LDAIConfigTracker:
        """
        Get the underlying AI configuration tracker used to initialize this TrackedChat.
        
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

    def get_judges(self) -> Dict[str, AIJudge]:
        """
        Get the judges associated with this TrackedChat.
        
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

