import asyncio
from typing import List, Optional

from ldai import log
from ldai.models import AICompletionConfig, LDMessage
from ldai.providers.model_runner import ModelRunner
from ldai.providers.types import JudgeResult, ModelResponse
from ldai.tracker import LDAIConfigTracker


class ManagedModel:
    """
    LaunchDarkly managed wrapper for AI model invocations.

    Holds a ModelRunner. Handles conversation management, judge evaluation
    dispatch, and tracking automatically via ``create_tracker()``.
    Obtain an instance via ``LDAIClient.create_model()``.
    """

    def __init__(
        self,
        ai_config: AICompletionConfig,
        model_runner: ModelRunner,
    ):
        self._ai_config = ai_config
        self._model_runner = model_runner
        self._messages: List[LDMessage] = []

    async def invoke(self, prompt: str) -> ModelResponse:
        """
        Invoke the model with a prompt string.

        Appends the prompt to the conversation history, prepends any
        system messages from the config, delegates to the runner, and
        appends the response to the history.

        :param prompt: The user prompt to send to the model
        :return: ModelResponse containing the model's response and metrics
        """
        tracker = self._ai_config.create_tracker()

        user_message = LDMessage(role='user', content=prompt)
        self._messages.append(user_message)

        config_messages = self._ai_config.messages or []
        all_messages = config_messages + self._messages

        response = await tracker.track_metrics_of_async(
            lambda result: result.metrics,
            lambda: self._model_runner.invoke_model(all_messages),
        )

        input_text = '\r\n'.join(m.content for m in self._messages) if self._messages else ''
        output_text = response.message.content
        response.evaluations = self._track_judge_results(tracker, input_text, output_text)

        self._messages.append(response.message)
        return response

    def _track_judge_results(
        self,
        tracker: LDAIConfigTracker,
        input_text: str,
        output_text: str,
    ) -> asyncio.Task[List[JudgeResult]]:
        evaluator_task = self._ai_config.evaluator.evaluate(input_text, output_text)

        async def _run_and_track(eval_task: asyncio.Task) -> List[JudgeResult]:
            results = await eval_task
            for r in results:
                if r.success:
                    try:
                        tracker.track_judge_result(r)
                    except Exception:
                        pass
                else:
                    log.warning("Judge evaluation failed: %s", r.error_message)
            return results

        return asyncio.create_task(_run_and_track(evaluator_task))

    def get_messages(self, include_config_messages: bool = False) -> List[LDMessage]:
        """
        Get all messages in the conversation history.

        :param include_config_messages: When True, prepends config messages.
        :return: List of conversation messages.
        """
        if include_config_messages:
            return (self._ai_config.messages or []) + self._messages
        return list(self._messages)

    def append_messages(self, messages: List[LDMessage]) -> None:
        """
        Append messages to the conversation history without invoking the model.

        :param messages: Messages to append.
        """
        self._messages.extend(messages)

    def get_model_runner(self) -> ModelRunner:
        """
        Return the underlying ModelRunner for advanced use.

        :return: The ModelRunner instance.
        """
        return self._model_runner

    def get_config(self) -> AICompletionConfig:
        """Return the AI completion config."""
        return self._ai_config
