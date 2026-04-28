import asyncio
import warnings
from typing import List, Union

from ldai.models import AICompletionConfig, LDMessage
from ldai.providers.model_runner import ModelRunner
from ldai.providers.runner import Runner
from ldai.providers.types import JudgeResult, ManagedResult, ModelResponse, RunnerResult
from ldai.tracker import LDAIConfigTracker


class ManagedModel:
    """
    LaunchDarkly managed wrapper for AI model invocations.

    Holds a Runner (or legacy ModelRunner). Handles conversation management,
    judge evaluation dispatch, and tracking automatically via ``create_tracker()``.
    Obtain an instance via ``LDAIClient.create_model()``.
    """

    def __init__(
        self,
        ai_config: AICompletionConfig,
        model_runner: Union[Runner, ModelRunner],
    ):
        self._ai_config = ai_config
        self._model_runner = model_runner
        self._messages: List[LDMessage] = []

    async def run(self, prompt: str) -> ManagedResult:
        """
        Run the model with a prompt string.

        Appends the prompt to the conversation history, prepends any
        system messages from the config, delegates to the runner, and
        appends the response to the history.

        :param prompt: The user prompt to send to the model
        :return: ManagedResult containing the model's response, metric summary,
            and an optional evaluations task
        """
        tracker = self._ai_config.create_tracker()

        user_message = LDMessage(role='user', content=prompt)
        self._messages.append(user_message)

        config_messages = self._ai_config.messages or []
        all_messages = config_messages + self._messages

        result: Union[RunnerResult, ModelResponse] = await tracker.track_metrics_of_async(
            lambda r: r.metrics,
            lambda: self._invoke_runner(all_messages),
        )

        # Support both new RunnerResult and legacy ModelResponse
        if isinstance(result, RunnerResult):
            content = result.content
            raw = result.raw
            parsed = result.parsed
            assistant_message = LDMessage(role='assistant', content=content)
        else:
            content = result.message.content
            raw = getattr(result, 'raw', None)
            parsed = getattr(result, 'parsed', None)
            assistant_message = result.message

        input_text = '\r\n'.join(m.content for m in self._messages) if self._messages else ''

        evaluations_task = self._track_judge_results(tracker, input_text, content)

        self._messages.append(assistant_message)

        return ManagedResult(
            content=content,
            metrics=tracker.get_summary(),
            raw=raw,
            parsed=parsed,
            evaluations=evaluations_task,
        )

    async def _invoke_runner(
        self, all_messages: List[LDMessage]
    ) -> Union[RunnerResult, ModelResponse]:
        """
        Delegate to the runner.  Supports both the new ``Runner`` protocol
        (``run(messages) → RunnerResult``) and the legacy ``ModelRunner``
        (``invoke_model(messages) → ModelResponse``).
        """
        if isinstance(self._model_runner, Runner):
            return await self._model_runner.run(all_messages)
        # Legacy ModelRunner path
        return await self._model_runner.invoke_model(all_messages)  # type: ignore[union-attr]

    async def invoke(self, prompt: str) -> ModelResponse:
        """
        Invoke the model with a prompt string.

        .. deprecated::
            Use :meth:`run` instead. This method will be removed in a future
            release once the migration to :class:`ManagedResult` is complete.

        :param prompt: The user prompt to send to the model
        :return: ModelResponse containing the model's response and metrics
        """
        warnings.warn(
            "ManagedModel.invoke() is deprecated. Use run() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        tracker = self._ai_config.create_tracker()

        user_message = LDMessage(role='user', content=prompt)
        self._messages.append(user_message)

        config_messages = self._ai_config.messages or []
        all_messages = config_messages + self._messages

        response: ModelResponse = await tracker.track_metrics_of_async(
            lambda result: result.metrics,
            lambda: self._model_runner.invoke_model(all_messages),  # type: ignore[union-attr]
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
                    tracker.track_judge_result(r)
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

    def get_model_runner(self) -> Union[Runner, ModelRunner]:
        """
        Return the underlying runner for advanced use.

        :return: The Runner or legacy ModelRunner instance.
        """
        return self._model_runner

    def get_config(self) -> AICompletionConfig:
        """Return the AI completion config."""
        return self._ai_config
