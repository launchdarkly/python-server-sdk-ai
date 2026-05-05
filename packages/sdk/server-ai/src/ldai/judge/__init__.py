"""Judge implementation for AI evaluation."""

import random
from typing import Any, Dict, List, Optional, Tuple

from ldai import log
from ldai.judge.evaluation_schema_builder import EvaluationSchemaBuilder
from ldai.models import AIJudgeConfig, LDMessage
from ldai.providers.runner import Runner
from ldai.providers.types import JudgeResult, RunnerResult


def _strip_legacy_judge_messages(messages: List[LDMessage]) -> List[LDMessage]:
    """
    Remove legacy judge template messages from a message list.

    Strips any non-system message whose content contains ``{{message_history}}``
    or ``{{response_to_evaluate}}``.  These were used by older judge configs to
    indicate where the SDK should interpolate the evaluated conversation; new
    configs omit them entirely and rely on the string input built by
    :meth:`Judge._build_evaluation_input`.

    :param messages: The raw message list from the judge AI config.
    :return: A new list with legacy template messages removed.
    """
    result = []
    for msg in messages:
        if msg.role != 'system' and (
            '{{message_history}}' in msg.content
            or '{{response_to_evaluate}}' in msg.content
        ):
            continue
        result.append(msg)
    return result


class Judge:
    """
    Judge implementation that handles evaluation functionality and conversation management.

    According to the AIEval spec, judges are AI Configs with mode: "judge" that evaluate
    other AI Configs using structured output.
    """

    def __init__(
        self,
        ai_config: AIJudgeConfig,
        model_runner: Runner,
        sample_rate: float = 1.0,
    ):
        """
        Initialize the Judge.

        :param ai_config: The judge AI configuration
        :param model_runner: The model runner to use for evaluation
        :param sample_rate: Default sampling rate (0-1) used when ``evaluate``
            is called without an explicit ``sampling_rate`` (defaults to 1).
        """
        self._ai_config = ai_config
        self._model_runner = model_runner
        self.sample_rate = sample_rate
        self._evaluation_response_structure = EvaluationSchemaBuilder.build()

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        sampling_rate: Optional[float] = None,
    ) -> JudgeResult:
        """
        Evaluates an AI response using the judge's configuration.

        :param input_text: The input prompt or question that was provided to the AI
        :param output_text: The AI-generated response to be evaluated
        :param sampling_rate: Sampling rate (0-1) to determine if evaluation should be processed.
            When ``None`` (the default), falls back to ``self.sample_rate``.
        :return: The result of the judge evaluation.
        """
        effective_rate = sampling_rate if sampling_rate is not None else self.sample_rate
        judge_result = JudgeResult(judge_config_key=self._ai_config.key)

        try:
            if not self._ai_config.evaluation_metric_key:
                log.warning(
                    'Judge configuration is missing required evaluationMetricKey'
                )
                judge_result.error_message = 'Judge configuration is missing required evaluationMetricKey'
                return judge_result

            if random.random() > effective_rate:
                log.debug(f'Judge evaluation skipped due to sampling rate: {effective_rate}')
                return judge_result

            judge_result.sampled = True

            tracker = self._ai_config.create_tracker()
            evaluation_input = self._build_evaluation_input(input_text, output_text)
            assert self._evaluation_response_structure is not None

            response = await tracker.track_metrics_of_async(
                lambda result: result.metrics,
                lambda: self._model_runner.run(evaluation_input, output_type=self._evaluation_response_structure),
            )

            if response.parsed is None:
                log.warning('Judge evaluation did not return structured output')
                return judge_result

            parsed = self._parse_evaluation_response(response.parsed)

            if parsed is None:
                log.warning('Judge evaluation did not return the expected evaluation')
                return judge_result

            score, reasoning = parsed
            judge_result.metric_key = self._ai_config.evaluation_metric_key
            judge_result.score = score
            judge_result.reasoning = reasoning
            judge_result.success = response.metrics.success
            return judge_result
        except Exception as error:
            log.error(f'Judge evaluation failed: {error}')
            judge_result.error_message = str(error) if isinstance(error, Exception) else 'Unknown error'
            return judge_result

    async def evaluate_messages(
        self,
        messages: list[LDMessage],
        response: RunnerResult,
        sampling_ratio: Optional[float] = None,
    ) -> JudgeResult:
        """
        Evaluates an AI response from chat messages and response.

        :param messages: Array of messages representing the conversation history
        :param response: The runner result to be evaluated
        :param sampling_ratio: Sampling ratio (0-1) to determine if evaluation should be processed.
            When ``None`` (the default), falls back to ``self.sample_rate``.
        :return: The result of the judge evaluation.
        """
        input_text = '\r\n'.join([msg.content for msg in messages]) if messages else ''
        output_text = response.content

        return await self.evaluate(input_text, output_text, sampling_ratio)

    def get_ai_config(self) -> AIJudgeConfig:
        """
        Returns the AI Config used by this judge.

        :return: The judge AI configuration
        """
        return self._ai_config

    def get_model_runner(self) -> Runner:
        """
        Returns the model runner used by this judge.

        :return: The model runner
        """
        return self._model_runner

    def _build_evaluation_input(self, input_text: str, output_text: str) -> str:
        return f"MESSAGE HISTORY:\n{input_text}\n\nRESPONSE TO EVALUATE:\n{output_text}"

    def _parse_evaluation_response(self, data: Dict[str, Any]) -> Optional[Tuple[float, str]]:
        """
        Parses the structured evaluation response. Expects {"score": n, "reasoning": "..."}.

        :return: ``(score, reasoning)`` on success, or ``None`` if the response is invalid.
        """
        if not isinstance(data, dict):
            log.warning('Invalid response: missing or invalid evaluation')
            return None

        score = data.get('score')
        reasoning = data.get('reasoning')
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            log.warning(f'Invalid score: {score}. Score must be a number between 0 and 1 inclusive')
            return None
        if not isinstance(reasoning, str):
            log.warning('Invalid reasoning: must be a string')
            return None

        return (float(score), reasoning)
