"""Judge implementation for AI evaluation."""

import random
from typing import Any, Dict, Optional

import chevron

from ldai.judge.evaluation_schema_builder import EvaluationSchemaBuilder
from ldai.models import AIJudgeConfig, LDMessage
from ldai.providers.ai_provider import AIProvider
from ldai.providers.types import (ChatResponse, EvalScore, JudgeResponse,
                                  StructuredResponse)
from ldai.tracker import LDAIConfigTracker


class Judge:
    """
    Judge implementation that handles evaluation functionality and conversation management.

    According to the AIEval spec, judges are AI Configs with mode: "judge" that evaluate
    other AI Configs using structured output.
    """

    def __init__(
        self,
        ai_config: AIJudgeConfig,
        ai_config_tracker: LDAIConfigTracker,
        ai_provider: AIProvider,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the Judge.

        :param ai_config: The judge AI configuration
        :param ai_config_tracker: The tracker for the judge configuration
        :param ai_provider: The AI provider to use for evaluation
        :param logger: Optional logger for logging
        """
        self._ai_config = ai_config
        self._ai_config_tracker = ai_config_tracker
        self._ai_provider = ai_provider
        self._logger = logger
        self._evaluation_response_structure = EvaluationSchemaBuilder.build(
            ai_config.evaluation_metric_keys
        )

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        sampling_rate: float = 1.0,
    ) -> Optional[JudgeResponse]:
        """
        Evaluates an AI response using the judge's configuration.

        :param input_text: The input prompt or question that was provided to the AI
        :param output_text: The AI-generated response to be evaluated
        :param sampling_rate: Sampling rate (0-1) to determine if evaluation should be processed (defaults to 1)
        :return: Evaluation results or None if not sampled
        """
        try:
            if not self._ai_config.evaluation_metric_keys or len(self._ai_config.evaluation_metric_keys) == 0:
                if self._logger:
                    self._logger.warn(
                        'Judge configuration is missing required evaluationMetricKeys'
                    )
                return None

            if not self._ai_config.messages:
                if self._logger:
                    self._logger.warn('Judge configuration must include messages')
                return None

            if random.random() > sampling_rate:
                if self._logger:
                    self._logger.debug(f'Judge evaluation skipped due to sampling rate: {sampling_rate}')
                return None

            messages = self._construct_evaluation_messages(input_text, output_text)

            # Track metrics of the structured model invocation
            response = await self._ai_config_tracker.track_metrics_of(
                lambda: self._ai_provider.invoke_structured_model(messages, self._evaluation_response_structure),
                lambda result: result.metrics,
            )

            success = response.metrics.success

            evals = self._parse_evaluation_response(response.data)

            if len(evals) != len(self._ai_config.evaluation_metric_keys):
                if self._logger:
                    self._logger.warn('Judge evaluation did not return all evaluations')
                success = False

            return JudgeResponse(
                judge_config_key=self._ai_config.key,
                evals=evals,
                success=success,
            )
        except Exception as error:
            if self._logger:
                self._logger.error(f'Judge evaluation failed: {error}')
            return JudgeResponse(
                evals={},
                success=False,
                error=str(error) if isinstance(error, Exception) else 'Unknown error',
            )

    async def evaluate_messages(
        self,
        messages: list[LDMessage],
        response: ChatResponse,
        sampling_ratio: float = 1.0,
    ) -> Optional[JudgeResponse]:
        """
        Evaluates an AI response from chat messages and response.

        :param messages: Array of messages representing the conversation history
        :param response: The AI response to be evaluated
        :param sampling_ratio: Sampling ratio (0-1) to determine if evaluation should be processed (defaults to 1)
        :return: Evaluation results or None if not sampled
        """
        input_text = '\r\n'.join([msg.content for msg in messages]) if messages else ''
        output_text = response.message.content

        return await self.evaluate(input_text, output_text, sampling_ratio)

    def get_ai_config(self) -> AIJudgeConfig:
        """
        Returns the AI Config used by this judge.

        :return: The judge AI configuration
        """
        return self._ai_config

    def get_tracker(self) -> LDAIConfigTracker:
        """
        Returns the tracker associated with this judge.

        :return: The tracker for the judge configuration
        """
        return self._ai_config_tracker

    def get_provider(self) -> AIProvider:
        """
        Returns the AI provider used by this judge.

        :return: The AI provider
        """
        return self._ai_provider

    def _construct_evaluation_messages(self, input_text: str, output_text: str) -> list[LDMessage]:
        """
        Constructs evaluation messages by combining judge's config messages with input/output.

        :param input_text: The input text
        :param output_text: The output text to evaluate
        :return: List of messages for evaluation
        """
        if not self._ai_config.messages:
            return []

        messages: list[LDMessage] = []
        for msg in self._ai_config.messages:
            # Interpolate message content with reserved variables
            content = self._interpolate_message(msg.content, {
                'message_history': input_text,
                'response_to_evaluate': output_text,
            })
            messages.append(LDMessage(role=msg.role, content=content))

        return messages

    def _interpolate_message(self, content: str, variables: Dict[str, str]) -> str:
        """
        Interpolates message content with variables using Mustache templating.

        :param content: The message content template
        :param variables: Variables to interpolate
        :return: Interpolated message content
        """
        # Use chevron (Mustache) for templating, with no escaping
        return chevron.render(content, variables)

    def _parse_evaluation_response(self, data: Dict[str, Any]) -> Dict[str, EvalScore]:
        """
        Parses the structured evaluation response from the AI provider.

        :param data: The structured response data
        :return: Dictionary of evaluation scores keyed by metric key
        """
        results: Dict[str, EvalScore] = {}

        if not data.get('evaluations') or not isinstance(data['evaluations'], dict):
            if self._logger:
                self._logger.warn('Invalid response: missing or invalid evaluations object')
            return results

        evaluations = data['evaluations']

        for metric_key in self._ai_config.evaluation_metric_keys:
            evaluation = evaluations.get(metric_key)

            if not evaluation or not isinstance(evaluation, dict):
                if self._logger:
                    self._logger.warn(f'Missing evaluation for metric key: {metric_key}')
                continue

            score = evaluation.get('score')
            reasoning = evaluation.get('reasoning')

            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                if self._logger:
                    self._logger.warn(
                        f'Invalid score evaluated for {metric_key}: {score}. '
                        'Score must be a number between 0 and 1 inclusive'
                    )
                continue

            if not isinstance(reasoning, str):
                if self._logger:
                    self._logger.warn(
                        f'Invalid reasoning evaluated for {metric_key}: {reasoning}. '
                        'Reasoning must be a string'
                    )
                continue

            results[metric_key] = EvalScore(score=float(score), reasoning=reasoning)

        return results
