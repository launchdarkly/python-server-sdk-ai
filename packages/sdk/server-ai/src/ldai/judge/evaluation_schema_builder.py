"""Internal class for building dynamic evaluation response schemas."""

from typing import Any, Dict, Optional


class EvaluationSchemaBuilder:
    """
    Internal class for building dynamic evaluation response schemas.
    Not exported - only used internally by Judge.
    """

    @staticmethod
    def build(evaluation_metric_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Build an evaluation response schema from evaluation metric key.

        :param evaluation_metric_key: Evaluation metric key, or None if not available
        :return: Schema dictionary for structured output, or None if evaluation_metric_key is None
        """
        if not evaluation_metric_key:
            return None

        return {
            'title': 'EvaluationResponse',
            'description': f"Response containing evaluation results for {evaluation_metric_key} metric",
            'type': 'object',
            'properties': {
                'evaluations': {
                    'type': 'object',
                    'description': (
                        f"Object containing evaluation results for "
                        f"{evaluation_metric_key} metric"
                    ),
                    'properties': EvaluationSchemaBuilder._build_key_properties(evaluation_metric_key),
                    'required': [evaluation_metric_key],
                    'additionalProperties': False,
                },
            },
            'required': ['evaluations'],
            'additionalProperties': False,
        }

    @staticmethod
    def _build_key_properties(evaluation_metric_key: str) -> Dict[str, Any]:
        """
        Build properties for a single evaluation metric key.

        :param evaluation_metric_key: Evaluation metric key
        :return: Dictionary of properties for the key
        """
        return {
            evaluation_metric_key: EvaluationSchemaBuilder._build_key_schema(evaluation_metric_key)
        }

    @staticmethod
    def _build_key_schema(key: str) -> Dict[str, Any]:
        """
        Build schema for a single evaluation metric key.

        :param key: Evaluation metric key
        :return: Schema dictionary for the key
        """
        return {
            'type': 'object',
            'properties': {
                'score': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                    'description': f'Score between 0.0 and 1.0 for {key}',
                },
                'reasoning': {
                    'type': 'string',
                    'description': f'Reasoning behind the score for {key}',
                },
            },
            'required': ['score', 'reasoning'],
            'additionalProperties': False,
        }
