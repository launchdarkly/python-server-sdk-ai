"""Internal class for building dynamic evaluation response schemas."""

from typing import Any, Dict


class EvaluationSchemaBuilder:
    """
    Internal class for building dynamic evaluation response schemas.
    Not exported - only used internally by Judge.
    """

    @staticmethod
    def build(evaluation_metric_keys: list[str]) -> Dict[str, Any]:
        """
        Build an evaluation response schema from evaluation metric keys.

        :param evaluation_metric_keys: List of evaluation metric keys
        :return: Schema dictionary for structured output
        """
        return {
            'title': 'EvaluationResponse',
            'description': f"Response containing evaluation results for {', '.join(evaluation_metric_keys)} metrics",
            'type': 'object',
            'properties': {
                'evaluations': {
                    'type': 'object',
                    'description': (
                        f"Object containing evaluation results for "
                        f"{', '.join(evaluation_metric_keys)} metrics"
                    ),
                    'properties': EvaluationSchemaBuilder._build_key_properties(evaluation_metric_keys),
                    'required': evaluation_metric_keys,
                    'additionalProperties': False,
                },
            },
            'required': ['evaluations'],
            'additionalProperties': False,
        }

    @staticmethod
    def _build_key_properties(evaluation_metric_keys: list[str]) -> Dict[str, Any]:
        """
        Build properties for each evaluation metric key.

        :param evaluation_metric_keys: List of evaluation metric keys
        :return: Dictionary of properties for each key
        """
        result: Dict[str, Any] = {}
        for key in evaluation_metric_keys:
            result[key] = EvaluationSchemaBuilder._build_key_schema(key)
        return result

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
