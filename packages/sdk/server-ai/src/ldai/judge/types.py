"""Types for judge evaluation responses."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EvalScore:
    """
    Score and reasoning for a single evaluation metric.
    """
    score: float  # Score between 0.0 and 1.0
    reasoning: str  # Reasoning behind the provided score

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the evaluation score as a dictionary object.
        """
        return {
            'score': self.score,
            'reasoning': self.reasoning,
        }


@dataclass
class JudgeResponse:
    """
    Response from a judge evaluation containing scores and reasoning for multiple metrics.
    """
    evals: Dict[str, EvalScore]  # Dictionary where keys are metric names and values contain score and reasoning
    success: bool  # Whether the evaluation completed successfully
    error: Optional[str] = None  # Error message if evaluation failed

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the judge response as a dictionary object.
        """
        result: Dict[str, Any] = {
            'evals': {key: eval_score.to_dict() for key, eval_score in self.evals.items()},
            'success': self.success,
        }
        if self.error is not None:
            result['error'] = self.error
        return result
