"""Dataclasses for the LaunchDarkly AI optimization package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from ldclient import Context


@dataclass
class JudgeResult:
    """Result from a judge evaluation."""

    score: float
    rationale: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the judge result to a JSON-serializable dictionary.

        :return: Dictionary representation of the judge result that can be serialized with json.dumps()
        """
        return {
            "score": self.score,
            "rationale": self.rationale,
        }


@dataclass
class StructuredOutputTool:
    """
    Generic tool definition for enforcing structured output from LLM responses.

    This tool can be used with any LLM provider to ensure responses conform to
    a specific JSON schema. The tool takes the LLM's response and returns
    parsed and validated data according to the input_schema.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON schema defining the expected output structure

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool definition to a dictionary format compatible with LLM APIs.

        :return: Dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class Message:
    """A message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class OptimizationJudge:
    threshold: float
    judge_key: Optional[str] = None
    acceptance_statement: Optional[str] = None


@dataclass
class AutoCommitConfig:
    """Configuration for auto-committing optimization results to LaunchDarkly."""

    enabled: bool = False
    project_key: Optional[str] = None


@dataclass
class OptimizationContext:
    """Context for a single optimization iteration."""

    scores: Dict[str, JudgeResult]  # the scores and rationales from the judges, if configured
    completion_response: str
    current_instructions: str
    current_parameters: Dict[str, Any]
    current_model: Optional[str] = None  # the current model being used
    user_input: Optional[str] = None  # the user input message for this iteration
    history: Sequence[OptimizationContext] = field(
        default_factory=list
    )  # previous context items
    iteration: int = 0  # current iteration number
    structured_output_tool: Optional[StructuredOutputTool] = (
        None  # tool definition for structured output
    )

    def copy_without_history(self) -> OptimizationContext:
        """
        Create a copy of this context without the history field (for flattening).

        :return: A new OptimizeContext with the same data but empty history
        """
        return OptimizationContext(
            scores=self.scores,
            completion_response=self.completion_response,
            current_instructions=self.current_instructions,
            current_parameters=self.current_parameters,
            current_model=self.current_model,
            user_input=self.user_input,
            history=(),  # Empty history to keep it flat
            iteration=self.iteration,
            structured_output_tool=self.structured_output_tool,
        )

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the optimization context to a JSON-serializable dictionary.

        :return: Dictionary representation of the context that can be serialized with json.dumps()
        """
        scores_dict = {}
        for judge_key, judge_result in self.scores.items():
            scores_dict[judge_key] = judge_result.to_json()

        structured_output_tool_dict = None
        if self.structured_output_tool:
            structured_output_tool_dict = self.structured_output_tool.to_dict()

        history_list = [ctx.to_json() for ctx in self.history]

        return {
            "scores": scores_dict,
            "completion_response": self.completion_response,
            "current_instructions": self.current_instructions,
            "current_parameters": self.current_parameters,
            "current_model": self.current_model,
            "user_input": self.user_input,
            "history": history_list,
            "iteration": self.iteration,
            "structured_output_tool": structured_output_tool_dict,
        }


@dataclass
class OptimizationJudgeContext:
    """Context for judge evaluation."""

    messages: List[Message]
    parameters: Dict[str, Any]
    tools: Optional[List[Dict[str, Any]]] = None


@dataclass
class OptimizationOptions:
    """Options for agent optimization."""

    # Required
    context_choices: List[Context]  # choices of contexts to be used, 1 min required
    # Configuration
    max_attempts: int
    model_choices: List[str]  # model ids the LLM can choose from, 1 min required
    judge_model: str  # which model to use as judge; this should remain consistent
    variable_choices: List[
        Dict[str, Any]
    ]  # choices of interpolated variables to be chosen at random per turn, 1 min required
    # Actual agent/completion (judge) calls - Required
    handle_agent_call: Union[
        Callable[[str, OptimizationContext], str],
        Callable[[str, OptimizationContext], Awaitable[str]],
    ]
    handle_judge_call: Union[
        Callable[[str, OptimizationContext], str],
        Callable[[str, OptimizationJudgeContext], Awaitable[str]],
    ]
    # Criteria for pass/fail - Optional
    user_input_options: Optional[List[str]] = (
        None  # optional list of user input messages to randomly select from
    )
    judges: Optional[Dict[str, OptimizationJudge]] = (
        None  # auto-judges for this model that the LLM will use
    )
    on_turn: Optional[Callable[[OptimizationContext], bool]] = (
        None  # if you want manual control of pass/fail
    )
    # Results - Optional
    auto_commit: Optional[AutoCommitConfig] = (
        None  # configuration for automatically saving results back to LaunchDarkly
    )
    on_passing_result: Optional[Callable[[OptimizationContext], None]] = None
    on_failing_result: Optional[Callable[[OptimizationContext], None]] = None
    on_status_update: Optional[
        Callable[
            [
                Literal[
                    "init",
                    "generating",
                    "evaluating",
                    "generating variation",
                    "turn completed",
                    "success",
                    "failure",
                ],
                OptimizationContext,
            ],
            None,
        ]
    ] = None  # called to provide status updates during the optimization flow

    def __post_init__(self):
        """Validate required options."""
        if len(self.context_choices) < 1:
            raise ValueError("context_choices must have at least 1 context")
        if len(self.model_choices) < 1:
            raise ValueError("model_choices must have at least 1 model")
        if len(self.variable_choices) < 1:
            raise ValueError("variable_choices must have at least 1 variable choice")
        if self.judges is None and self.on_turn is None:
            raise ValueError("Either judges or on_turn must be provided")
        if self.judge_model is None:
            raise ValueError("judge_model must be provided")
