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

from ldai import AIAgentConfig
from ldai.models import LDMessage, ModelConfig
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
class ToolDefinition:
    """
    Generic tool definition for enforcing structured output from LLM responses.

    This tool can be used with any LLM provider to ensure responses conform to
    a specific JSON schema. The tool takes the LLM's response and returns
    parsed and validated data according to the input_schema.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON schema defining the expected output structure
    type: Literal["function"] = "function"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool definition to a dictionary format compatible with LLM APIs.

        :return: Dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolDefinition":
        """
        Construct a ToolDefinition from a plain dictionary.

        :param data: Dictionary with at least a ``name`` key; ``description`` and
            ``input_schema`` default to empty values when absent.
        :return: A new ToolDefinition instance
        """
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            type=data.get("type", "function"),
        )


@dataclass
class AIJudgeCallConfig:
    """
    Configuration passed to ``handle_judge_call``.

    Carries everything needed to run a judge in either paradigm:

    * **Completions path** — pass ``messages`` directly to ``chat.completions.create``.
      The full system + user turn sequence is already assembled and interpolated.
    * **Agents path** — use ``instructions`` as the system prompt and
      ``OptimizationJudgeContext.user_input`` as the ``Runner.run`` input.

    Both fields are always populated, regardless of whether the judge comes from a
    LaunchDarkly flag (config judge) or an inline acceptance statement.
    """

    key: str
    model: ModelConfig
    instructions: str
    messages: List[LDMessage]


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
    # variable set chosen for this iteration; interpolated into instructions at call time
    current_variables: Dict[str, Any]
    current_model: Optional[str] = None  # the current model being used
    user_input: Optional[str] = None  # the user input message for this iteration
    history: Sequence[OptimizationContext] = field(
        default_factory=list
    )  # previous context items
    iteration: int = 0  # current iteration number

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
            current_variables=self.current_variables,
            current_model=self.current_model,
            user_input=self.user_input,
            history=(),  # Empty history to keep it flat
            iteration=self.iteration,
        )

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the optimization context to a JSON-serializable dictionary.

        :return: Dictionary representation of the context that can be serialized with json.dumps()
        """
        scores_dict = {}
        for judge_key, judge_result in self.scores.items():
            scores_dict[judge_key] = judge_result.to_json()

        history_list = [ctx.to_json() for ctx in self.history]

        return {
            "scores": scores_dict,
            "completion_response": self.completion_response,
            "current_instructions": self.current_instructions,
            "current_parameters": self.current_parameters,
            "current_model": self.current_model,
            "user_input": self.user_input,
            "current_variables": self.current_variables,
            "history": history_list,
            "iteration": self.iteration,
        }


@dataclass
class OptimizationJudgeContext:
    """Context for a single judge evaluation turn."""

    user_input: str  # the agent response being evaluated
    variables: Dict[str, Any] = field(default_factory=dict)  # variable set used during agent generation


# Shared callback type aliases used by both OptimizationOptions and
# OptimizationFromConfigOptions to avoid duplicating the full signatures.
# Placed here so all referenced types (OptimizationContext, AIJudgeCallConfig,
# OptimizationJudgeContext) are already defined above.
HandleAgentCall = Union[
    Callable[[str, AIAgentConfig, OptimizationContext, Dict[str, Callable[..., Any]]], str],
    Callable[[str, AIAgentConfig, OptimizationContext, Dict[str, Callable[..., Any]]], Awaitable[str]],
]
HandleJudgeCall = Union[
    Callable[[str, AIJudgeCallConfig, OptimizationJudgeContext, Dict[str, Callable[..., Any]]], str],
    Callable[[str, AIJudgeCallConfig, OptimizationJudgeContext, Dict[str, Callable[..., Any]]], Awaitable[str]],
]

_StatusLiteral = Literal[
    "init",
    "generating",
    "evaluating",
    "generating variation",
    "validating",
    "turn completed",
    "success",
    "failure",
]


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
    handle_agent_call: HandleAgentCall
    handle_judge_call: HandleJudgeCall
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
    # called to provide status updates during the optimization flow
    on_status_update: Optional[Callable[[_StatusLiteral, OptimizationContext], None]] = None

    def __post_init__(self):
        """Validate required options."""
        if len(self.context_choices) < 1:
            raise ValueError("context_choices must have at least 1 context")
        if len(self.model_choices) < 1:
            raise ValueError("model_choices must have at least 1 model")
        if self.judges is None and self.on_turn is None:
            raise ValueError("Either judges or on_turn must be provided")
        if self.judge_model is None:
            raise ValueError("judge_model must be provided")


@dataclass
class GroundTruthSample:
    """A single ground truth evaluation sample for use with optimize_from_ground_truth_options.

    Each sample ties together the user input, expected response, and variable set for one
    evaluation. Samples are evaluated in order; the optimization only passes if all samples
    pass their judges in the same attempt.

    :param user_input: The user message to send to the agent for this evaluation.
    :param expected_response: The ideal response the agent should produce. Injected into
        judge context so judges can score actual vs. expected.
    :param variables: Variable set interpolated into the agent instructions for this sample.
        Defaults to an empty dict if no placeholders are used.
    """

    user_input: str
    expected_response: str
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthOptimizationOptions:
    """Options for optimize_from_ground_truth_options.

    Mirrors OptimizationOptions but replaces variable_choices / user_input_options with
    ground_truth_responses. Each GroundTruthSample bundles the user input, expected
    response, and variable set for one evaluation. All N samples must pass their judges
    in the same attempt for the optimization to succeed.

    :param context_choices: One or more LD evaluation contexts to use.
    :param ground_truth_responses: Ordered list of ground truth samples to evaluate.
        At least 1 required. All samples share the same instructions and model being optimized.
    :param max_attempts: Maximum number of variation attempts before the run is marked failed.
    :param model_choices: Model IDs the variation generator may select from. At least 1 required.
    :param judge_model: Model used for judge evaluation. Should remain consistent across attempts.
    :param handle_agent_call: Callback that invokes the agent and returns its response.
    :param handle_judge_call: Callback that invokes a judge LLM and returns its response.
    :param judges: Auto-judges (config judges and/or acceptance statements) to score each response.
    :param on_turn: Optional manual pass/fail callback applied per sample; skips judge scoring when provided.
    :param on_sample_result: Called with each sample's OptimizationContext as results arrive,
        before the overall pass/fail decision is made for the attempt.
    :param on_passing_result: Called once with the last context when all N samples pass.
    :param on_failing_result: Called once with the last context when max attempts are exhausted.
    :param on_status_update: Called on each status transition during the run.
    """

    context_choices: List[Context]
    ground_truth_responses: List[GroundTruthSample]
    max_attempts: int
    model_choices: List[str]
    judge_model: str
    handle_agent_call: HandleAgentCall
    handle_judge_call: HandleJudgeCall
    judges: Optional[Dict[str, OptimizationJudge]] = None
    on_turn: Optional[Callable[[OptimizationContext], bool]] = None
    on_sample_result: Optional[Callable[[OptimizationContext], None]] = None
    on_passing_result: Optional[Callable[[OptimizationContext], None]] = None
    on_failing_result: Optional[Callable[[OptimizationContext], None]] = None
    on_status_update: Optional[
        Callable[
            [
                _StatusLiteral,
                OptimizationContext,
            ],
            None,
        ]
    ] = None

    def __post_init__(self):
        """Validate required options."""
        if len(self.context_choices) < 1:
            raise ValueError("context_choices must have at least 1 context")
        if len(self.model_choices) < 1:
            raise ValueError("model_choices must have at least 1 model")
        if len(self.ground_truth_responses) < 1:
            raise ValueError("ground_truth_responses must have at least 1 sample")
        if self.judges is None and self.on_turn is None:
            raise ValueError("Either judges or on_turn must be provided")


@dataclass
class OptimizationFromConfigOptions:
    """User-provided options for optimize_from_config.

    Fields that come from the LaunchDarkly API (max_attempts, model_choices,
    judge_model, variable_choices, user_input_options, judges) are omitted here
    and sourced from the fetched agent optimization config instead.

    :param project_key: LaunchDarkly project key used to build API paths.
    :param context_choices: One or more LD evaluation contexts to use.
    :param handle_agent_call: Callback that invokes the agent and returns its response.
    :param handle_judge_call: Callback that invokes a judge and returns its response.
    :param on_turn: Optional manual pass/fail callback; when provided, judge scoring is skipped.
    :param on_sample_result: Ground truth path only. Called with each sample's
        OptimizationContext as results arrive during a ground truth run.
    :param on_passing_result: Called with the winning OptimizationContext on success.
    :param on_failing_result: Called with the final OptimizationContext on failure.
    :param on_status_update: Called on each status transition; chained after the
        automatic result-persistence POST so it always runs after the record is saved.
    :param base_url: Base URL of the LaunchDarkly instance. Defaults to
        https://app.launchdarkly.com. Override to target a staging instance.
    """

    project_key: str
    context_choices: List[Context]
    handle_agent_call: HandleAgentCall
    handle_judge_call: HandleJudgeCall
    on_turn: Optional[Callable[["OptimizationContext"], bool]] = None
    on_sample_result: Optional[Callable[["OptimizationContext"], None]] = None
    on_passing_result: Optional[Callable[["OptimizationContext"], None]] = None
    on_failing_result: Optional[Callable[["OptimizationContext"], None]] = None
    on_status_update: Optional[Callable[[_StatusLiteral, "OptimizationContext"], None]] = None
    base_url: Optional[str] = None

    def __post_init__(self):
        """Validate required options."""
        if len(self.context_choices) < 1:
            raise ValueError("context_choices must have at least 1 context")
