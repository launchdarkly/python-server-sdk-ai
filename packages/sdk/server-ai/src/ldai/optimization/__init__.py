"""Agent optimization types and dataclasses for self-optimizing agent configurations."""

import inspect
import json
import os
import random
import uuid
import urllib.request
import urllib.error
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

from ldai import log
from ldai.models import AIAgentConfig, AIJudgeConfig, AIJudgeConfigDefault


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
    """Configuration for an optimization judge."""

    judge_key: Optional[str] = None
    threshold: Optional[float] = None  # threshold the judge needs to match to pass
    acceptance_statement: Optional[str] = (
        None  # statement that the judge needs to confirm to be true to pass
    )

    def __post_init__(self):
        """Validate that either judge_key (config-type) or acceptance_statement is provided."""
        if self.judge_key is None and self.acceptance_statement is None:
            raise ValueError(
                "OptimizationJudge must have either judge_key (for config-type judges) or acceptance_statement (for acceptance statement judges)"
            )


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
class OptimizeJudgeContext:
    """Context for judge evaluation."""

    messages: List[Message]
    parameters: Dict[str, Any]
    tools: Optional[List[Dict[str, Any]]] = None


@dataclass
class OptimizeContext:
    """Context for a single optimization iteration."""

    scores: Dict[str, JudgeResult]  # the scores and rationales from the judges, if configured
    completion_response: str
    current_instructions: str
    current_parameters: Dict[str, Any]
    current_model: Optional[str] = None  # the current model being used
    user_input: Optional[str] = None  # the user input message for this iteration
    history: Sequence["OptimizeContext"] = field(
        default_factory=list
    )  # previous context items
    iteration: int = 0  # current iteration number
    structured_output_tool: Optional[StructuredOutputTool] = (
        None  # tool definition for structured output
    )

    def copy_without_history(self) -> "OptimizeContext":
        """
        Create a copy of this context without the history field (for flattening).

        :return: A new OptimizeContext with the same data but empty history
        """
        return OptimizeContext(
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
        # Convert scores (JudgeResult objects) to dicts using their to_json method
        scores_dict = {}
        for judge_key, judge_result in self.scores.items():
            scores_dict[judge_key] = judge_result.to_json()

        # Convert structured_output_tool to dict if present
        structured_output_tool_dict = None
        if self.structured_output_tool:
            structured_output_tool_dict = self.structured_output_tool.to_dict()

        # Recursively convert history items
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
class AutoCommitConfig:
    """Configuration for auto-committing optimization results to LaunchDarkly."""

    enabled: bool = False
    project_key: Optional[str] = None


@dataclass
class OptimizeOptions:
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
        Callable[[str, OptimizeContext], str], Callable[[str, OptimizeContext], Awaitable[str]]
    ]
    handle_judge_call: Union[
        Callable[[str, OptimizeJudgeContext], str],
        Callable[[str, OptimizeJudgeContext], Awaitable[str]],
    ]
    # Criteria for pass/fail - Optional
    user_input_options: Optional[List[str]] = (
        None  # optional list of user input messages to randomly select from
    )
    judges: Optional[Dict[str, OptimizationJudge]] = (
        None  # auto-judges for this model that the LLM will use
    )
    on_turn: Optional[Callable[[OptimizeContext], bool]] = (
        None  # if you want manual control of pass/fail
    )
    # Results - Optional
    auto_commit: Optional[AutoCommitConfig] = (
        None  # configuration for automatically saving results back to LaunchDarkly
    )
    on_passing_result: Optional[Callable[[OptimizeContext], None]] = None
    on_failing_result: Optional[Callable[[OptimizeContext], None]] = None
    on_status_update: Optional[
        Callable[[Literal["init", "generating", "evaluating", "generating variation", "turn completed", "success", "failure"], OptimizeContext], None]
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


class AgentOptimizer:
    """
    Handles the optimization loop for agent configurations.
    """

    def __init__(
        self,
        key: str,
        options: OptimizeOptions,
        config: "AIAgentConfig",
        judge_config: Callable[
            [str, Context, AIJudgeConfigDefault, Optional[Dict[str, Any]]],
            AIJudgeConfig,
        ],
    ):
        """
        Initialize the agent optimizer.

        :param key: The key of the agent to optimize
        :param options: Configuration options for the optimization process
        :param config: The agent configuration
        :param judge_config: Callable to fetch judge configs on-demand
        """
        self._key = key
        self._config = config
        self._options = options
        self._judge_config = judge_config
        self._current_instructions = (
            config.instructions or ""
        )  # Will be set from initial agent config
        self._current_parameters: Dict[str, Any] = config.model._parameters or {}
        self._current_model: Optional[str] = config.model.name if config.model else None
        self._history: List[OptimizeContext] = []

        if os.environ.get("LD_API_KEY"):
            self.has_api_key = True
            self.api_key = os.environ.get("LD_API_KEY")
        else:
            self.has_api_key = False
            self.api_key = None

    def _create_evaluation_tool(self) -> StructuredOutputTool:
        """
        Create the structured output tool for judge evaluations.

        :return: A StructuredOutputTool for evaluation responses
        """
        return StructuredOutputTool(
            name="return_evaluation",
            description="Returns an evaluation with a score and rationale.",
            input_schema={
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "The evaluation score (typically 0.0 to 1.0)",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Explanation of the evaluation",
                    },
                },
                "required": ["score", "rationale"],
            },
        )

    def _create_boolean_tool(self) -> StructuredOutputTool:
        """
        Create the structured output tool for acceptance judges.

        :return: A StructuredOutputTool for boolean evaluation responses
        """
        return StructuredOutputTool(
            name="return_boolean",
            description="Returns a boolean value and reasoning for the evaluation.",
            input_schema={
                "type": "object",
                "properties": {
                    "passed": {
                        "type": "boolean",
                        "description": "Whether the response passes the evaluation criteria",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Explanation of the evaluation decision",
                    },
                },
                "required": ["passed", "rationale"],
            },
        )

    def _create_variation_tool(self) -> StructuredOutputTool:
        """
        Create the structured output tool for variation generation.

        :return: A StructuredOutputTool for variation generation responses
        """
        return StructuredOutputTool(
            name="return_improved_configuration",
            description=(
                "Returns the improved agent configuration with updated instructions and parameters. "
                "This tool enforces structured output to ensure the response can be parsed and validated."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "current_instructions": {
                        "type": "string",
                        "description": "The improved agent instructions based on the evaluation feedback",
                    },
                    "current_parameters": {
                        "type": "object",
                        "description": "The improved agent parameters (e.g., temperature, max_tokens, etc.)",
                        "additionalProperties": True,
                    },
                    "model": {
                        "type": "string",
                        "description": "The model to use for the improved agent",
                        "enum": self._options.model_choices,
                    },
                },
                "required": ["current_instructions", "current_parameters", "model"],
                "additionalProperties": False,
            },
        )

    def _safe_status_update(
        self, status: str, context: OptimizeContext, iteration: int
    ) -> None:
        """
        Safely call on_status_update callback, catching and logging errors.

        :param status: The status string to pass to the callback
        :param context: The optimization context to pass to the callback
        :param iteration: Current iteration number for logging
        """
        if self._options.on_status_update:
            try:
                self._options.on_status_update(status, context.copy_without_history())
            except Exception as e:
                log.error(
                    f"[Turn {iteration}] -> on_status_update callback failed: {e}"
                )
                # Don't let callback errors affect the optimization flow

    async def _await_if_needed(
        self, result: Union[str, Awaitable[str]]
    ) -> str:
        """
        Handle both sync and async callable results.

        :param result: Either a string or an awaitable that returns a string
        :return: The string result
        """
        if inspect.iscoroutine(result):
            return await result
        else:
            return result

    def _serialize_scores(
        self, scores: Dict[str, JudgeResult]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Serialize judge results to JSON-serializable format.

        :param scores: Dictionary of judge results
        :return: Dictionary with serialized judge results
        """
        return {
            judge_key: judge_result.to_json()
            for judge_key, judge_result in scores.items()
        }

    def _build_message_history_text(
        self, base_text: str, reasoning_history: str
    ) -> str:
        """
        Combine base text with reasoning history.

        :param base_text: The base text (e.g., current instructions)
        :param reasoning_history: The formatted reasoning history from previous iterations
        :return: Combined text with reasoning history
        """
        if not reasoning_history:
            return base_text
        if base_text:
            return f"{base_text}\n\n## Previous Judge Evaluations:\n{reasoning_history}"
        else:
            return f"## Previous Judge Evaluations:\n{reasoning_history}"

    def _create_optimize_context(
        self,
        iteration: int,
        user_input: Optional[str] = None,
        completion_response: str = "",
        scores: Optional[Dict[str, JudgeResult]] = None,
    ) -> OptimizeContext:
        """
        Create an OptimizeContext with current state.

        :param iteration: Current iteration number
        :param user_input: Optional user input for this iteration
        :param completion_response: Completion response string
        :param scores: Optional dictionary of judge results
        :return: A new OptimizeContext instance
        """
        flat_history = [
            prev_ctx.copy_without_history() for prev_ctx in self._history
        ]
        return OptimizeContext(
            scores=scores or {},
            completion_response=completion_response,
            current_instructions=self._current_instructions,
            current_parameters=self._current_parameters.copy(),
            current_model=self._current_model,
            user_input=user_input,
            history=tuple(flat_history),
            iteration=iteration,
        )

    async def optimize(self) -> OptimizeContext:
        """
        Run the optimization loop following the optimize flow diagram.

        This method runs an optimization loop that:
        1. Executes the agent with current instructions (handle_agent_call)
        2. Evaluates the result using judges (handle_judge_call)
        3. Checks on_turn or evaluates response
        4. Generates new variations and loops until success or max_attempts

        :return: The final optimization context (may be passing or failing)
        """
        print(f"[Optimization] Starting optimization for agent: {self._key}")
        print("[Optimization] Starting configuration:")
        print(f"  Model: {self._current_model}")
        print(f"  Instructions: {self._current_instructions}")
        print(f"  Parameters: {self._current_parameters}")
        print(f"  Max attempts: {self._options.max_attempts}")

        # Create initial context for init status update
        initial_ctx = self._create_optimize_context(iteration=0)
        
        # Call on_status_update with "init" status (runs only once at initialization)
        self._safe_status_update("init", initial_ctx, 0)

        iteration = 0
        while True:
            iteration += 1
            print(f"[Turn {iteration}] -> Executing agent call...")

            # Select random user input if user_input_options is provided
            user_input = None
            if self._options.user_input_options:
                user_input = random.choice(self._options.user_input_options)

            # Create optimization context for this iteration
            optimize_ctx = self._create_optimize_context(
                iteration=iteration, user_input=user_input
            )

            # Step 1: handle_agent_call (user-defined action)
            # Call on_status_update with "generating" status
            self._safe_status_update("generating", optimize_ctx, iteration)
            
            try:
                result = self._options.handle_agent_call(self._key, optimize_ctx)
                completion_response = await self._await_if_needed(result)
                optimize_ctx.completion_response = completion_response
            except Exception as e:
                log.error(f"[Turn {iteration}] -> Agent call failed: {e}")
                if self._options.on_failing_result:
                    self._options.on_failing_result(optimize_ctx)
                raise

            # Step 2: handle_judge_call for each judge (auto-path, parallel)
            if self._options.judges:
                # Call on_status_update with "evaluating" status
                self._safe_status_update("evaluating", optimize_ctx, iteration)
                
                optimize_ctx.scores = await self._call_judges(
                    completion_response, iteration
                )

            # Step 3: Check if on_turn is defined (conditional path)
            if self._options.on_turn is not None:
                # User-defined path: on_turn() return
                try:
                    on_turn_result = self._options.on_turn(optimize_ctx)
                    if on_turn_result:
                        # on_turn returned True - success!
                        return self._handle_success(optimize_ctx, iteration)
                    else:
                        # on_turn returned False - generate new variation
                        if iteration >= self._options.max_attempts:
                            return self._handle_failure(optimize_ctx, iteration)
                        self._history.append(optimize_ctx)
                        await self._generate_new_variation(iteration)
                        # Call on_status_update with "turn completed" status before starting next turn
                        self._safe_status_update("turn completed", optimize_ctx, iteration)
                        continue
                except Exception as e:
                    log.error(f"[Turn {iteration}] -> on_turn evaluation failed: {e}")
                    # Treat exception as failure, generate new variation
                    self._history.append(optimize_ctx)
                    await self._generate_new_variation(iteration)
                    if iteration >= self._options.max_attempts:
                        return self._handle_failure(optimize_ctx, iteration)
                    # Call on_status_update with "turn completed" status before starting next turn
                    self._safe_status_update("turn completed", optimize_ctx, iteration)
                    continue
            else:
                # Auto-path: _evaluate_response() Passes?)
                passes = self._evaluate_response(optimize_ctx)
                if passes:
                    # Success path
                    return self._handle_success(optimize_ctx, iteration)
                else:
                    # Generate new variation
                    self._history.append(optimize_ctx)
                    await self._generate_new_variation(iteration)
                    # Check max_turns after generating variation
                    if iteration >= self._options.max_attempts:
                        return self._handle_failure(optimize_ctx, iteration)
                    # Call on_status_update with "turn completed" status before starting next turn
                    self._safe_status_update("turn completed", optimize_ctx, iteration)
                    continue

    def _build_reasoning_history(self) -> str:
        """
        Build a formatted string of reasoning from previous iterations.

        :return: Formatted string containing reasoning history
        """
        if not self._history:
            return ""

        reasoning_parts = []
        for i, prev_ctx in enumerate(self._history, 1):
            if prev_ctx.scores:
                reasoning_parts.append(f"## Iteration {i} Judge Evaluations:")
                for judge_key, result in prev_ctx.scores.items():
                    reasoning_parts.append(f"- {judge_key}: Score {result.score}")
                    if result.rationale:
                        reasoning_parts.append(f"  Reasoning: {result.rationale}")
                reasoning_parts.append("")

        return "\n".join(reasoning_parts)

    def _parse_judge_response(
        self,
        judge_response_str: str,
        judge_key: str,
        judge_identifier: str,
        iteration: int,
        clamp_score: bool = True,
    ) -> JudgeResult:
        """
        Parse judge response to extract score and rationale.

        :param judge_response_str: The raw judge response string
        :param judge_key: The key for this judge in the judges dict
        :param judge_identifier: Identifier for logging (e.g., judge_key or judge_key from config)
        :param iteration: Current iteration number
        :param clamp_score: Whether to clamp score to [0.0, 1.0] range
        :return: JudgeResult with score and rationale
        """
        try:
            response_data = json.loads(judge_response_str)
            if isinstance(response_data, dict):
                # Look for score field (primary) or fallback to keys ending with "_score"
                score_key = "score" if "score" in response_data else None
                if not score_key:
                    for key in response_data.keys():
                        if key.endswith("_score"):
                            score_key = key
                            break
                
                if score_key:
                    score = float(response_data[score_key])
                    if clamp_score:
                        score = max(0.0, min(1.0, score))
                    rationale = response_data.get("rationale") or response_data.get("reasoning")
                    return JudgeResult(score=score, rationale=rationale)
                else:
                    log.warn(
                        f"[Turn {iteration}] -> Judge {judge_identifier} response missing score field. "
                        f"Available keys: {list(response_data.keys())}"
                    )
                    raise ValueError("No score field in JSON response")
            else:
                # If JSON but not a dict, try to convert to float
                score = float(response_data)
                if clamp_score:
                    score = max(0.0, min(1.0, score))
                return JudgeResult(score=score, rationale=None)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Fall back to parsing as plain numeric string (legacy support)
            try:
                score = float(judge_response_str.strip())
                if clamp_score:
                    score = max(0.0, min(1.0, score))
                log.warn(
                    f"[Turn {iteration}] -> Judge {judge_identifier} returned non-JSON response, "
                    f"parsed as numeric: {score}"
                )
                return JudgeResult(score=score, rationale=None)
            except ValueError:
                log.warn(
                    f"[Turn {iteration}] -> Judge {judge_identifier} returned invalid response: "
                    f"{judge_response_str[:200]}. Error: {e}"
                )
                return JudgeResult(score=0.0, rationale=None)

    async def _call_judges(
        self, completion_response: str, iteration: int
    ) -> Dict[str, JudgeResult]:
        """
        Call all judges in parallel (auto-path).

        For judges with judge_key: Fetches judge config on-demand from LaunchDarkly SDK.
        For judges with acceptance_statement: Uses handle_judge_call callback.

        :param completion_response: The agent's completion response to evaluate
        :param iteration: Current iteration number
        :return: Dictionary of judge results (score and rationale)
        """
        if not self._options.judges:
            return {}

        print(f"[Turn {iteration}] -> Executing evaluation...")
        reasoning_history = self._build_reasoning_history()
        judge_results: Dict[str, JudgeResult] = {}

        for judge_key, optimization_judge in self._options.judges.items():
            try:
                if optimization_judge.judge_key is not None:
                    result = await self._evaluate_config_judge(
                        judge_key, optimization_judge, completion_response, iteration, reasoning_history
                    )
                    judge_results[judge_key] = result
                else:
                    result = await self._evaluate_acceptance_judge(
                        judge_key, optimization_judge, completion_response, iteration, reasoning_history
                    )
                    judge_results[judge_key] = result
            except Exception as e:
                log.error(
                    f"[Turn {iteration}] -> Judge {judge_key} evaluation failed: {e}"
                )
                judge_results[judge_key] = JudgeResult(score=0.0, rationale=None)

        # Serialize judge results to JSON for proper output
        judge_results_json = self._serialize_scores(judge_results)
        print(f"[Turn {iteration}] -> Evaluation result: {json.dumps(judge_results_json, indent=2)}")
        return judge_results

    async def _evaluate_config_judge(
        self,
        judge_key: str,
        optimization_judge: "OptimizationJudge",
        completion_response: str,
        iteration: int,
        reasoning_history: str,
    ) -> JudgeResult:
        """
        Evaluate using a config-type judge (with judge_key).

        :param judge_key: The key for this judge in the judges dict
        :param optimization_judge: The optimization judge configuration
        :param completion_response: The agent's completion response to evaluate
        :param iteration: Current iteration number
        :param reasoning_history: Formatted string of reasoning from previous iterations
        :return: The judge result with score and rationale
        """
        # Config-type judge: fetch judge config on-demand
        input_text = self._current_instructions or ""
        # Combine current instructions with reasoning history for message_history
        message_history_text = self._build_message_history_text(input_text, reasoning_history)
        
        judge_config = self._judge_config(
            optimization_judge.judge_key,
            self._options.context_choices[0],
            AIJudgeConfigDefault(enabled=False),
            {
                "message_history": message_history_text,
                "response_to_evaluate": completion_response,
            },
        )

        if not judge_config.enabled:
            log.warn(
                f"[Turn {iteration}] -> Judge {optimization_judge.judge_key} is disabled"
            )
            return JudgeResult(score=0.0, rationale=None)

        if not judge_config.messages:
            log.warn(
                f"[Turn {iteration}] -> Judge {optimization_judge.judge_key} has no messages"
            )
            return JudgeResult(score=0.0, rationale=None)

        # Convert LDMessage to Message objects
        # Append structured output tool instruction to system messages
        judge_messages = []
        for msg in judge_config.messages:
            content = msg.content
            if msg.role == "system":
                content += " Use the structured output tool to format your response. You should always return a JSON object with a score and rationale."
            judge_messages.append(Message(role=msg.role, content=content))

        # Build parameters from judge config
        parameters = {}
        tools = []
        if judge_config.model:
            parameters["model"] = judge_config.model.name
            if judge_config.model._parameters:
                # Extract tools if present
                existing_tools = judge_config.model._parameters.get("tools")
                if existing_tools:
                    tools = existing_tools if isinstance(existing_tools, list) else [existing_tools]
                    # Convert to dicts if needed
                    tools = [tool.to_dict() if hasattr(tool, 'to_dict') else tool for tool in tools]
                # Copy parameters excluding tools
                parameters.update({
                    k: v for k, v in judge_config.model._parameters.items()
                    if k != "tools"
                })

        # Add structured output tool for score and rationale
        tools.append(self._create_evaluation_tool().to_dict())

        judge_ctx = OptimizeJudgeContext(
            messages=judge_messages,
            parameters=parameters,
            tools=tools,
        )

        result = self._options.handle_judge_call(self._options.judge_model, judge_ctx)
        judge_response_str = await self._await_if_needed(result)

        print(f"[Turn {iteration}] -> Judge response ({judge_key}): {judge_response_str}")

        # Parse judge response - expect structured JSON output
        judge_identifier = optimization_judge.judge_key or judge_key
        return self._parse_judge_response(
            judge_response_str, judge_key, judge_identifier, iteration, clamp_score=False
        )

    async def _evaluate_acceptance_judge(
        self,
        judge_key: str,
        optimization_judge: "OptimizationJudge",
        completion_response: str,
        iteration: int,
        reasoning_history: str,
    ) -> JudgeResult:
        """
        Evaluate using an acceptance statement judge.

        :param judge_key: The key for this judge in the judges dict
        :param optimization_judge: The optimization judge configuration
        :param completion_response: The agent's completion response to evaluate
        :param iteration: Current iteration number
        :param reasoning_history: Formatted string of reasoning from previous iterations
        :return: The judge result with score and rationale
        """
        if not optimization_judge.acceptance_statement:
            log.error(
                f"[Turn {iteration}] -> Judge {judge_key} has no acceptance_statement"
            )
            return JudgeResult(score=0.0, rationale=None)

        # Build message history with reasoning
        message_history_text = self._build_message_history_text("", reasoning_history)

        # Build judge context for LLM call
        judge_messages = [
            Message(
                role="system",
                content=f"""You are a judge that evaluates the response to the user's question. 
                
                Here is the statement that you should evaluate the response against: '{optimization_judge.acceptance_statement}'
                Here is the history of all messages between the user and the assistant: {message_history_text}
                You should score the response based on how well it meets the acceptance statement using a score between 0.0 and 1.0.
                A score of 0.0 means it does not match at all, while a score of 1.0 means it matches perfectly. 
                A score of 0.3-0.7 means it matches partially, while a score of 0.7-1.0 means it matches well.
                A score of 0.0-0.3 means that it does not match well at all. You can return any value between 0.0 and 1.0.
                You should also provide a rationale for your score.
                You should call the structured output tool to format your response.

                Here is an example of a good response:
                {{
                    "score": 0.8,
                    "rationale": "The response matches the acceptance statement well. It provides a detailed explanation of the concept and its applications."
                }}
                """,
            ),
            Message(
                role="user",
                content=f"Here is the response to evaluate: {completion_response}",
            ),
        ]

        # Create structured output tool for evaluation response with score and rationale
        evaluation_tool = self._create_evaluation_tool()
        
        judge_ctx = OptimizeJudgeContext(
            messages=judge_messages,
            parameters={"model": self._options.judge_model},
            tools=[evaluation_tool.to_dict()],
        )

        result = self._options.handle_judge_call(self._options.judge_model, judge_ctx)
        judge_response = await self._await_if_needed(result)

        print(f"""--------------------------------
Judge response: {judge_response}
--------------------------------""")

        print(f"[Turn {iteration}] -> Judge response ({judge_key}): {judge_response}")
        # Parse judge response - expect structured JSON output with score and rationale
        return self._parse_judge_response(
            judge_response, judge_key, judge_key, iteration, clamp_score=True
        )

    def _check_judge_threshold(
        self,
        judge_key: str,
        score: float,
        optimization_judge: "OptimizationJudge",
    ) -> bool:
        """
        Check if a judge score meets the threshold requirement.

        :param judge_key: The key for this judge
        :param score: The score from the judge evaluation
        :param optimization_judge: The optimization judge configuration
        :return: True if the score meets the threshold, False otherwise
        """
        if optimization_judge.judge_key is not None:
            # Config-type judge: compare score to threshold
            threshold = optimization_judge.threshold if optimization_judge.threshold is not None else 0.9
            if score < threshold:
                print(
                    f"Judge {judge_key} failed: score {score} < threshold {threshold}"
                )
                return False
            else:
                log.debug(
                    f"Judge {judge_key} passed: score {score} >= threshold {threshold}"
                )
                return True
        else:
            # Acceptance statement judge: compare score to threshold (reuse threshold from OptimizationJudge)
            threshold = optimization_judge.threshold if optimization_judge.threshold is not None else 0.9
            
            if score < threshold:
                print(
                    f"Judge {judge_key} (acceptance_statement) failed: score {score} < threshold {threshold}"
                )
                return False
            else:
                log.debug(
                    f"Judge {judge_key} (acceptance_statement) passed: score {score} >= threshold {threshold}"
                )
                return True

    def _evaluate_response(self, optimize_ctx: OptimizeContext) -> bool:
        """
        Evaluate if the response passes all judge criteria (auto-path).

        :param optimize_ctx: The optimization context with scores
        :return: True if all judges passed, False otherwise
        """
        if not self._options.judges:
            log.warn("No judges configured for evaluation")
            return False
        
        for judge_key, optimization_judge in self._options.judges.items():
            result = optimize_ctx.scores.get(judge_key, JudgeResult(score=0.0, rationale=None))
            score = result.score
            log.debug(f"Evaluating judge {judge_key}: score={score}, judge_key={optimization_judge.judge_key}, threshold={optimization_judge.threshold}")
            
            if not self._check_judge_threshold(judge_key, score, optimization_judge):
                return False
        
        return True

    async def _generate_new_variation(self, iteration: int) -> OptimizeContext:
        """
        Generate new variation for next iteration (auto-path).

        Calls handle_agent_call to generate a new variation and updates current_instructions
        and current_parameters based on the returned OptimizeContext.
        
        :param iteration: The current iteration number for logging
        """
        print(f"[Turn {iteration}] -> Generating new variation...")
        
        # Create a context for status update (before generating variation)
        status_ctx = self._create_optimize_context(iteration=iteration)
        
        # Call on_status_update with "generating variation" status
        self._safe_status_update("generating variation", status_ctx, iteration)
        # Get the most recent context for previous result and feedback
        previous_ctx = self._history[-1] if self._history else None

        # Build instructions with previous result, config, and feedback
        instructions_parts = [
            "You are an assistant that helps improve agent configurations through iterative optimization.",
            "",
            "Your task is to generate improved agent instructions and parameters based on the feedback provided.",
            "",
        ]

        parameters_instructions = [
            "Return these values in a JSON object with the following keys: current_instructions, current_parameters, and model.",
            "Example:",
            "{",
            '  "current_instructions": "...',
            '  "current_parameters": {',
            '    "...": "..."',
            "  },",
            '  "model": "gpt-4o"',
            "}",
            "Parameters should only be things that are directly parseable by an LLM call, for example, temperature, max_tokens, etc."
            "Do not include any other parameters that are not directly parseable by an LLM call. If you want to provide instruction for tone or other attributes, provide them directly in the instructions.",            
        ]

        model_instructions = [
            "You may also choose to change the model if you believe that the current model is not performing well or a different model would be better suited for the task. "
            f"Here are the models you may choose from: {self._options.model_choices}. You must always return a model property, even if it's the same as the current model.",
            "When suggesting a new model, you should provide a rationale for why you believe the new model would be better suited for the task.",
        ]

        prev_context_instructions = [
            f"Model: {previous_ctx.current_model}",
            f"Instructions: {previous_ctx.current_instructions}",
            f"Parameters: {previous_ctx.current_parameters}",
        ]

        if previous_ctx:
            instructions_parts.extend(
                [
                    "## Previous Configuration:",
                    *prev_context_instructions,
                    "",
                    "## Previous Result:",
                    f"{previous_ctx.completion_response}",
                    "",
                ]
            )

            if previous_ctx.scores:
                instructions_parts.extend(
                    [
                        "## Evaluation Feedback:",
                    ]
                )
                for judge_key, result in previous_ctx.scores.items():
                    optimization_judge = (
                        self._options.judges.get(judge_key)
                        if self._options.judges
                        else None
                    )
                    if optimization_judge:
                        score = result.score
                        if optimization_judge.threshold is not None:
                            passed = score >= optimization_judge.threshold
                            status = "PASSED" if passed else "FAILED"
                            feedback_line = f"- {judge_key}: Score {score:.3f} (threshold: {optimization_judge.threshold}) - {status}"
                            if result.rationale:
                                feedback_line += f"\n  Reasoning: {result.rationale}"
                            instructions_parts.append(feedback_line)
                        else:
                            passed = score >= 1.0
                            status = "PASSED" if passed else "FAILED"
                            feedback_line = f"- {judge_key}: {status}"
                            if result.rationale:
                                feedback_line += f"\n  Reasoning: {result.rationale}"
                            instructions_parts.append(feedback_line)
                instructions_parts.append("")

            instructions_parts.extend(
                [
                    "## Improvement Instructions:",
                    "Based on the evaluation feedback above, generate improved agent instructions and parameters.",
                    "Focus on addressing the areas where the evaluation failed or scored below threshold.",
                    "The new configuration should aim to improve the agent's performance on the evaluation criteria.",
                    *model_instructions,
                    "",
                    "Return the improved configuration in a structured format that can be parsed to update:",
                    "1. The agent instructions (current_instructions)",
                    "2. The agent parameters (current_parameters)",
                    "3. The model (model) - you must always return a model, even if it's the same as the current model.",
                    *parameters_instructions,
                ]
            )
        else:
            instructions_parts.extend(
                [
                    "## Current Configuration:",
                    *prev_context_instructions,
                    "",
                    "Generate an improved version of this configuration.",
                    *model_instructions,
                    *parameters_instructions,
                ]
            )

        instructions = "\n".join(instructions_parts)

        # Create structured output tool definition for variation generation
        structured_output_tool = self._create_variation_tool()

        # Create a flat history list (without nested history)
        flat_history = [prev_ctx.copy_without_history() for prev_ctx in self._history]

        # Create context for variation generation
        variation_ctx = OptimizeContext(
            scores={},
            completion_response="",
            current_instructions=instructions,
            current_parameters={"temperature": 0.1},
            current_model=self._current_model,
            user_input=None,  # No user input for variation generation
            history=tuple(flat_history),
            iteration=len(self._history) + 1,  # Next iteration number
            structured_output_tool=structured_output_tool,
        )

        # Call handle_agent_call to generate new variation
        # This should return a JSON string matching the structured output schema
        result = self._options.handle_agent_call(self._key, variation_ctx)
        response_str = await self._await_if_needed(result)

        # Parse the JSON response to extract instructions and parameters
        response_data = None
        try:
            # Try to parse as JSON directly
            response_data = json.loads(response_str)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            # Some LLMs may wrap the JSON in markdown code blocks or add extra text
            import re

            # First, try to extract JSON from markdown code blocks
            code_block_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```',
                response_str,
                re.DOTALL,
            )
            if code_block_match:
                try:
                    response_data = json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    pass  # Fall through to general extraction

            # If code block extraction failed, try to find the first complete JSON object
            if response_data is None:
                # Find the first opening brace and then match balanced braces
                brace_count = 0
                start_idx = response_str.find('{')
                if start_idx != -1:
                    for i in range(start_idx, len(response_str)):
                        if response_str[i] == '{':
                            brace_count += 1
                        elif response_str[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found complete JSON object
                                json_str = response_str[start_idx:i+1]
                                try:
                                    response_data = json.loads(json_str)
                                    break
                                except json.JSONDecodeError:
                                    # Try next JSON object
                                    start_idx = response_str.find('{', start_idx + 1)
                                    if start_idx == -1:
                                        break
                                    brace_count = 0
                                    i = start_idx - 1

            # If still no valid JSON, try the old regex as fallback
            if response_data is None:
                json_match = re.search(
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"current_instructions"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                    response_str,
                    re.DOTALL,
                )
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        log.error(
                            f"[Optimization] Extracted JSON string failed to parse: {json_match.group()[:200]}"
                        )
                        raise ValueError(
                            "Failed to parse extracted JSON from variation generation response"
                        )

            # Final check - if we still don't have response_data, raise error
            if response_data is None:
                log.error(
                    f"[Optimization] Failed to extract JSON from variation generation response. "
                    f"Response length: {len(response_str)}, first 200 chars: {response_str[:200]}"
                )
                raise ValueError(
                    "Failed to parse structured output from variation generation. "
                    "Expected JSON object with 'current_instructions', 'current_parameters', and 'model' fields."
                )

        # Extract and update current state from the parsed response
        # Log the parsed response for debugging
        log.debug(
            f"[Turn {iteration}] -> Parsed response_data keys: {list(response_data.keys()) if response_data else 'None'}"
        )
        if response_data and "model" in response_data:
            log.debug(
                f"[Turn {iteration}] -> Model field value: '{response_data['model']}'"
            )
        
        missing_fields = []
        if "current_instructions" not in response_data:
            missing_fields.append("current_instructions")
        if "current_parameters" not in response_data:
            missing_fields.append("current_parameters")
        if "model" not in response_data:
            missing_fields.append("model")
        
        if missing_fields:
            log.error(
                f"[Turn {iteration}] -> Response missing required fields: {', '.join(missing_fields)}. "
                f"Received fields: {list(response_data.keys())}. "
                f"Full response_data: {json.dumps(response_data, indent=2)}"
            )
            raise ValueError(
                f"Response missing required fields: {', '.join(missing_fields)}. "
                f"Received fields: {list(response_data.keys())}"
            )

        self._current_instructions = response_data["current_instructions"]
        self._current_parameters = response_data["current_parameters"]
        # Update model - it should always be provided since it's required
        model_value = response_data.get("model", "").strip() if isinstance(response_data.get("model"), str) else response_data.get("model")
        if not model_value:
            log.warn(
                f"[Turn {iteration}] -> Model field is empty or None in response, keeping current model {self._current_model}"
            )
        elif model_value not in self._options.model_choices:
            log.warn(
                f"[Turn {iteration}] -> Model '{model_value}' not in model_choices {self._options.model_choices}, keeping current model {self._current_model}"
            )
        else:
            old_model = self._current_model
            self._current_model = model_value
            log.info(
                f"[Turn {iteration}] -> Model updated from '{old_model}' to '{self._current_model}'"
            )

        print(f"""[Turn {iteration}] -> New variation generated: 
        Instructions: {self._current_instructions[:100]}...
        Model: {self._current_model}
        Parameters: {self._current_parameters}""")

        # Create a new context with the updated values for return
        updated_ctx = OptimizeContext(
            scores={},
            completion_response=response_str,
            current_instructions=self._current_instructions,
            current_parameters=self._current_parameters.copy(),
            current_model=self._current_model,
            user_input=None,
            history=variation_ctx.history,
            iteration=variation_ctx.iteration,
            structured_output_tool=variation_ctx.structured_output_tool,
        )

        return updated_ctx

    def _handle_success(
        self, optimize_ctx: OptimizeContext, iteration: int
    ) -> OptimizeContext:
        """
        Handle successful optimization result.

        :param optimize_ctx: The successful optimization context
        :param iteration: Current iteration number
        :return: The optimization context
        """
        print(f"[Turn {iteration}] Successful result")
        print("[Optimization] Final configuration:")
        print(f"  Model: {optimize_ctx.current_model}")
        print(f"  Instructions: {optimize_ctx.current_instructions}")
        print(f"  Parameters: {optimize_ctx.current_parameters}")
        # Serialize scores to JSON for proper output
        scores_json = self._serialize_scores(optimize_ctx.scores)
        print(f"[Optimization] Final scores: {json.dumps(scores_json, indent=2)}")

        # Call on_status_update with "success" status
        self._safe_status_update("success", optimize_ctx, iteration)

        if self._options.auto_commit and self._options.auto_commit.enabled:
            self.auto_commit(optimize_ctx)

        if self._options.on_passing_result:
            self._options.on_passing_result(optimize_ctx)

        return optimize_ctx

    def _handle_failure(
        self, optimize_ctx: OptimizeContext, iteration: int
    ) -> OptimizeContext:
        """
        Handle failure when max_turns is reached (failure state).

        :param optimize_ctx: The final optimization context
        :param iteration: Current iteration number
        :return: The optimization context
        """
        log.warning(
            f"[Optimization] Max attempts ({self._options.max_attempts}) reached without success"
        )
        final_ctx = self._history[-1] if self._history else optimize_ctx
        print("[Optimization] Final configuration:")
        print(f"  Model: {final_ctx.current_model}")
        print(f"  Instructions: {final_ctx.current_instructions}")
        print(f"  Parameters: {final_ctx.current_parameters}")
        # Serialize scores to JSON for proper output
        scores_json = self._serialize_scores(final_ctx.scores)
        print(f"[Optimization] Final scores: {json.dumps(scores_json, indent=2)}")
        
        # Call on_status_update with "failure" status
        self._safe_status_update("failure", final_ctx, iteration)
        
        if self._options.on_failing_result:
            self._options.on_failing_result(final_ctx)
        return final_ctx

    def auto_commit(self, optimize_ctx: OptimizeContext) -> None:
        """
        Auto-commit the optimization context to LaunchDarkly.

        :param optimize_ctx: The optimization context to commit
        """
        if not self._options.auto_commit or not self._options.auto_commit.enabled:
            return
        
        if not self.has_api_key:
            log.warn("Auto-commit is enabled but no API key is available. Skipping commit.")
            return

        project_key = self._options.auto_commit.project_key
        if not project_key:
            log.warn("Auto-commit is enabled but no project_key is provided. Skipping commit.")
            return

        base_url = "https://api.launchdarkly.com"
        url_endpoint = f"{base_url}/api/v2/projects/{project_key}/ai-configs/{self._key}/variations"

        print(url_endpoint)
        
        request_body = {
          "key": f"autogenerated-variation-{uuid.uuid4()}",
          "name": f"Autogenerated Variation {uuid.uuid4()}",
          "modelConfigKey": optimize_ctx.current_model,
          "model": {
            "parameters": optimize_ctx.current_parameters,
          },
          "instructions": optimize_ctx.current_instructions,
        }

        request_data = json.dumps(request_body).encode('utf-8')
        req = urllib.request.Request(url_endpoint, data=request_data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('LD-API-Version', 'beta')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        
        try:
            response = urllib.request.urlopen(req)
            contents = response.read()
            log.info(f"Auto-commit successful: {response.getcode()}")
            print(contents.decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else "No error body"
            log.error(
                f"Auto-commit failed with HTTP {e.code}: {e.reason}. "
                f"Response: {error_body}"
            )
            raise
        except urllib.error.URLError as e:
            log.error(
                f"Auto-commit failed with URL error: {e.reason}. "
                f"URL: {url_endpoint}"
            )
            raise
        except Exception as e:
            log.error(f"Auto-commit failed with unexpected error: {e}")
            raise
        