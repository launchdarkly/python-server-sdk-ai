"""Client for LaunchDarkly AI agent optimization."""

import dataclasses
import json
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from coolname import generate_slug

from ldai import AIAgentConfig, AIJudgeConfig, AIJudgeConfigDefault, LDAIClient
from ldai.models import LDMessage, ModelConfig
from ldclient import Context

from ldai_optimization.dataclasses import (
    AIJudgeCallConfig,
    GroundTruthOptimizationOptions,
    GroundTruthSample,
    JudgeResult,
    OptimizationContext,
    OptimizationFromConfigOptions,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
    OptimizationResponse,
    ToolDefinition,
)
from ldai_optimization.ld_api_client import (
    AgentOptimizationConfig,
    AgentOptimizationResultPatch,
    AgentOptimizationResultPost,
    LDApiClient,
)
from ldai_optimization.prompts import (
    _acceptance_criteria_implies_duration_optimization,
    build_message_history_text,
    build_new_variation_prompt,
    build_reasoning_history,
)
from ldai_optimization.util import (
    await_if_needed,
    extract_json_from_response,
    interpolate_variables,
    restore_variable_placeholders,
)

logger = logging.getLogger(__name__)


def _find_model_config(
    model_name: str, configs: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find the best matching model config for a given model name.

    When multiple configs share the same ``id``, the one marked ``global=True``
    is preferred over project-specific configs. Falls back to the first
    non-global match if no global entry exists.

    :param model_name: The model id to look up.
    :param configs: List of model config dicts from the LD API.
    :return: Best-matching model config dict, or None if no match.
    """
    matching = [mc for mc in configs if mc.get("id") == model_name]
    if not matching:
        return None
    global_match = next((mc for mc in matching if mc.get("global") is True), None)
    return global_match if global_match is not None else matching[0]


def _strip_provider_prefix(model: str) -> str:
    """Strip the provider prefix from a model identifier returned by the LD API.

    API model keys are formatted as "Provider.model-name" (e.g. "OpenAI.gpt-5",
    "Anthropic.claude-opus-4.6"). Only the part after the first period is needed
    by the underlying LLM clients. If no period is present the string is returned
    unchanged.

    :param model: Raw model string from the API.
    :return: Model name with provider prefix removed.
    """
    return model.split(".", 1)[-1]


def _compute_validation_count(pool_size: int) -> int:
    """Compute how many validation samples to run after a candidate passes in chaos mode.

    Scales with the size of the available input/variable pool so that larger
    option sets receive proportionally more validation coverage, capped at 5.
    The floor of 2 ensures at least a minimal cross-check even for small pools.

    :param pool_size: Total number of distinct choices in the sampling pool
        (user_input_options count when provided, otherwise variable_choices count).
    :return: Number of validation samples to run (between 2 and 5 inclusive).
    """
    return min(5, max(2, pool_size // 4))


# Maximum number of attempts for variation generation. Transient empty or
# unparseable responses from the LLM are retried up to this many times before
# the variation step is treated as a failure.
_MAX_VARIATION_RETRIES = 3

# Duration gate: a candidate must be at least this much faster than the baseline
# (history[0].duration_ms) to pass the duration check when acceptance criteria
# imply a latency optimization goal. 0.80 means the candidate must clock in at
# under 80% of the baseline — i.e. at least 20% improvement.
_DURATION_TOLERANCE = 0.80

# Maps SDK status strings to the API status/activity values expected by
# agent_optimization_result records. Defined at module level to avoid
# allocating the dict on every on_status_update invocation.
_OPTIMIZATION_STATUS_MAP: Dict[str, Dict[str, str]] = {
    "init": {"status": "RUNNING", "activity": "PENDING"},
    "generating": {"status": "RUNNING", "activity": "GENERATING"},
    "evaluating": {"status": "RUNNING", "activity": "EVALUATING"},
    "generating variation": {"status": "RUNNING", "activity": "GENERATING_VARIATION"},
    "validating": {"status": "RUNNING", "activity": "EVALUATING"},
    "turn completed": {"status": "RUNNING", "activity": "COMPLETED"},
    "success": {"status": "PASSED", "activity": "COMPLETED"},
    "failure": {"status": "FAILED", "activity": "COMPLETED"},
}


class OptimizationClient:
    _options: OptimizationOptions
    _ldClient: LDAIClient
    _agent_config: AIAgentConfig
    _has_api_key: bool
    _api_key: Optional[str]
    _agent_key: str
    _initial_instructions: str

    def __init__(self, ldClient: LDAIClient) -> None:
        self._ldClient = ldClient
        self._last_run_succeeded: bool = False
        self._last_succeeded_context: Optional[OptimizationContext] = None
        self._last_optimization_result_id: Optional[str] = None
        self._initial_tool_keys: List[str] = []

        if os.environ.get("LAUNCHDARKLY_API_KEY"):
            self._has_api_key = True
            self._api_key = os.environ.get("LAUNCHDARKLY_API_KEY")
        else:
            self._has_api_key = False
            self._api_key = None
            logger.warning(
                "LAUNCHDARKLY_API_KEY is not set, functionality will be limited"
            )

    def _initialize_class_members_from_config(
        self, agent_config: AIAgentConfig
    ) -> None:
        if not agent_config.instructions:
            raise ValueError(
                f"Agent '{agent_config.key}' has no instructions configured. "
                "Ensure the agent flag has instructions set before running an optimization."
            )
        self._current_instructions = agent_config.instructions
        self._current_parameters: Dict[str, Any] = (
            agent_config.model._parameters if agent_config.model else None
        ) or {}
        self._current_model: Optional[str] = (
            agent_config.model.name if agent_config.model else None
        )
        self._history: List[OptimizationContext] = []

    def _build_agent_config_for_context(
        self, ctx: OptimizationContext
    ) -> AIAgentConfig:
        """
        Construct an AIAgentConfig that reflects the current optimization iteration.

        Uses the instructions, model, and parameters from the given context so the
        caller receives the variation being evaluated rather than the original base config.
        ``{{placeholder}}`` tokens in the instructions are substituted using
        ctx.current_variables at call time so the stored template is never mutated.

        :param ctx: The OptimizationContext for this iteration
        :return: A fresh AIAgentConfig populated from the context's current state
        """
        instructions = (
            interpolate_variables(ctx.current_instructions, ctx.current_variables)
            if ctx.current_variables
            else ctx.current_instructions
        )
        return AIAgentConfig(
            key=self._agent_key,
            enabled=True,
            model=ModelConfig(
                name=ctx.current_model or "",
                parameters=ctx.current_parameters,
            ),
            instructions=instructions,
            provider=self._agent_config.provider,
        )

    def _create_optimization_context(
        self,
        iteration: int,
        variables: Dict[str, Any],
        user_input: Optional[str] = None,
        completion_response: str = "",
        scores: Optional[Dict[str, JudgeResult]] = None,
    ) -> OptimizationContext:
        """
        Create an OptimizeContext with current state.

        :param iteration: Current iteration number
        :param variables: Variable set chosen for this iteration
        :param user_input: Optional user input for this iteration
        :param completion_response: Completion response string
        :param scores: Optional dictionary of judge results
        :return: A new OptimizeContext instance
        """
        flat_history = [prev_ctx.copy_without_history() for prev_ctx in self._history]
        return OptimizationContext(
            scores=scores or {},
            completion_response=completion_response,
            current_instructions=self._current_instructions,
            current_parameters=self._current_parameters.copy(),
            current_variables=variables,
            current_model=self._current_model,
            user_input=user_input,
            history=tuple(flat_history),
            iteration=iteration,
        )

    def _safe_status_update(
        self,
        status: Literal[
            "init",
            "generating",
            "evaluating",
            "generating variation",
            "validating",
            "turn completed",
            "success",
            "failure",
        ],
        context: OptimizationContext,
        iteration: int,
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
            except Exception:
                logger.exception(
                    "[Iteration %d] -> on_status_update callback failed", iteration
                )

    def _judge_config(
        self,
        judge_key: str,
        context: Context,
        default: AIJudgeConfigDefault,
        variables: Dict[str, Any],
    ) -> AIJudgeConfig:
        """
        Fetch a judge configuration from the LaunchDarkly client.

        Thin wrapper around LDAIClient.judge_config so callers do not need a
        direct reference to the client.

        :param judge_key: The key for the judge configuration in LaunchDarkly
        :param context: The evaluation context
        :param default: Fallback config when the flag is disabled or unreachable
        :param variables: Template variables for instruction interpolation
        :return: The resolved AIJudgeConfig
        """
        return self._ldClient.judge_config(judge_key, context, default, variables)

    def _serialize_scores(
        self, judge_results: Dict[str, JudgeResult]
    ) -> Dict[str, Any]:
        """
        Convert judge results to a JSON-serializable dictionary.

        :param judge_results: Dictionary of judge keys to JudgeResult instances
        :return: Dictionary suitable for json.dumps
        """
        return {key: result.to_json() for key, result in judge_results.items()}

    def _extract_agent_tools(self, parameters: Dict[str, Any]) -> List[ToolDefinition]:
        """
        Extract and normalise the tools list from agent parameters.

        Reads the ``tools`` key from *parameters* (if present) and converts
        every entry to a ToolDefinition so judges receive typed objects.

        :param parameters: The agent's current_parameters dict
        :return: List of ToolDefinition instances, empty list if no tools are configured
        """
        raw_tools = parameters.get("tools", [])
        if not raw_tools:
            return []
        if not isinstance(raw_tools, list):
            raw_tools = [raw_tools]

        result = []
        for tool in raw_tools:
            if isinstance(tool, ToolDefinition):
                result.append(tool)
            elif hasattr(tool, "to_dict"):
                result.append(ToolDefinition.from_dict(tool.to_dict()))
            elif isinstance(tool, dict):
                result.append(ToolDefinition.from_dict(tool))
        return result

    def _parse_judge_response(
        self,
        response_str: str,
        judge_key: str,
        judge_identifier: str,
        iteration: int,
        clamp_score: bool = True,
    ) -> JudgeResult:
        """
        Parse a structured LLM judge response into a JudgeResult.

        Expects a JSON object with "score" (float) and optionally "rationale"
        (str). On any parsing failure, logs the exception and returns a zero score.

        :param response_str: Raw string response from the judge LLM
        :param judge_key: Key used to identify this judge in results dicts
        :param judge_identifier: Human-readable identifier for log messages
        :param iteration: Current iteration number for logging
        :param clamp_score: When True, clamps score to [0.0, 1.0]
        :return: Parsed JudgeResult, or a zero-score result on failure
        """
        try:
            response_data = extract_json_from_response(response_str)
            score = float(response_data.get("score", 0.0))
            if clamp_score:
                score = max(0.0, min(1.0, score))
            rationale = response_data.get("rationale")
            return JudgeResult(score=score, rationale=rationale)
        except Exception:
            logger.exception(
                "[Iteration %d] -> Failed to parse judge response for %s",
                iteration,
                judge_identifier,
            )
            return JudgeResult(score=0.0, rationale=None)


    async def _call_judges(
        self,
        completion_response: str,
        iteration: int,
        user_input: str,
        variables: Optional[Dict[str, Any]] = None,
        agent_tools: Optional[List[ToolDefinition]] = None,
        expected_response: Optional[str] = None,
        agent_duration_ms: Optional[float] = None,
    ) -> Dict[str, JudgeResult]:
        """
        Call all judges in parallel (auto-path).

        For judges with judge_key: Fetches judge config on-demand from LaunchDarkly SDK.
        For judges with acceptance_statement: Uses handle_judge_call callback.

        :param completion_response: The agent's completion response to evaluate
        :param iteration: Current iteration number
        :param user_input: The user's question for this turn, forwarded to judges so
            they know what was actually asked (the current turn is not yet in
            self._history when judges run)
        :param variables: The variable set that was used during the agent generation
        :param agent_tools: Normalised list of tool dicts that were available to the agent
        :param expected_response: Optional ground truth expected response. When provided,
            judges are instructed to factor it into their scoring alongside acceptance criteria.
        :param agent_duration_ms: Wall-clock duration of the agent call in milliseconds.
            Forwarded to acceptance judges whose statement implies a latency goal so they
            can mention the duration change in their rationale.
        :return: Dictionary of judge results (score and rationale)
        """
        if not self._options.judges:
            return {}

        resolved_variables: Dict[str, Any] = variables or {}
        resolved_agent_tools: List[ToolDefinition] = agent_tools or []

        logger.info("[Iteration %d] -> Executing evaluation...", iteration)
        reasoning_history = build_reasoning_history(self._history)
        judge_results: Dict[str, JudgeResult] = {}

        judge_count = len(self._options.judges)
        for idx, (judge_key, optimization_judge) in enumerate(
            self._options.judges.items(), 1
        ):
            judge_type = (
                "config" if optimization_judge.judge_key is not None else "acceptance"
            )
            logger.info(
                "[Iteration %d] -> Running judge %d/%d '%s' (%s)...",
                iteration,
                idx,
                judge_count,
                judge_key,
                judge_type,
            )
            try:
                if optimization_judge.judge_key is not None:
                    result = await self._evaluate_config_judge(
                        judge_key,
                        optimization_judge,
                        completion_response,
                        iteration,
                        reasoning_history,
                        user_input=user_input,
                        variables=resolved_variables,
                        agent_tools=resolved_agent_tools,
                        expected_response=expected_response,
                    )
                    judge_results[judge_key] = result
                else:
                    result = await self._evaluate_acceptance_judge(
                        judge_key,
                        optimization_judge,
                        completion_response,
                        iteration,
                        reasoning_history,
                        user_input=user_input,
                        variables=resolved_variables,
                        agent_tools=resolved_agent_tools,
                        expected_response=expected_response,
                        agent_duration_ms=agent_duration_ms,
                    )
                    judge_results[judge_key] = result

                threshold = (
                    optimization_judge.threshold
                    if optimization_judge.threshold is not None
                    else 1.0
                )
                passed = result.score >= threshold
                logger.debug(
                    "[Iteration %d] -> Judge '%s' scored %.3f (threshold=%.3f) -> %s%s",
                    iteration,
                    judge_key,
                    result.score,
                    threshold,
                    "PASSED" if passed else "FAILED",
                    f" | {result.rationale}" if result.rationale else "",
                )
            except Exception:
                logger.exception(
                    "[Iteration %d] -> Judge %s evaluation failed", iteration, judge_key
                )
                judge_results[judge_key] = JudgeResult(score=0.0, rationale=None)

        judge_results_json = self._serialize_scores(judge_results)
        logger.debug(
            "[Iteration %d] -> Evaluation result: %s",
            iteration,
            json.dumps(judge_results_json, indent=2),
        )
        return judge_results

    async def _evaluate_config_judge(
        self,
        judge_key: str,
        optimization_judge: "OptimizationJudge",
        completion_response: str,
        iteration: int,
        reasoning_history: str,
        user_input: str,
        variables: Optional[Dict[str, Any]] = None,
        agent_tools: Optional[List[ToolDefinition]] = None,
        expected_response: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluate using a config-type judge (with judge_key).

        :param judge_key: The key for this judge in the judges dict
        :param optimization_judge: The optimization judge configuration
        :param completion_response: The agent's completion response to evaluate
        :param iteration: Current iteration number
        :param reasoning_history: Formatted string of reasoning from previous iterations
        :param user_input: The user's question for this turn
        :param variables: The variable set that was used during agent generation
        :param agent_tools: Normalised list of tool dicts that were available to the agent
        :param expected_response: Optional ground truth expected response. When provided,
            injected into template variables and judge messages.
        :return: The judge result with score and rationale
        """
        # Config-type judge: fetch judge config on-demand from LaunchDarkly SDK
        input_text = self._current_instructions or ""
        # Combine current instructions, history, and current question for message_history
        message_history_text = build_message_history_text(
            self._history, input_text, reasoning_history, user_input
        )

        # Merge agent variables so the judge's LD-managed instructions can reference
        # {{variable_name}} tokens alongside the standard judge template variables.
        template_variables: Dict[str, Any] = {
            **(variables or {}),
            "message_history": message_history_text,
            "response_to_evaluate": completion_response,
        }
        if expected_response is not None:
            template_variables["expected_response"] = expected_response

        assert optimization_judge.judge_key is not None
        judge_config = self._judge_config(
            optimization_judge.judge_key,
            self._options.context_choices[0],
            AIJudgeConfigDefault(enabled=False),
            template_variables,
        )

        if not judge_config.enabled:
            logger.warning(
                "[Iteration %d] -> Judge %s is disabled",
                iteration,
                optimization_judge.judge_key,
            )
            return JudgeResult(score=0.0, rationale=None)

        if not judge_config.messages:
            logger.warning(
                "[Iteration %d] -> Judge %s has no messages",
                iteration,
                optimization_judge.judge_key,
            )
            return JudgeResult(score=0.0, rationale=None)

        # Split messages into system and user turns.
        # System turns are joined into a single instructions string (agents SDK path).
        # All messages are forwarded as-is for the completions path.
        system_parts = []
        user_parts = []
        for msg in judge_config.messages:
            if msg.role == "system":
                system_parts.append(
                    msg.content
                    + " Return your response as a JSON object with 'score' and 'rationale' fields."
                )
            elif msg.role == "user":
                user_parts.append(msg.content)

        instructions = "\n\n".join(system_parts)
        judge_user_input = (
            "\n\n".join(user_parts)
            if user_parts
            else f"Here is the response to evaluate: {completion_response}"
        )

        if expected_response is not None:
            judge_user_input += (
                f"\n\nHere is the expected response: {expected_response}"
                "\n\nEvaluate the actual response against both the acceptance criteria AND "
                "how closely it matches the expected response. Factor both into your score."
            )

        # Rebuild the message list with the updated system content so completions users
        # receive the same scoring instructions that are baked into `instructions`.
        updated_messages: List[LDMessage] = [
            LDMessage(role="system", content=instructions),
            LDMessage(role="user", content=judge_user_input),
        ]

        # Collect model parameters from the judge config, separating out any existing tools
        model_name = (
            judge_config.model.name if judge_config.model else self._options.judge_model
        )
        model_params: Dict[str, Any] = {}
        tools: List[ToolDefinition] = []
        if judge_config.model and judge_config.model._parameters:
            existing_tools = judge_config.model._parameters.get("tools")
            if existing_tools:
                raw = (
                    existing_tools
                    if isinstance(existing_tools, list)
                    else [existing_tools]
                )
                for t in raw:
                    if isinstance(t, ToolDefinition):
                        tools.append(t)
                    elif hasattr(t, "to_dict"):
                        tools.append(ToolDefinition.from_dict(t.to_dict()))
                    elif isinstance(t, dict):
                        tools.append(ToolDefinition.from_dict(t))
            model_params = {
                k: v for k, v in judge_config.model._parameters.items() if k != "tools"
            }

        # Prepend agent tools so the judge can call them when verifying the response
        if agent_tools:
            tools = list(agent_tools) + tools

        tool_params = {"tools": [t.to_dict() for t in tools]} if tools else {}
        judge_call_config = AIJudgeCallConfig(
            key=judge_key,
            model=ModelConfig(
                name=model_name,
                parameters={**model_params, **tool_params},
            ),
            instructions=instructions,
            messages=updated_messages,
        )

        judge_ctx = OptimizationJudgeContext(
            user_input=judge_user_input,
            current_variables=variables or {},
        )

        _judge_start = time.monotonic()
        result = self._options.handle_judge_call(
            judge_key, judge_call_config, judge_ctx
        )
        judge_response: OptimizationResponse = await await_if_needed(result)
        judge_duration_ms = (time.monotonic() - _judge_start) * 1000
        judge_response_str = judge_response.output

        logger.debug(
            "[Iteration %d] -> Judge response (%s): %s",
            iteration,
            judge_key,
            judge_response_str,
        )

        # Parse judge response — expect structured JSON output
        judge_identifier = optimization_judge.judge_key or judge_key
        judge_result = self._parse_judge_response(
            judge_response_str,
            judge_key,
            judge_identifier,
            iteration,
            clamp_score=False,
        )
        return dataclasses.replace(judge_result, duration_ms=judge_duration_ms, usage=judge_response.usage)

    async def _evaluate_acceptance_judge(
        self,
        judge_key: str,
        optimization_judge: "OptimizationJudge",
        completion_response: str,
        iteration: int,
        reasoning_history: str,
        user_input: str,
        variables: Optional[Dict[str, Any]] = None,
        agent_tools: Optional[List[ToolDefinition]] = None,
        expected_response: Optional[str] = None,
        agent_duration_ms: Optional[float] = None,
    ) -> JudgeResult:
        """
        Evaluate using an acceptance statement judge.

        :param judge_key: The key for this judge in the judges dict
        :param optimization_judge: The optimization judge configuration
        :param completion_response: The agent's completion response to evaluate
        :param iteration: Current iteration number
        :param reasoning_history: Formatted string of reasoning from previous iterations
        :param user_input: The user's question for this turn
        :param variables: The variable set that was used during agent generation
        :param agent_tools: Normalised list of tool dicts that were available to the agent
        :param expected_response: Optional ground truth expected response. When provided,
            injected into instructions and judge message so the judge can score actual vs. expected.
        :param agent_duration_ms: Wall-clock duration of the agent call in milliseconds.
            When the acceptance statement implies a latency goal, the judge is instructed
            to mention the duration change in its rationale.
        :return: The judge result with score and rationale
        """
        if not optimization_judge.acceptance_statement:
            logger.error(
                "[Iteration %d] -> Judge %s has no acceptance_statement",
                iteration,
                judge_key,
            )
            return JudgeResult(score=0.0, rationale=None)

        resolved_variables = variables or {}
        resolved_agent_tools = agent_tools or []

        # Build message history including the current user question
        message_history_text = build_message_history_text(
            self._history, "", reasoning_history, user_input
        )

        # Build instructions for the judge
        instructions = (
            "You are a judge that evaluates the response to the user's question.\n\n"
            "Here is the statement that you should evaluate the response against: "
            f"'{optimization_judge.acceptance_statement}'\n"
            f"Here is the history of all messages between the user and the assistant: {message_history_text}\n"
            "You should score the response based on how well it meets the acceptance statement "
            "using a score between 0.0 and 1.0.\n"
            "A score of 0.0 means it does not match at all, while a score of 1.0 means it matches perfectly.\n"
            "A score of 0.3-0.7 means it matches partially, while a score of 0.7-1.0 means it matches well.\n"
            "A score of 0.0-0.3 means that it does not match well at all. "
            "You can return any value between 0.0 and 1.0.\n"
            "You should also provide a rationale for your score.\n"
            "Return your response as a JSON object with 'score' and 'rationale' fields.\n\n"
            'Example: {"score": 0.8, "rationale": "The response matches the acceptance statement well."}'
        )

        if (
            agent_duration_ms is not None
            and _acceptance_criteria_implies_duration_optimization(
                {judge_key: optimization_judge}
            )
        ):
            baseline_ms = (
                self._history[0].duration_ms
                if self._history and self._history[0].duration_ms is not None
                else None
            )
            instructions += (
                f"\n\nThe acceptance criteria for this judge includes a latency/duration goal. "
                f"The agent's response took {agent_duration_ms:.0f}ms to generate. "
            )
            if baseline_ms is not None:
                delta_ms = agent_duration_ms - baseline_ms
                direction = "faster" if delta_ms < 0 else "slower"
                instructions += (
                    f"The baseline duration (first iteration) was {baseline_ms:.0f}ms. "
                    f"This response was {abs(delta_ms):.0f}ms {direction} than the baseline. "
                )
            instructions += (
                "Please mention the duration and any change from baseline in your rationale."
            )

        if resolved_variables:
            instructions += f"\n\nThe following variables were available to the agent: {json.dumps(resolved_variables)}"

        if resolved_agent_tools:
            tool_names = [t.name for t in resolved_agent_tools]
            instructions += (
                "\n\nThe following tools were available to the agent and "
                f"may be called by you to verify the response: {json.dumps(tool_names)}."
                "\nIf verifying the response requires looking up external information, "
                "call the appropriate tool before scoring. "
                "You should only call the tools for the most recent response, "
                "and should only call the tools if necessary. "
                "Assume that previous feedback will have addressed bad tool call results from prior iterations."
            )

        # Agent tools are passed through so the judge can invoke them for verification
        tools: List[ToolDefinition] = list(resolved_agent_tools)

        judge_user_input = f"Here is the response to evaluate: {completion_response}"
        if expected_response is not None:
            judge_user_input += (
                f"\n\nHere is the expected response: {expected_response}"
                "\n\nEvaluate the actual response against both the acceptance statement AND "
                "how closely it matches the expected response. Factor both into your score."
            )

        tool_params = {"tools": [t.to_dict() for t in tools]} if tools else {}
        judge_call_config = AIJudgeCallConfig(
            key=judge_key,
            model=ModelConfig(
                name=self._options.judge_model,
                parameters=tool_params,
            ),
            instructions=instructions,
            messages=[
                LDMessage(role="system", content=instructions),
                LDMessage(role="user", content=judge_user_input),
            ],
        )

        judge_ctx = OptimizationJudgeContext(
            user_input=judge_user_input,
            current_variables=resolved_variables,
        )

        _judge_start = time.monotonic()
        result = self._options.handle_judge_call(
            judge_key, judge_call_config, judge_ctx
        )
        judge_response: OptimizationResponse = await await_if_needed(result)
        judge_duration_ms = (time.monotonic() - _judge_start) * 1000
        judge_response_str = judge_response.output

        logger.debug(
            "[Iteration %d] -> Judge response (%s): %s",
            iteration,
            judge_key,
            judge_response_str,
        )

        # Parse judge response — expect structured JSON output with score and rationale
        judge_result = self._parse_judge_response(
            judge_response_str, judge_key, judge_key, iteration, clamp_score=True
        )
        return dataclasses.replace(judge_result, duration_ms=judge_duration_ms, usage=judge_response.usage)

    async def _get_agent_config(
        self, agent_key: str, context: Context
    ) -> AIAgentConfig:
        """
        Fetch the agent configuration, replacing the instructions with the raw variation
        template so that {{placeholder}} tokens are preserved for client-side interpolation.

        agent_config() is called normally so we get a fully populated AIAgentConfig
        (including the tracker). We then call variation() separately to retrieve the
        unrendered instruction template and swap it in, keeping everything else intact.

        :param agent_key: The key for the agent to get the configuration for
        :param context: The evaluation context
        :return: AIAgentConfig with raw {{placeholder}} instruction templates intact
        """
        try:
            agent_config = self._ldClient.agent_config(agent_key, context)

            # variation() returns the raw JSON before chevron.render(), so instructions
            # still contain {{placeholder}} tokens rather than empty strings.
            raw_variation = self._ldClient._client.variation(agent_key, context, {})
            raw_instructions = raw_variation.get(
                "instructions", agent_config.instructions
            )
            if not raw_instructions:
                raise ValueError(
                    f"Agent '{agent_key}' has no instructions configured. "
                    "Ensure the agent flag has instructions set before running an optimization."
                )
            self._initial_instructions = raw_instructions

            raw_tools = raw_variation.get("tools", [])
            self._initial_tool_keys = [
                t["key"]
                for t in raw_tools
                if isinstance(t, dict) and "key" in t
            ]

            agent_config = dataclasses.replace(
                agent_config, instructions=raw_instructions
            )
            self._initialize_class_members_from_config(agent_config)
            return agent_config
        except Exception:
            logger.exception("[Optimization] -> Failed to get agent configuration")
            raise

    async def optimize_from_options(
        self, agent_key: str, options: OptimizationOptions
    ) -> Any:
        """Execute an optimization on the given agent with the given options.

        :param agent_key: Identifier of the agent to optimize.
        :param options: Optimization options.
        :return: Optimization result.
        """
        if options.auto_commit:
            if not self._has_api_key:
                raise ValueError(
                    "auto_commit requires LAUNCHDARKLY_API_KEY to be set"
                )
            if not options.project_key:
                raise ValueError(
                    "auto_commit requires project_key to be set on OptimizationOptions"
                )
        self._agent_key = agent_key
        context = random.choice(options.context_choices)
        agent_config = await self._get_agent_config(agent_key, context)
        result = await self._run_optimization(agent_config, options)
        if options.auto_commit and self._last_run_succeeded and self._last_succeeded_context:
            self._commit_variation(
                self._last_succeeded_context,
                project_key=options.project_key,  # type: ignore[arg-type]
                ai_config_key=agent_key,
                output_key=options.output_key,
                base_url=options.base_url,
            )
        return result

    async def optimize_from_ground_truth_options(
        self, agent_key: str, options: GroundTruthOptimizationOptions
    ) -> List[OptimizationContext]:
        """Execute a ground truth optimization on the given agent.

        Unlike optimize_from_options (which tests random choices until one passes),
        this path evaluates all N ground truth samples in each attempt and only
        succeeds when every sample passes its judges. A new variation is generated
        whenever any sample fails, and all N samples are re-evaluated from scratch
        with the updated configuration, up to max_attempts.

        :param agent_key: Identifier of the agent to optimize.
        :param options: Ground truth optimization options including the ordered sample list.
        :return: List of OptimizationContexts from the final attempt (one per sample).
        """
        if options.auto_commit:
            if not self._has_api_key:
                raise ValueError(
                    "auto_commit requires LAUNCHDARKLY_API_KEY to be set"
                )
            if not options.project_key:
                raise ValueError(
                    "auto_commit requires project_key to be set on GroundTruthOptimizationOptions"
                )
        self._agent_key = agent_key
        context = random.choice(options.context_choices)
        agent_config = await self._get_agent_config(agent_key, context)
        result = await self._run_ground_truth_optimization(agent_config, options)
        if options.auto_commit and self._last_run_succeeded and self._last_succeeded_context:
            self._commit_variation(
                self._last_succeeded_context,
                project_key=options.project_key,  # type: ignore[arg-type]
                ai_config_key=agent_key,
                output_key=options.output_key,
                base_url=options.base_url,
            )
        return result

    async def _run_ground_truth_optimization(
        self,
        agent_config: AIAgentConfig,
        gt_options: GroundTruthOptimizationOptions,
    ) -> List[OptimizationContext]:
        """Run the ground truth optimization loop.

        Uses the "bridge" pattern to reuse existing internal methods (judge evaluation,
        variation generation, status callbacks) for the ground truth optimization.

        :param agent_config: Agent configuration from LaunchDarkly.
        :param gt_options: Ground truth options supplied by the caller.
        :return: List of OptimizationContexts from the final attempt (one per sample).
        """
        bridge = OptimizationOptions(
            context_choices=gt_options.context_choices,
            max_attempts=gt_options.max_attempts,
            model_choices=gt_options.model_choices,
            judge_model=gt_options.judge_model,
            variable_choices=[s.variables for s in gt_options.ground_truth_responses],
            handle_agent_call=gt_options.handle_agent_call,
            handle_judge_call=gt_options.handle_judge_call,
            judges=gt_options.judges,
            on_turn=gt_options.on_turn,
            on_passing_result=gt_options.on_passing_result,
            on_failing_result=gt_options.on_failing_result,
            on_status_update=gt_options.on_status_update,
        )
        self._options = bridge
        self._agent_config = agent_config
        self._last_run_succeeded = False
        self._last_succeeded_context = None
        self._last_optimization_result_id = None
        self._initialize_class_members_from_config(agent_config)

        # Seed from the first model choice on the first iteration
        # so agent calls never receive an empty model string.
        if not self._current_model and bridge.model_choices:
            self._current_model = bridge.model_choices[0]
            logger.debug(
                "[GT] -> No model in agent config; defaulting to first model choice: %s",
                self._current_model,
            )

        samples = gt_options.ground_truth_responses
        n = len(samples)

        initial_context = self._create_optimization_context(
            iteration=0,
            variables=samples[0].variables,
        )
        self._safe_status_update("init", initial_context, 0)

        # Attempt tracks the current "batch" loop that runs
        # through all N samples. Iteration in this context refers to the
        # total number of batch runs so far.
        attempt = 0
        while True:
            attempt += 1
            logger.info(
                "[GT Attempt %d/%d] -> Starting ground truth run (%d samples, model=%s)",
                attempt,
                gt_options.max_attempts,
                n,
                self._current_model,
            )

            attempt_results: List[OptimizationContext] = []
            all_passed = True
            failed_count = 0

            # Now iterate through each individual sample in the batch,
            # creating a new context for each sample + running judges etc.
            for i, sample in enumerate(samples):
                linear_iter = (attempt - 1) * n + i + 1
                truncated = len(sample.user_input) > 100
                logger.info(
                    "[GT Attempt %d] -> Sample %d/%d (user_input=%.100s%s)",
                    attempt,
                    i + 1,
                    n,
                    sample.user_input,
                    "..." if truncated else "",
                )

                optimize_context = self._create_optimization_context(
                    iteration=linear_iter,
                    user_input=sample.user_input,
                    variables=sample.variables,
                )

                self._safe_status_update("generating", optimize_context, linear_iter)
                optimize_context = await self._execute_agent_turn(
                    optimize_context,
                    linear_iter,
                    expected_response=sample.expected_response,
                )

                # Per-sample pass/fail check
                if self._options.on_turn is not None:
                    try:
                        sample_passed = self._options.on_turn(optimize_context)
                    except Exception:
                        logger.exception(
                            "[GT Attempt %d] -> Sample %d on_turn evaluation failed",
                            attempt,
                            i + 1,
                        )
                        sample_passed = False
                else:
                    sample_passed = self._evaluate_response(optimize_context)

                if sample_passed and _acceptance_criteria_implies_duration_optimization(
                    self._options.judges
                ):
                    sample_passed = self._evaluate_duration(optimize_context)

                if not sample_passed:
                    logger.info(
                        "[GT Attempt %d] -> Sample %d/%d FAILED",
                        attempt,
                        i + 1,
                        n,
                    )
                    all_passed = False
                    failed_count += 1
                else:
                    logger.debug(
                        "[GT Attempt %d] -> Sample %d/%d passed",
                        attempt,
                        i + 1,
                        n,
                    )

                attempt_results.append(optimize_context)

                if gt_options.on_sample_result is not None:
                    try:
                        gt_options.on_sample_result(optimize_context)
                    except Exception:
                        logger.exception(
                            "[GT Attempt %d] -> on_sample_result callback failed for sample %d",
                            attempt,
                            i + 1,
                        )

            last_ctx = attempt_results[-1]

            if all_passed:
                logger.info(
                    "[GT Attempt %d] -> All %d samples passed — optimization succeeded",
                    attempt,
                    n,
                )
                self._last_run_succeeded = True
                self._last_succeeded_context = last_ctx
                self._safe_status_update("success", last_ctx, last_ctx.iteration)
                if self._options.on_passing_result:
                    try:
                        self._options.on_passing_result(last_ctx)
                    except Exception:
                        logger.exception(
                            "[GT Attempt %d] -> on_passing_result callback failed", attempt
                        )
                return attempt_results

            # We've hit max attempts for the batches, bail at this point
            if attempt >= gt_options.max_attempts:
                logger.warning(
                    "[GT Optimization] -> Failed after %d attempt(s) — not all samples passed",
                    attempt,
                )
                self._last_run_succeeded = False
                self._last_succeeded_context = None
                self._safe_status_update("failure", last_ctx, last_ctx.iteration)
                if self._options.on_failing_result:
                    try:
                        self._options.on_failing_result(last_ctx)
                    except Exception:
                        logger.exception(
                            "[GT Attempt %d] -> on_failing_result callback failed", attempt
                        )
                return attempt_results

            # Append all N results to history so the variation generator has full context
            # from all of the previous samples
            self._history.extend(attempt_results)

            logger.info(
                "[GT Attempt %d] -> %d/%d samples failed — generating new variation",
                attempt,
                failed_count,
                n,
            )
            try:
                await self._generate_new_variation(last_ctx.iteration, last_ctx.current_variables)
            except Exception:
                logger.exception(
                    "[GT Attempt %d] -> Variation generation failed", attempt
                )
                self._last_run_succeeded = False
                self._last_succeeded_context = None
                self._safe_status_update("failure", last_ctx, last_ctx.iteration)
                if self._options.on_failing_result:
                    try:
                        self._options.on_failing_result(last_ctx)
                    except Exception:
                        logger.exception(
                            "[GT Attempt %d] -> on_failing_result callback failed", attempt
                        )
                return attempt_results

            self._safe_status_update("turn completed", last_ctx, last_ctx.iteration)

        # Every branch inside the while True loop returns explicitly (success, max-attempts
        # exhaustion, or variation-generation failure). This line is structurally unreachable,
        # but without it type checkers infer the return type as List[OptimizationContext] | None
        # because they don't always treat `while True` as exhaustive. The RuntimeError makes
        # the intent unambiguous and causes a loud failure if that invariant is ever broken.
        raise RuntimeError("unreachable: ground truth loop exited without returning")

    def _apply_new_variation_response(
        self,
        response_data: Dict[str, Any],
        variation_ctx: OptimizationContext,
        response_str: str,
        iteration: int,
    ) -> OptimizationContext:
        """
        Validate the parsed variation response, mutate instance state, and return
        an updated OptimizationContext reflecting the new configuration.

        Updates self._current_instructions, self._current_parameters, and
        self._current_model in place so subsequent turns use the new configuration.

        :param response_data: Parsed JSON dict from the LLM variation response
        :param variation_ctx: The context that was sent to the LLM (used to carry history/iteration)
        :param response_str: The raw response string (stored as completion_response)
        :param iteration: Current iteration number for logging
        :return: A new OptimizationContext populated with the updated configuration
        """
        missing_fields = []
        if "current_instructions" not in response_data:
            missing_fields.append("current_instructions")
        if "current_parameters" not in response_data:
            missing_fields.append("current_parameters")
        if "model" not in response_data:
            missing_fields.append("model")

        if missing_fields:
            logger.debug(
                "[Iteration %d] -> Response missing required fields: %s. Received fields: %s. Full response_data: %s",
                iteration,
                ", ".join(missing_fields),
                list(response_data.keys()),
                json.dumps(response_data, indent=2),
            )
            raise ValueError(
                f"Response missing required fields: {', '.join(missing_fields)}. "
                f"Received fields: {list(response_data.keys())}"
            )

        self._current_instructions = response_data["current_instructions"]

        # Post-process: replace any leaked variable values back to {{key}} form.
        # This is a deterministic safety net for when the LLM ignores the prompt
        # instructions and hardcodes a concrete value (e.g. "user-123") instead
        # of the placeholder ("{{user_id}}").
        self._current_instructions, placeholder_warnings = restore_variable_placeholders(
            self._current_instructions,
            self._options.variable_choices,
        )
        for msg in placeholder_warnings:
            logger.warning("[Iteration %d] -> %s", iteration, msg)

        self._current_parameters = response_data["current_parameters"]

        # Update model — it should always be provided since it's required in the schema
        model_value = (
            response_data.get("model", "").strip()
            if isinstance(response_data.get("model"), str)
            else response_data.get("model")
        )
        if not model_value:
            logger.warning(
                "[Iteration %d] -> Model field is empty or None in response, keeping current model %s",
                iteration,
                self._current_model,
            )
        elif model_value not in self._options.model_choices:
            logger.warning(
                "[Iteration %d] -> Model '%s' not in model_choices %s, keeping current model %s",
                iteration,
                model_value,
                self._options.model_choices,
                self._current_model,
            )
        else:
            old_model = self._current_model
            self._current_model = model_value

            # Log regardless of whether we change the model so that logs
            # are consistently structured
            if old_model != self._current_model:
                logger.info(
                    "[Iteration %d] -> Model updated from '%s' to '%s'",
                    iteration,
                    old_model,
                    self._current_model,
                )
            else:
                logger.debug(
                    "[Iteration %d] -> Keeping model '%s'",
                    iteration,
                    self._current_model,
                )

        logger.debug(
            "[Iteration %d] -> New variation generated: instructions='%s', model=%s, parameters=%s",
            iteration,
            self._current_instructions,
            self._current_model,
            self._current_parameters,
        )

        # Create a new context with the updated values for return
        return OptimizationContext(
            scores={},
            completion_response=response_str,
            current_instructions=self._current_instructions,
            current_parameters=self._current_parameters.copy(),
            current_variables=variation_ctx.current_variables,
            current_model=self._current_model,
            user_input=None,
            history=variation_ctx.history,
            iteration=variation_ctx.iteration,
        )

    async def _generate_new_variation(
        self, iteration: int, variables: Dict[str, Any]
    ) -> OptimizationContext:
        """
        Generate new variation for next iteration (auto-path).

        Calls handle_agent_call to generate a new variation and updates current_instructions
        and current_parameters based on the returned OptimizeContext.

        :param iteration: The current iteration number for logging
        :param variables: The variable set for this iteration, chosen once by the caller
        """
        logger.info("[Iteration %d] -> Generating new variation...", iteration)

        # Create a context for status update before generating the variation
        status_ctx = self._create_optimization_context(
            iteration=iteration,
            variables=variables,
        )
        self._safe_status_update("generating variation", status_ctx, iteration)

        optimize_for_duration = _acceptance_criteria_implies_duration_optimization(
            self._options.judges
        )
        instructions = build_new_variation_prompt(
            self._history,
            self._options.judges,
            self._current_model,
            self._current_instructions,
            self._current_parameters,
            self._options.model_choices,
            self._options.variable_choices,
            self._initial_instructions,
            optimize_for_duration=optimize_for_duration,
        )

        # Create a flat history list (without nested history) to avoid exponential growth
        flat_history = [prev_ctx.copy_without_history() for prev_ctx in self._history]

        # Create context for variation generation — low temperature for deterministic output.
        variation_ctx = OptimizationContext(
            scores={},
            completion_response="",
            current_instructions=instructions,
            current_parameters={
                "temperature": 0.1,
            },
            current_variables=variables,
            current_model=self._current_model,
            user_input=None,
            history=tuple(flat_history),
            iteration=len(self._history) + 1,
        )

        # Call handle_agent_call to generate new variation; expects a JSON string
        # matching the structured output schema (current_instructions, current_parameters, model).
        # Retry up to _MAX_VARIATION_RETRIES times to handle transient empty or unparseable
        # responses (e.g. when the agent SDK returns the LLM's post-tool-call empty text
        # instead of the tool result).
        agent_config = self._build_agent_config_for_context(variation_ctx)
        response_data = None
        response_str = ""
        for attempt in range(1, _MAX_VARIATION_RETRIES + 1):
            result = self._options.handle_agent_call(
                self._agent_key,
                agent_config,
                variation_ctx,
            )
            variation_response: OptimizationResponse = await await_if_needed(result)
            response_str = variation_response.output
            try:
                response_data = extract_json_from_response(response_str)
                break
            except ValueError:
                if attempt == _MAX_VARIATION_RETRIES:
                    raise
                logger.warning(
                    "[Iteration %d] -> Variation response empty or unparseable "
                    "(attempt %d/%d), retrying...",
                    iteration,
                    attempt,
                    _MAX_VARIATION_RETRIES,
                )

        assert response_data is not None  # loop always raises or breaks with data
        return self._apply_new_variation_response(
            response_data, variation_ctx, response_str, iteration
        )

    async def optimize_from_config(
        self, optimization_config_key: str, options: OptimizationFromConfigOptions
    ) -> Any:
        """Optimize an agent using a configuration fetched from the LaunchDarkly API.

        The agent key, judge configuration, model choices, and other optimization
        parameters are all sourced from the remote agent optimization config. The
        caller only needs to provide the execution callbacks and evaluation contexts.

        Iteration results are automatically persisted to the LaunchDarkly API so
        the UI can display live run progress.

        :param optimization_config_key: Key of the agent optimization config to fetch.
        :param options: User-provided callbacks and evaluation contexts.
        :return: Optimization result (OptimizationContext from the final iteration).
        """
        if not self._has_api_key:
            raise ValueError(
                "LAUNCHDARKLY_API_KEY is not set, so optimize_from_config is not available"
            )
        if options.auto_commit and not self._has_api_key:
            raise ValueError(
                "auto_commit requires LAUNCHDARKLY_API_KEY to be set"
            )

        assert self._api_key is not None
        api_client = LDApiClient(
            self._api_key,
            **({"base_url": options.base_url} if options.base_url else {}),
        )
        config = api_client.get_agent_optimization(options.project_key, optimization_config_key)

        self._agent_key = config["aiConfigKey"]
        optimization_key: str = config["key"]
        run_id = str(uuid.uuid4())

        model_configs: List[Dict[str, Any]] = []
        try:
            model_configs = api_client.get_model_configs(options.project_key)
        except Exception as exc:
            logger.debug("Could not pre-fetch model configs: %s", exc)

        context = random.choice(options.context_choices)
        # _get_agent_config calls _initialize_class_members_from_config internally;
        # _run_optimization calls it again to reset history before the loop starts.
        agent_config = await self._get_agent_config(self._agent_key, context)

        optimization_options = self._build_options_from_config(
            config, options, api_client, optimization_key, run_id, model_configs
        )
        if isinstance(optimization_options, GroundTruthOptimizationOptions):
            result = await self._run_ground_truth_optimization(agent_config, optimization_options)
        else:
            result = await self._run_optimization(agent_config, optimization_options)

        if options.auto_commit and self._last_run_succeeded and self._last_succeeded_context:
            created_key = self._commit_variation(
                self._last_succeeded_context,
                project_key=options.project_key,
                ai_config_key=config["aiConfigKey"],
                output_key=options.output_key,
                api_client=api_client,
                model_configs=model_configs,
            )
            if created_key and self._last_optimization_result_id:
                api_client.patch_agent_optimization_result(
                    options.project_key,
                    optimization_key,
                    self._last_optimization_result_id,
                    {"createdVariationKey": created_key},
                )
        return result

    def _build_options_from_config(
        self,
        config: AgentOptimizationConfig,
        options: OptimizationFromConfigOptions,
        api_client: LDApiClient,
        optimization_key: str,
        run_id: str,
        model_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> "Union[OptimizationOptions, GroundTruthOptimizationOptions]":
        """Map a fetched AgentOptimization config + user options into the appropriate options type.

        When the config contains groundTruthResponses, the three lists (groundTruthResponses,
        userInputOptions, variableChoices) are zipped by index into GroundTruthSample objects
        and a GroundTruthOptimizationOptions is returned. Otherwise a standard OptimizationOptions
        is returned.

        Acceptance statements and judge configs from the API are merged into a single
        judges dict. An on_status_update closure is injected to persist each iteration
        result to the LaunchDarkly API; any user-supplied on_status_update is chained
        after the persistence call.

        :param config: Validated AgentOptimizationConfig from the API.
        :param options: User-provided options from optimize_from_config.
        :param api_client: Initialised LDApiClient for result persistence.
        :param optimization_key: String key of the parent agent_optimization record.
        :param run_id: UUID that groups all result records for this run.
        :param model_configs: Pre-fetched list of model config dicts for resolving modelConfigKey.
        :return: OptimizationOptions or GroundTruthOptimizationOptions.
        """
        judges: Dict[str, OptimizationJudge] = {}

        for i, stmt in enumerate(config["acceptanceStatements"]):
            key = f"acceptance-statement-{i}"
            judges[key] = OptimizationJudge(
                threshold=float(stmt.get("threshold", 0.95)),
                acceptance_statement=stmt["statement"],
            )

        for judge in config["judges"]:
            judges[judge["key"]] = OptimizationJudge(
                threshold=float(judge.get("threshold", 0.95)),
                judge_key=judge["key"],
            )

        raw_ground_truth: List[str] = config.get("groundTruthResponses") or []
        has_ground_truth = bool(raw_ground_truth)
        if not judges and not has_ground_truth and options.on_turn is None:
            raise ValueError(
                "The optimization config has no acceptance statements, judges, or ground truth "
                "responses, and no on_turn callback was provided. At least one is required to "
                "evaluate optimization results."
            )

        project_key = options.project_key
        config_version: int = config["version"]
        _cached_model_configs: List[Dict[str, Any]] = list(model_configs or [])

        # Maps logical iteration number → result record id. Each new main-loop
        # iteration (plus the init iteration 0) POSTs a fresh record; subsequent
        # status events for that same iteration PATCH the existing record.
        _iteration_result_ids: Dict[int, str] = {}

        # Validation phase tracking. When a candidate passes initial checks the
        # SDK fires validation sub-iterations (val_iter = main_iter + 1, +2, …).
        # These are internal cross-checks and should NOT create separate records;
        # instead they are folded back into the parent main-loop iteration's record.
        _in_validation_phase: bool = False
        _validation_parent_iteration: int = -1

        # Tracks the most recently opened (POSTed) iteration so we can close it
        # with a RUNNING:COMPLETED patch when the next iteration begins. Without
        # this, iterations that don't naturally receive a terminal event (e.g. the
        # init iteration 0, or non-final GT samples) are left in a stale state.
        _last_open_iteration: int = -1

        def _resolve_model_config_key(model_name: str) -> str:
            if not model_name:
                return ""
            match = _find_model_config(model_name, _cached_model_configs)
            return match["key"] if match else model_name

        def _persist_and_forward(
            status: Literal[
                "init",
                "generating",
                "evaluating",
                "generating variation",
                "validating",
                "turn completed",
                "success",
                "failure",
            ],
            ctx: OptimizationContext,
        ) -> None:
            nonlocal _in_validation_phase, _validation_parent_iteration, _last_open_iteration
            # _safe_status_update (the caller) already wraps this entire function in
            # a try/except, so errors here are caught and logged without aborting the run.
            mapped = _OPTIMIZATION_STATUS_MAP.get(
                status, {"status": "RUNNING", "activity": "PENDING"}
            )
            snapshot = ctx.copy_without_history()

            # "validating" fires with the parent main-loop iteration's context, so
            # we capture that number as the anchor for all subsequent validation events.
            if status == "validating":
                _in_validation_phase = True
                _validation_parent_iteration = snapshot.iteration

            # Any event whose ctx.iteration differs from the validation anchor is a
            # validation sub-iteration; fold it back to the parent's record.
            if _in_validation_phase and snapshot.iteration != _validation_parent_iteration:
                logical_iteration = _validation_parent_iteration
            else:
                logical_iteration = snapshot.iteration

            # When a new iteration begins (generating), close out whatever iteration
            # was last open so it doesn't remain in a non-terminal state. This covers
            # the init iteration (0 → 1) and GT batches where non-final samples never
            # receive an explicit terminal event.
            if (
                status == "generating"
                and _last_open_iteration >= 0
                and logical_iteration != _last_open_iteration
            ):
                prev_result_id = _iteration_result_ids.get(_last_open_iteration)
                if prev_result_id:
                    api_client.patch_agent_optimization_result(
                        project_key,
                        optimization_key,
                        prev_result_id,
                        {"status": "RUNNING", "activity": "COMPLETED"},
                    )
                _last_open_iteration = -1

            # Phase 1: POST to create the record on first encounter of each logical iteration.
            if logical_iteration not in _iteration_result_ids:
                post_payload: AgentOptimizationResultPost = {
                    "runId": run_id,
                    "agentOptimizationVersion": config_version,
                    "iteration": logical_iteration,
                    "instructions": snapshot.current_instructions,
                }
                if snapshot.current_parameters:
                    post_payload["parameters"] = snapshot.current_parameters
                if snapshot.user_input:
                    post_payload["userInput"] = snapshot.user_input
                result_id = api_client.post_agent_optimization_result(
                    project_key, optimization_key, post_payload
                )
                if result_id:
                    _iteration_result_ids[logical_iteration] = result_id
                    self._last_optimization_result_id = result_id
                    _last_open_iteration = logical_iteration

            # Phase 2: PATCH the record with current status and available telemetry.
            result_id = _iteration_result_ids.get(logical_iteration)
            if result_id:
                patch: AgentOptimizationResultPatch = {
                    "status": mapped["status"],
                    "activity": mapped["activity"],
                }
                if snapshot.completion_response:
                    patch["completionResponse"] = snapshot.completion_response
                if snapshot.scores:
                    patch["scores"] = {
                        k: {
                            **v.to_json(),
                            **({"threshold": judges[k].threshold} if k in judges else {}),
                        }
                        for k, v in snapshot.scores.items()
                    }
                if snapshot.duration_ms is not None:
                    patch["generationLatency"] = int(snapshot.duration_ms)
                if snapshot.usage is not None:
                    patch["generationTokens"] = {
                        "total": snapshot.usage.total,
                        "input": snapshot.usage.input,
                        "output": snapshot.usage.output,
                    }
                eval_latencies = {
                    k: v.duration_ms
                    for k, v in snapshot.scores.items()
                    if v.duration_ms is not None
                }
                if eval_latencies:
                    patch["evaluationLatencies"] = eval_latencies
                eval_tokens = {
                    k: {"total": v.usage.total, "input": v.usage.input, "output": v.usage.output}
                    for k, v in snapshot.scores.items()
                    if v.usage is not None
                }
                if eval_tokens:
                    patch["evaluationTokens"] = eval_tokens
                patch["variation"] = {
                    "instructions": snapshot.current_instructions,
                    "parameters": snapshot.current_parameters,
                    "modelConfigKey": _resolve_model_config_key(snapshot.current_model or ""),
                }
                api_client.patch_agent_optimization_result(
                    project_key, optimization_key, result_id, patch
                )

            # Reset tracking state after terminal events so the next main-loop
            # attempt starts fresh.
            if status in ("turn completed", "success", "failure"):
                _in_validation_phase = False
                _validation_parent_iteration = -1
                _last_open_iteration = -1

            if options.on_status_update:
                try:
                    options.on_status_update(status, ctx)
                except Exception:
                    logger.exception("User on_status_update callback failed for status=%s", status)

        # If we have ground truth responses, we provide a different
        # configuration options type that contains the bundled GroundTruthSamples
        # so that the ultimate output is correctly formatted.
        if has_ground_truth:
            user_inputs: List[str] = config["userInputOptions"] or []
            variable_choices_raw: List[Dict[str, Any]] = config["variableChoices"] or []

            if len(raw_ground_truth) != len(user_inputs) or len(raw_ground_truth) != len(variable_choices_raw):
                raise ValueError(
                    f"groundTruthResponses ({len(raw_ground_truth)}), userInputOptions "
                    f"({len(user_inputs)}), and variableChoices ({len(variable_choices_raw)}) "
                    "must all have the same length when groundTruthResponses is provided."
                )

            gt_samples = [
                GroundTruthSample(
                    user_input=user_inputs[idx],
                    expected_response=raw_ground_truth[idx],
                    variables=variable_choices_raw[idx],
                )
                for idx in range(len(raw_ground_truth))
            ]

            return GroundTruthOptimizationOptions(
                context_choices=options.context_choices,
                ground_truth_responses=gt_samples,
                max_attempts=config["maxAttempts"],
                model_choices=[_strip_provider_prefix(m) for m in config["modelChoices"]],
                judge_model=_strip_provider_prefix(config["judgeModel"]),
                handle_agent_call=options.handle_agent_call,
                handle_judge_call=options.handle_judge_call,
                judges=judges or None,
                on_turn=options.on_turn,
                on_sample_result=options.on_sample_result,
                on_passing_result=options.on_passing_result,
                on_failing_result=options.on_failing_result,
                on_status_update=_persist_and_forward,
            )

        variable_choices: List[Dict[str, Any]] = config["variableChoices"] or [{}]
        user_input_options: Optional[List[str]] = config["userInputOptions"] or None

        return OptimizationOptions(
            context_choices=options.context_choices,
            max_attempts=config["maxAttempts"],
            model_choices=[_strip_provider_prefix(m) for m in config["modelChoices"]],
            judge_model=_strip_provider_prefix(config["judgeModel"]),
            variable_choices=variable_choices,
            handle_agent_call=options.handle_agent_call,
            handle_judge_call=options.handle_judge_call,
            judges=judges or None,
            user_input_options=user_input_options,
            on_turn=options.on_turn,
            on_passing_result=options.on_passing_result,
            on_failing_result=options.on_failing_result,
            on_status_update=_persist_and_forward,
        )

    async def _execute_agent_turn(
        self,
        optimize_context: OptimizationContext,
        iteration: int,
        expected_response: Optional[str] = None,
    ) -> OptimizationContext:
        """
        Run the agent call and judge scoring for one optimization turn.

        Returns a new OptimizationContext with completion_response and scores
        populated, leaving the input context unchanged. Variables are read from
        optimize_context.current_variables and interpolated into the agent's
        instructions at call time so the stored template is never mutated.

        :param optimize_context: The context for this turn (instructions, model, history, etc.)
        :param iteration: Current iteration number for logging and status callbacks
        :param expected_response: Optional ground truth expected response. When provided,
            injected into judge context so judges can score actual vs. expected.
        :return: Updated context with completion_response and scores filled in
        """
        logger.info(
            "[Iteration %d] -> Calling agent (model=%s)...",
            iteration,
            optimize_context.current_model,
        )
        try:
            _agent_start = time.monotonic()
            result = self._options.handle_agent_call(
                self._agent_key,
                self._build_agent_config_for_context(optimize_context),
                optimize_context,
            )
            agent_response: OptimizationResponse = await await_if_needed(result)
            agent_duration_ms = (time.monotonic() - _agent_start) * 1000
            completion_response = agent_response.output
            logger.debug(
                "[Iteration %d] -> Agent response: %.300s%s",
                iteration,
                completion_response,
                "..." if len(completion_response) > 300 else "",
            )
        except Exception:
            logger.exception("[Iteration %d] -> Agent call failed", iteration)
            if self._options.on_failing_result:
                self._options.on_failing_result(optimize_context)
            raise

        scores: Dict[str, JudgeResult] = {}
        if self._options.judges:
            agent_tools = self._extract_agent_tools(optimize_context.current_parameters)
            scores = await self._call_judges(
                completion_response,
                iteration,
                user_input=optimize_context.user_input or "",
                variables=optimize_context.current_variables,
                agent_tools=agent_tools,
                expected_response=expected_response,
                agent_duration_ms=agent_duration_ms,
            )

        # Build the fully-populated result context before firing the evaluating event so
        # the PATCH includes scores, generationLatency, and completionResponse. This is
        # particularly important for non-final GT samples which receive no further status
        # events — without this, those fields would never be written to their API records.
        result_ctx = dataclasses.replace(
            optimize_context,
            completion_response=completion_response,
            scores=scores,
            duration_ms=agent_duration_ms,
            usage=agent_response.usage,
        )

        if self._options.judges:
            self._safe_status_update("evaluating", result_ctx, iteration)

        return result_ctx

    def _evaluate_response(self, optimize_context: OptimizationContext) -> bool:
        """
        Determine whether the current iteration's scores meet all judge thresholds.

        A judge without an explicit threshold is treated as requiring a perfect
        score of 1.0. Returns True immediately when no judges are configured.

        :param optimize_context: The completed turn context containing scores
        :return: True if all judges passed, False if any judge failed or is missing
        """
        if not self._options.judges:
            return True

        for judge_key, optimization_judge in self._options.judges.items():
            result = optimize_context.scores.get(judge_key)
            if result is None:
                return False
            threshold = (
                optimization_judge.threshold
                if optimization_judge.threshold is not None
                else 1.0
            )
            if result.score < threshold:
                return False

        return True

    def _evaluate_duration(self, optimize_context: OptimizationContext) -> bool:
        """
        Check whether the candidate's duration meets the improvement target vs. the baseline.

        The baseline is history[0].duration_ms — the very first completed iteration,
        representing the original unoptimized configuration's latency. The candidate
        must be at least _DURATION_TOLERANCE faster (default: 20% improvement).

        Returns True without blocking when no baseline is available (empty history or
        history[0].duration_ms is None), or when the candidate's duration_ms was not
        captured. This avoids penalising configurations when timing data is missing.

        :param optimize_context: The completed turn context containing duration_ms
        :return: True if the duration requirement is met or cannot be checked
        """
        if not self._history or self._history[0].duration_ms is None:
            return True
        if optimize_context.duration_ms is None:
            return True
        baseline = self._history[0].duration_ms
        passed = optimize_context.duration_ms < baseline * _DURATION_TOLERANCE
        if not passed:
            logger.warning(
                "[Iteration %d] -> Duration check failed: %.0fms >= baseline %.0fms * %.0f%% (%.0fms)",
                optimize_context.iteration,
                optimize_context.duration_ms,
                baseline,
                _DURATION_TOLERANCE * 100,
                baseline * _DURATION_TOLERANCE,
            )
        return passed

    def _handle_success(
        self, optimize_context: OptimizationContext, iteration: int
    ) -> Any:
        """
        Handle a successful optimization result.

        Fires the "success" status update, invokes on_passing_result if set,
        and returns the winning OptimizationContext.

        :param optimize_context: The context from the passing iteration
        :param iteration: Current iteration number for logging
        :return: The passing OptimizationContext
        """
        logger.info("[Iteration %d] -> Optimization succeeded", iteration)
        self._last_run_succeeded = True
        self._last_succeeded_context = optimize_context
        self._safe_status_update("success", optimize_context, iteration)
        if self._options.on_passing_result:
            try:
                self._options.on_passing_result(optimize_context)
            except Exception:
                logger.exception(
                    "[Iteration %d] -> on_passing_result callback failed", iteration
                )
        return optimize_context

    def _handle_failure(
        self, optimize_context: OptimizationContext, iteration: int
    ) -> Any:
        """
        Handle a failed optimization result (max attempts reached).

        Fires the "failure" status update, invokes on_failing_result if set,
        and returns the last OptimizationContext.

        :param optimize_context: The context from the final iteration
        :param iteration: Current iteration number for logging
        :return: The last OptimizationContext
        """
        logger.warning(
            "[Optimization] -> Optimization failed after %d attempt(s)", iteration
        )
        self._last_run_succeeded = False
        self._last_succeeded_context = None
        self._safe_status_update("failure", optimize_context, iteration)
        if self._options.on_failing_result:
            try:
                self._options.on_failing_result(optimize_context)
            except Exception:
                logger.exception(
                    "[Iteration %d] -> on_failing_result callback failed", iteration
                )
        return optimize_context

    def _commit_variation(
        self,
        optimize_context: OptimizationContext,
        project_key: str,
        ai_config_key: str,
        output_key: Optional[str],
        api_client: Optional[LDApiClient] = None,
        base_url: Optional[str] = None,
        model_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Commit the winning optimization context as a new AI Config variation.

        Determines a unique variation key (from output_key or an auto-generated
        adjective-noun slug), checks for collisions against existing variation keys,
        appends a random hex suffix if the key is taken, then POSTs the new variation
        with up to 2 retries before raising on persistent failure.

        :param optimize_context: The winning OptimizationContext.
        :param project_key: LaunchDarkly project key.
        :param ai_config_key: The AI Config key to add the variation to.
        :param output_key: Desired variation key/name; auto-generated if None.
        :param api_client: Optional pre-built LDApiClient to reuse (e.g. from optimize_from_config).
        :param base_url: Optional base URL override forwarded to a newly created LDApiClient.
        :return: The created variation key.
        :raises LDApiError: If the variation cannot be created after retries.
        """
        if api_client is None:
            assert self._api_key is not None
            api_client = LDApiClient(
                self._api_key,
                **({"base_url": base_url} if base_url else {}),
            )

        candidate = output_key if output_key else generate_slug(2)

        try:
            ai_config = api_client.get_ai_config(project_key, ai_config_key)
            existing_keys = {v["key"] for v in ai_config.get("variations", [])}
        except Exception:
            logger.warning(
                "Could not fetch AI Config to check variation key collisions; proceeding with candidate key."
            )
            existing_keys = set()

        if candidate in existing_keys:
            suffix = "%04x" % random.randint(0, 0xFFFF)
            candidate = f"{candidate}-{suffix}"
            logger.info("Variation key collision detected; using '%s' instead.", candidate)

        model_name = optimize_context.current_model or ""
        model_config_key = model_name  # fallback if lookup fails
        try:
            configs_to_search = model_configs if model_configs is not None else api_client.get_model_configs(project_key)
            match = _find_model_config(model_name, configs_to_search)
            if match:
                model_config_key = match["key"]
            else:
                logger.debug(
                    "No model config found for model id '%s'; using model name as key.", model_name
                )
        except Exception as exc:
            logger.debug("Could not fetch model configs to resolve modelConfigKey: %s", exc)

        payload: Dict[str, Any] = {
            "key": candidate,
            "name": candidate,
            "mode": "agent",
            "instructions": optimize_context.current_instructions,
            "modelConfigKey": model_config_key,
        }
        if self._initial_tool_keys:
            payload["toolKeys"] = list(self._initial_tool_keys)

        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                api_client.create_ai_config_variation(project_key, ai_config_key, payload)
                logger.info(
                    "Auto-committed variation '%s' to AI Config '%s'.", candidate, ai_config_key
                )
                return candidate
            except Exception as exc:
                last_exc = exc
                if attempt < 3:
                    logger.warning(
                        "Failed to create variation (attempt %d/3): %s. Retrying...", attempt, exc
                    )

        raise last_exc  # type: ignore[misc]

    async def _run_validation_phase(
        self,
        passing_context: OptimizationContext,
        iteration: int,
    ) -> "tuple[bool, OptimizationContext]":
        """Run additional evaluations against distinct random samples to confirm a passing candidate.

        Mirrors the sampling logic of _run_optimization: each validation turn selects
        a user_input from user_input_options (when provided) AND a variables dict from
        variable_choices independently. The validation count and distinctness guarantee
        are driven by whichever pool is larger — user_input_options when present,
        otherwise variable_choices — ensuring validation turns use inputs the passing
        turn did not.

        If all samples pass, the caller should proceed to _handle_success. If any
        sample fails, the caller should treat the result as a normal failed attempt
        and generate a new variation.

        Validation turns are numbered sequentially in logs (iteration + 1, + 2, …)
        for readability, but this numbering is internal only — the caller's iteration
        counter is never advanced by this method so validation samples do not consume
        the attempt budget.

        :param passing_context: The OptimizationContext from the turn that just passed.
        :param iteration: The iteration number of the passing turn; used as the
            base for validation log line numbering only.
        :return: Tuple of (all_passed, last_context).
        """
        options = self._options

        # Determine the primary axis of distinctness and the pool size.
        # user_input_options drives the count when present; otherwise variable_choices does.
        # In either case, both user_input and variables are selected per-sample just as
        # they are in the main optimization loop.
        if options.user_input_options:
            primary_pool: List[str] = options.user_input_options
            passing_input: Optional[str] = passing_context.user_input
            remaining_inputs: List[str] = [
                inp for inp in primary_pool if inp != passing_input
            ]
            pool_size = len(primary_pool)
        else:
            var_pool: List[Dict[str, Any]] = options.variable_choices
            passing_vars: Dict[str, Any] = passing_context.current_variables
            remaining_vars: List[Dict[str, Any]] = [
                v for v in var_pool if v != passing_vars
            ]
            pool_size = len(var_pool)

        validation_count = _compute_validation_count(pool_size)
        # Cap to the number of distinct remaining items, but never below 1.
        # When the pool is exhausted (e.g. only one variable choice), sample
        # with replacement from the full pool so at least one validation run
        # always executes.
        if options.user_input_options:
            available = len(remaining_inputs)
        else:
            available = len(remaining_vars)

        allow_repeats = available == 0
        if allow_repeats:
            validation_count = 1
        else:
            validation_count = min(validation_count, available)

        logger.info(
            "[Iteration %d] -> Candidate passed — entering validation phase (%d sample(s)%s)",
            iteration,
            validation_count,
            ", repeated draw" if allow_repeats else "",
        )
        self._safe_status_update("validating", passing_context, iteration)

        # Sample primary items, falling back to the full pool when no distinct
        # items remain so the minimum-1 floor is always satisfied.
        if options.user_input_options:
            source_inputs = primary_pool if allow_repeats else remaining_inputs
            sampled_inputs: List[str] = random.sample(source_inputs, validation_count)
        else:
            source_vars = var_pool if allow_repeats else remaining_vars
            sampled_vars: List[Dict[str, Any]] = random.sample(source_vars, validation_count)

        last_ctx = passing_context
        for i in range(validation_count):
            val_iter = iteration + i + 1
            if options.user_input_options:
                user_input: Optional[str] = sampled_inputs[i]
                variables: Dict[str, Any] = random.choice(options.variable_choices)
            else:
                user_input = None
                variables = sampled_vars[i]

            logger.info(
                "[Validation %d/%d] -> Running sample (iteration=%d)",
                i + 1,
                validation_count,
                val_iter,
            )

            val_ctx = self._create_optimization_context(
                iteration=val_iter,
                user_input=user_input,
                variables=variables,
            )
            self._safe_status_update("generating", val_ctx, val_iter)
            val_ctx = await self._execute_agent_turn(val_ctx, val_iter)

            if options.on_turn is not None:
                try:
                    sample_passed = options.on_turn(val_ctx)
                except Exception:
                    logger.exception(
                        "[Validation %d/%d] -> on_turn evaluation failed", i + 1, validation_count
                    )
                    sample_passed = False
            else:
                sample_passed = self._evaluate_response(val_ctx)

            if sample_passed and _acceptance_criteria_implies_duration_optimization(
                self._options.judges
            ):
                sample_passed = self._evaluate_duration(val_ctx)

            last_ctx = val_ctx

            if not sample_passed:
                logger.info(
                    "[Validation %d/%d] -> FAILED (iteration=%d) — candidate rejected",
                    i + 1,
                    validation_count,
                    val_iter,
                )
                return False, last_ctx

            logger.debug(
                "[Validation %d/%d] -> passed (iteration=%d)",
                i + 1,
                validation_count,
                val_iter,
            )

        logger.info(
            "[Iteration %d] -> All %d validation sample(s) passed — candidate confirmed",
            iteration,
            validation_count,
        )
        return True, last_ctx

    async def _run_optimization(
        self, agent_config: AIAgentConfig, options: OptimizationOptions
    ) -> Any:
        """Run an optimization on the given agent with the given options.

        :param agent_config: Agent configuration from LaunchDarkly.
        :param options: Optimization options.
        :return: Optimization result.
        """
        self._options = options
        self._agent_config = agent_config
        self._last_run_succeeded = False
        self._last_succeeded_context = None
        self._last_optimization_result_id = None
        self._initialize_class_members_from_config(agent_config)

        # If the LD flag doesn't carry a model name, seed from the first model choice
        # so agent calls never receive an empty model string.
        if not self._current_model and options.model_choices:
            self._current_model = options.model_choices[0]
            logger.debug(
                "[Optimization] -> No model in agent config; defaulting to first model choice: %s",
                self._current_model,
            )

        initial_context = self._create_optimization_context(
            iteration=0,
            variables=random.choice(options.variable_choices),
        )

        self._safe_status_update("init", initial_context, 0)

        iteration = 0
        while True:
            iteration += 1
            logger.info(
                "[Iteration %d] -> Starting (attempt %d/%d, model=%s)",
                iteration,
                iteration,
                self._options.max_attempts,
                self._current_model,
            )
            user_input = None
            if self._options.user_input_options:
                user_input = random.choice(self._options.user_input_options)
            if user_input:
                logger.debug("[Iteration %d] -> User input: %s", iteration, user_input)

            optimize_context = self._create_optimization_context(
                iteration=iteration,
                user_input=user_input,
                # Pick a fresh variable set each turn for call-time interpolation
                variables=random.choice(self._options.variable_choices),
            )

            self._safe_status_update("generating", optimize_context, iteration)
            optimize_context = await self._execute_agent_turn(
                optimize_context, iteration
            )

            # Manual path: on_turn callback gives caller full control over pass/fail
            if self._options.on_turn is not None:
                try:
                    on_turn_result = self._options.on_turn(optimize_context)
                except Exception:
                    logger.exception(
                        "[Iteration %d] -> on_turn evaluation failed", iteration
                    )
                    on_turn_result = False

                initial_passed = on_turn_result
                if initial_passed:
                    logger.info(
                        "[Iteration %d] -> on_turn returned True — turn passed",
                        iteration,
                    )
            else:
                # Auto-path: judge scores determine pass/fail via _evaluate_response
                initial_passed = self._evaluate_response(optimize_context)
                if initial_passed:
                    logger.info(
                        "[Iteration %d] -> All judges passed — turn succeeded",
                        iteration,
                    )

            if initial_passed and _acceptance_criteria_implies_duration_optimization(
                self._options.judges
            ):
                initial_passed = self._evaluate_duration(optimize_context)

            if initial_passed:
                all_valid, last_ctx = await self._run_validation_phase(
                    optimize_context, iteration
                )
                if all_valid:
                    return self._handle_success(last_ctx, iteration)
                # Validation failed — treat as a normal failed attempt.
                # Use optimize_context (the main iteration) for terminal API events so
                # the persisted record's completionResponse and userInput stay aligned.
                # last_ctx (the failing validation run) goes into history so the
                # variation generator can see what went wrong.
                logger.info(
                    "[Iteration %d] -> Validation failed — generating new variation (attempt %d/%d)",
                    iteration,
                    iteration,
                    self._options.max_attempts,
                )
                if iteration >= self._options.max_attempts:
                    return self._handle_failure(optimize_context, iteration)
                self._history.append(last_ctx)
                try:
                    await self._generate_new_variation(
                        iteration, last_ctx.current_variables
                    )
                except Exception:
                    logger.exception(
                        "[Iteration %d] -> variation generation failed", iteration
                    )
                    return self._handle_failure(optimize_context, iteration)
                self._safe_status_update("turn completed", optimize_context, iteration)
                continue

            # Initial turn failed
            if self._options.on_turn is not None:
                logger.info(
                    "[Iteration %d] -> on_turn returned False — turn failed (attempt %d/%d)",
                    iteration,
                    iteration,
                    self._options.max_attempts,
                )
            else:
                logger.info(
                    "[Iteration %d] -> One or more judges failed (attempt %d/%d) — generating new variation",
                    iteration,
                    iteration,
                    self._options.max_attempts,
                )
            if iteration >= self._options.max_attempts:
                return self._handle_failure(optimize_context, iteration)
            self._history.append(optimize_context)
            try:
                await self._generate_new_variation(
                    iteration, optimize_context.current_variables
                )
            except Exception:
                logger.exception(
                    "[Iteration %d] -> variation generation failed", iteration
                )
                return self._handle_failure(optimize_context, iteration)
            self._safe_status_update("turn completed", optimize_context, iteration)
            continue
