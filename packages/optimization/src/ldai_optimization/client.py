"""Client for LaunchDarkly AI agent optimization."""

import dataclasses
import json
import logging
import os
import random
import uuid
from typing import Any, Dict, List, Literal, Optional

from ldai import AIAgentConfig, AIJudgeConfig, AIJudgeConfigDefault, LDAIClient
from ldai.models import LDMessage, ModelConfig
from ldclient import Context

from ldai_optimization.dataclasses import (
    AIJudgeCallConfig,
    AutoCommitConfig,
    JudgeResult,
    OptimizationContext,
    OptimizationFromConfigOptions,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
    ToolDefinition,
)
from ldai_optimization.ld_api_client import (
    AgentOptimizationConfig,
    LDApiClient,
    OptimizationResultPayload,
)
from ldai_optimization.prompts import (
    build_message_history_text,
    build_new_variation_prompt,
    build_reasoning_history,
)
from ldai_optimization.util import (
    await_if_needed,
    create_evaluation_tool,
    create_variation_tool,
    extract_json_from_response,
    handle_evaluation_tool_call,
    handle_variation_tool_call,
    interpolate_variables,
)

logger = logging.getLogger(__name__)


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


# Maps SDK status strings to the API status/activity values expected by
# agent_optimization_result records. Defined at module level to avoid
# allocating the dict on every on_status_update invocation.
_OPTIMIZATION_STATUS_MAP: Dict[str, Dict[str, str]] = {
    "init": {"status": "RUNNING", "activity": "PENDING"},
    "generating": {"status": "RUNNING", "activity": "GENERATING"},
    "evaluating": {"status": "RUNNING", "activity": "EVALUATING"},
    "generating variation": {"status": "RUNNING", "activity": "GENERATING_VARIATION"},
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
        self._current_instructions = agent_config.instructions or ""
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
            except Exception as e:
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

    def _builtin_judge_tool_handlers(self) -> Dict[str, Any]:
        """
        Build the dict of built-in tool name → handler passed to handle_judge_call.

        Each handler accepts the tool-call arguments dict produced by the LLM and
        returns a JSON string so the caller can forward it back to the model or use
        it directly as the judge response.

        :return: Mapping of built-in tool names to their handler callables
        """
        return {
            create_evaluation_tool().name: handle_evaluation_tool_call,
        }

    def _builtin_agent_tool_handlers(self, is_variation: bool) -> Dict[str, Any]:
        """
        Build the dict of built-in tool name → handler passed to handle_agent_call.

        For regular agent turns this is empty — the config only contains user-defined
        tools from the LD flag. For variation-generation turns the variation structured
        output tool is included so the caller can distinguish it from user tools and
        route the LLM tool call back to the framework.

        :param is_variation: True when called for a variation-generation turn
        :return: Mapping of built-in tool names to their handler callables
        """
        if is_variation:
            return {
                create_variation_tool(
                    self._options.model_choices
                ).name: handle_variation_tool_call,
            }
        return {}

    async def _call_judges(
        self,
        completion_response: str,
        iteration: int,
        user_input: str,
        variables: Optional[Dict[str, Any]] = None,
        agent_tools: Optional[List[ToolDefinition]] = None,
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
            except Exception as e:
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
                    + " Use the structured output tool to format your response."
                    " You should always return a JSON object with a score and rationale."
                )
            elif msg.role == "user":
                user_parts.append(msg.content)

        instructions = "\n\n".join(system_parts)
        judge_user_input = (
            "\n\n".join(user_parts)
            if user_parts
            else f"Here is the response to evaluate: {completion_response}"
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

        # Add structured output tool for score and rationale
        tools.append(create_evaluation_tool())

        judge_call_config = AIJudgeCallConfig(
            key=judge_key,
            model=ModelConfig(
                name=model_name,
                parameters={**model_params, "tools": [t.to_dict() for t in tools]},
            ),
            instructions=instructions,
            messages=updated_messages,
        )

        judge_ctx = OptimizationJudgeContext(
            user_input=judge_user_input,
            variables=variables or {},
        )

        result = self._options.handle_judge_call(
            judge_key, judge_call_config, judge_ctx, self._builtin_judge_tool_handlers()
        )
        judge_response_str = await await_if_needed(result)

        logger.debug(
            "[Iteration %d] -> Judge response (%s): %s",
            iteration,
            judge_key,
            judge_response_str,
        )

        # Parse judge response — expect structured JSON output
        judge_identifier = optimization_judge.judge_key or judge_key
        return self._parse_judge_response(
            judge_response_str,
            judge_key,
            judge_identifier,
            iteration,
            clamp_score=False,
        )

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
            "You should call the structured output tool to format your response.\n\n"
            'Example: {"score": 0.8, "rationale": "The response matches the acceptance statement well."}'
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

        # Prepend agent tools so the judge can invoke them for verification if needed
        tools: List[ToolDefinition] = list(resolved_agent_tools) + [
            create_evaluation_tool()
        ]

        judge_user_input = f"Here is the response to evaluate: {completion_response}"

        judge_call_config = AIJudgeCallConfig(
            key=judge_key,
            model=ModelConfig(
                name=self._options.judge_model,
                parameters={"tools": [t.to_dict() for t in tools]},
            ),
            instructions=instructions,
            messages=[
                LDMessage(role="system", content=instructions),
                LDMessage(role="user", content=judge_user_input),
            ],
        )

        judge_ctx = OptimizationJudgeContext(
            user_input=judge_user_input,
            variables=resolved_variables,
        )

        result = self._options.handle_judge_call(
            judge_key, judge_call_config, judge_ctx, self._builtin_judge_tool_handlers()
        )
        judge_response = await await_if_needed(result)

        logger.debug(
            "[Iteration %d] -> Judge response (%s): %s",
            iteration,
            judge_key,
            judge_response,
        )

        # Parse judge response — expect structured JSON output with score and rationale
        return self._parse_judge_response(
            judge_response, judge_key, judge_key, iteration, clamp_score=True
        )

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
            self._initial_instructions = raw_instructions

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
        self._agent_key = agent_key
        context = random.choice(options.context_choices)
        agent_config = await self._get_agent_config(agent_key, context)
        return await self._run_optimization(agent_config, options)

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
            logger.info(
                "[Iteration %d] -> Model updated from '%s' to '%s'",
                iteration,
                old_model,
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

        instructions = build_new_variation_prompt(
            self._history,
            self._options.judges,
            self._current_model,
            self._current_instructions,
            self._current_parameters,
            self._options.model_choices,
            self._options.variable_choices,
            self._initial_instructions,
        )

        # Create a flat history list (without nested history) to avoid exponential growth
        flat_history = [prev_ctx.copy_without_history() for prev_ctx in self._history]

        # Create context for variation generation — low temperature for deterministic output.
        # The variation tool is placed in current_parameters["tools"] so it surfaces through
        # AIAgentConfig.model.parameters like any other tool, rather than as a separate field.
        variation_ctx = OptimizationContext(
            scores={},
            completion_response="",
            current_instructions=instructions,
            current_parameters={
                "temperature": 0.1,
                "tools": [create_variation_tool(self._options.model_choices).to_dict()],
            },
            current_variables=variables,
            current_model=self._current_model,
            user_input=None,
            history=tuple(flat_history),
            iteration=len(self._history) + 1,
        )

        # Call handle_agent_call to generate new variation; expects a JSON string
        # matching the structured output schema (current_instructions, current_parameters, model)
        result = self._options.handle_agent_call(
            self._agent_key,
            self._build_agent_config_for_context(variation_ctx),
            variation_ctx,
            self._builtin_agent_tool_handlers(is_variation=True),
        )
        response_str = await await_if_needed(result)

        # Extract and update current state from the parsed response
        response_data = extract_json_from_response(response_str)
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

        assert self._api_key is not None
        api_client = LDApiClient(
            self._api_key,
            **({"base_url": options.base_url} if options.base_url else {}),
        )
        config = api_client.get_agent_optimization(options.project_key, optimization_config_key)

        self._agent_key = config["aiConfigKey"]
        optimization_id: str = config["id"]
        run_id = str(uuid.uuid4())

        context = random.choice(options.context_choices)
        # _get_agent_config calls _initialize_class_members_from_config internally;
        # _run_optimization calls it again to reset history before the loop starts.
        agent_config = await self._get_agent_config(self._agent_key, context)

        optimization_options = self._build_options_from_config(
            config, options, api_client, optimization_id, run_id
        )
        return await self._run_optimization(agent_config, optimization_options)

    def _build_options_from_config(
        self,
        config: AgentOptimizationConfig,
        options: OptimizationFromConfigOptions,
        api_client: LDApiClient,
        optimization_id: str,
        run_id: str,
    ) -> OptimizationOptions:
        """Map a fetched AgentOptimization config + user options into OptimizationOptions.

        Acceptance statements and judge configs from the API are merged into a single
        judges dict. An on_status_update closure is injected to persist each iteration
        result to the LaunchDarkly API; any user-supplied on_status_update is chained
        after the persistence call.

        :param config: Validated AgentOptimizationConfig from the API.
        :param options: User-provided options from optimize_from_config.
        :param api_client: Initialised LDApiClient for result persistence.
        :param optimization_id: UUID id of the parent agent_optimization record.
        :param run_id: UUID that groups all result records for this run.
        :return: A fully populated OptimizationOptions ready for _run_optimization.
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

        has_ground_truth = bool(config.get("groundTruthResponses"))
        if not judges and not has_ground_truth and options.on_turn is None:
            raise ValueError(
                "The optimization config has no acceptance statements, judges, or ground truth "
                "responses, and no on_turn callback was provided. At least one is required to "
                "evaluate optimization results."
            )

        variable_choices: List[Dict[str, Any]] = config["variableChoices"] or [{}]
        user_input_options: Optional[List[str]] = config["userInputOptions"] or None

        project_key = options.project_key
        config_version: int = config["version"]

        def _persist_and_forward(
            status: Literal[
                "init",
                "generating",
                "evaluating",
                "generating variation",
                "turn completed",
                "success",
                "failure",
            ],
            ctx: OptimizationContext,
        ) -> None:
            # _safe_status_update (the caller) already wraps this entire function in
            # a try/except, so errors here are caught and logged without aborting the run.
            mapped = _OPTIMIZATION_STATUS_MAP.get(
                status, {"status": "RUNNING", "activity": "PENDING"}
            )
            snapshot = ctx.copy_without_history()
            payload: OptimizationResultPayload = {
                "run_id": run_id,
                "config_optimization_version": config_version,
                "status": mapped["status"],
                "activity": mapped["activity"],
                "iteration": snapshot.iteration,
                "instructions": snapshot.current_instructions,
                "parameters": snapshot.current_parameters,
                "completion_response": snapshot.completion_response,
                "scores": {k: v.to_json() for k, v in snapshot.scores.items()},
                "user_input": snapshot.user_input,
            }
            api_client.post_agent_optimization_result(project_key, optimization_id, payload)

            if options.on_status_update:
                try:
                    options.on_status_update(status, ctx)
                except Exception:
                    logger.exception("User on_status_update callback failed for status=%s", status)

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
    ) -> OptimizationContext:
        """
        Run the agent call and judge scoring for one optimization turn.

        Returns a new OptimizationContext with completion_response and scores
        populated, leaving the input context unchanged. Variables are read from
        optimize_context.current_variables and interpolated into the agent's
        instructions at call time so the stored template is never mutated.

        :param optimize_context: The context for this turn (instructions, model, history, etc.)
        :param iteration: Current iteration number for logging and status callbacks
        :return: Updated context with completion_response and scores filled in
        """
        logger.info(
            "[Iteration %d] -> Calling agent (model=%s)...",
            iteration,
            optimize_context.current_model,
        )
        try:
            result = self._options.handle_agent_call(
                self._agent_key,
                self._build_agent_config_for_context(optimize_context),
                optimize_context,
                self._builtin_agent_tool_handlers(is_variation=False),
            )
            completion_response = await await_if_needed(result)
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
            self._safe_status_update("evaluating", optimize_context, iteration)
            agent_tools = self._extract_agent_tools(optimize_context.current_parameters)
            scores = await self._call_judges(
                completion_response,
                iteration,
                user_input=optimize_context.user_input or "",
                variables=optimize_context.current_variables,
                agent_tools=agent_tools,
            )

        return dataclasses.replace(
            optimize_context,
            completion_response=completion_response,
            scores=scores,
        )

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
        self._safe_status_update("failure", optimize_context, iteration)
        if self._options.on_failing_result:
            try:
                self._options.on_failing_result(optimize_context)
            except Exception:
                logger.exception(
                    "[Iteration %d] -> on_failing_result callback failed", iteration
                )
        return optimize_context

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
        self._initialize_class_members_from_config(agent_config)

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

                if on_turn_result:
                    logger.info(
                        "[Iteration %d] -> on_turn returned True — turn passed",
                        iteration,
                    )
                    return self._handle_success(optimize_context, iteration)

                logger.info(
                    "[Iteration %d] -> on_turn returned False — turn failed (attempt %d/%d)",
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
            else:
                # Auto-path: judge scores determine pass/fail via _evaluate_response
                passes = self._evaluate_response(optimize_context)
                if passes:
                    logger.info(
                        "[Iteration %d] -> All judges passed — turn succeeded",
                        iteration,
                    )
                    return self._handle_success(optimize_context, iteration)
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
                    self._safe_status_update(
                        "turn completed", optimize_context, iteration
                    )
                    continue
