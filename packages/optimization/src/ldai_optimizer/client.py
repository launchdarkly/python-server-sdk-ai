"""Client for LaunchDarkly AI agent optimization.

Security note — LAUNCHDARKLY_API_KEY scope
-------------------------------------------
When set, the ``LAUNCHDARKLY_API_KEY`` environment variable is used solely to
authenticate discrete LaunchDarkly REST API calls (e.g. fetching optimization
configs, publishing results via ``auto_commit``). It is:

- Never included in any LLM prompt.
- Never forwarded to user-supplied ``handle_agent_call`` or ``handle_judge_call``
  callbacks.
- Never accessible to any external service other than the LaunchDarkly REST API.

All LaunchDarkly API calls are isolated requests; they carry no information
about the caller's broader runtime environment beyond the key itself.
"""

import asyncio
import dataclasses
import json
import logging
import os
import random
import re
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from ldai import AIAgentConfig, AIJudgeConfig, AIJudgeConfigDefault, LDAIClient
from ldai.models import LDMessage, ModelConfig
from ldclient import Context

from ldai_optimizer.dataclasses import (
    AIJudgeCallConfig,
    GroundTruthOptimizationOptions,
    GroundTruthSample,
    HandleJudgeCall,
    JudgeResult,
    OptimizationContext,
    OptimizationFromConfigOptions,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
    OptimizationResponse,
    ToolDefinition,
)
from ldai_optimizer.ld_api_client import (
    AgentOptimizationConfig,
    AgentOptimizationResultPatch,
    AgentOptimizationResultPost,
    LDApiClient,
    LDApiError,
)
from ldai_optimizer.prompts import (
    build_message_history_text,
    build_new_variation_prompt,
    build_reasoning_history,
    build_token_latency_variation_prompt,
)
from ldai_optimizer.util import (
    RedactionFilter,
    await_if_needed,
    estimate_cost,
    extract_json_from_response,
    generate_slug,
    interpolate_variables,
    judge_passed,
    restore_variable_placeholders,
    validate_variation_response,
)

logger = logging.getLogger(__name__)
logger.addFilter(RedactionFilter())


_TRANSIENT_RETRY_STATUSES = frozenset({429, 503, 529})
_TRANSIENT_NAME_FRAGMENTS = frozenset({"overloaded", "ratelimit", "rate_limit", "toomanyrequests"})
# Default per-call retry budget for LLM calls that hit transient provider errors.
_LLM_CALL_MAX_RETRIES = 3
_LLM_CALL_BASE_DELAY_S = 2.0


def _is_transient_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a recoverable provider error worth retrying.

    Works without importing any provider SDK: checks the ``status_code`` /
    ``status`` / ``http_status`` attribute that most HTTP-client libraries
    expose, and falls back to a keyword scan of the exception class name.
    """
    for attr in ("status_code", "status", "http_status", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int) and code in _TRANSIENT_RETRY_STATUSES:
            return True
    name = type(exc).__name__.lower().replace("-", "_")
    return any(frag in name for frag in _TRANSIENT_NAME_FRAGMENTS)


async def _invoke_with_retry(
    label: str,
    coro_factory: Any,
    max_retries: int = _LLM_CALL_MAX_RETRIES,
    base_delay: float = _LLM_CALL_BASE_DELAY_S,
) -> Any:
    """Await *coro_factory()* and retry up to *max_retries* times on transient errors.

    :param label: Short human-readable name used in log messages (e.g. "agent call").
    :param coro_factory: Zero-argument callable that returns a coroutine (or any
        awaitable) representing a single LLM call attempt.
    :param max_retries: Maximum number of additional attempts after the first.
    :param base_delay: Base sleep duration in seconds; doubled after each attempt.
    :returns: The return value of the successful coroutine.
    :raises: The last exception if all attempts are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return await await_if_needed(coro_factory())
        except Exception as exc:
            if attempt < max_retries and _is_transient_error(exc):
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Transient error on %s (attempt %d/%d), retrying in %.1fs: %s",
                    label,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
            else:
                raise


def _interpolate(template: str, variables: Dict[str, Any]) -> str:
    """Replace {{key}} tokens with values from variables; unresolved tokens become empty string."""
    return re.sub(
        r"\{\{([\w-]+)\}\}",
        lambda m: str(variables.get(m.group(1), "")),
        template,
    )


def _find_model_config(
    model_name: str, configs: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find the best matching model config for a given model name.

    Matches on either the catalog ``id`` (e.g. ``"gpt-4o"``) or the catalog
    ``key`` (e.g. ``"OpenAI.gpt-4o"``). The ``key`` form is what the API
    returns as ``modelConfigKey`` on a variation, so both must be checked.

    When multiple configs match, the one marked ``global=True`` is preferred
    over project-specific configs. Falls back to the first non-global match.

    :param model_name: The model id or key to look up.
    :param configs: List of model config dicts from the LD API.
    :return: Best-matching model config dict, or None if no match.
    """
    matching = [
        mc for mc in configs
        if mc.get("id") == model_name or mc.get("key") == model_name
    ]
    if not matching:
        return None
    global_match = next((mc for mc in matching if mc.get("global") is True), None)
    return global_match if global_match is not None else matching[0]


# Known provider prefixes used by the LD API (e.g. "Anthropic.claude-3").
# Only strip the first segment when it is one of these known values so that
# model IDs whose first dotted segment is NOT a provider — such as Bedrock
# cross-region inference IDs like "us.amazon.nova-pro-v1:0" — are left intact.
_KNOWN_PROVIDER_PREFIXES: frozenset = frozenset({
    "Anthropic",
    "Bedrock",
    "Cohere",
    "Google",
    "Groq",
    "Meta",
    "Mistral",
    "OpenAI",
    "Perplexity",
})


def _strip_provider_prefix(model: str) -> str:
    """Strip the provider prefix from a model identifier returned by the LD API.

    API model keys are formatted as "Provider.model-name" (e.g. "OpenAI.gpt-5",
    "Anthropic.claude-opus-4.6"). Only the segment before the first period is
    stripped, and only when it is a recognised provider name. This prevents
    incorrectly stripping region prefixes from Bedrock cross-region inference
    IDs such as "us.amazon.nova-pro-v1:0".

    :param model: Raw model string from the API.
    :return: Model name with provider prefix removed, or the original string if
        the first segment is not a known provider.
    """
    prefix, _, rest = model.partition(".")
    if prefix in _KNOWN_PROVIDER_PREFIXES and rest:
        return rest
    return model


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

# Cost gate: a candidate must cost at most this fraction of the baseline
# (history[0].estimated_cost_usd) to pass when acceptance criteria imply a
# cost reduction goal. 0.90 means at least 10% cheaper than the baseline.
_COST_TOLERANCE = 0.90

# Maximum number of history items retained in the standard (non-GT) optimizer.
# Since user inputs are randomly selected there is no "full pass" concept, so
# a small fixed window is sufficient context for variation generation.
_MAX_STANDARD_HISTORY_LENGTH = 5


def _trim_history(
    history: List["OptimizationContext"], max_len: int
) -> List["OptimizationContext"]:
    """Trim history to at most max_len of the most recent items.

    The duration/cost baselines are captured explicitly in ``_baseline_duration_ms``
    and ``_baseline_cost_usd`` so the oldest entry no longer needs to be preserved.

    :param history: Current accumulated history list.
    :param max_len: Maximum number of items to retain (must be >= 1).
    :return: Trimmed history list, or the original list if already within limit.
    """
    if len(history) <= max_len:
        return history
    return history[-max_len:]


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
        self._initial_model_custom: Optional[Dict[str, Any]] = None
        self._total_token_usage: int = 0
        self._model_configs: List[Dict[str, Any]] = []
        self._last_batch_size: int = 1
        self._in_cost_latency_phase: bool = False
        self._ld_context: Optional[Context] = None

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
        # Explicit baseline captured from the first iteration ever appended to history.
        # Stored separately so that history truncation can be a pure slice without
        # having to preserve history[0] as an anchor.
        self._baseline_duration_ms: Optional[float] = None
        self._baseline_cost_usd: Optional[float] = None

    def _record_baseline(self, ctx: OptimizationContext) -> None:
        """Capture duration/cost baseline from a single context.

        Used by the standard (non-GT) optimization loop where each iteration
        produces one result. Called once per run (subsequent calls are no-ops
        once both values are set). Storing these explicitly lets
        ``_trim_history`` use a simple tail slice without needing to preserve
        ``history[0]`` as an anchor.
        """
        if self._baseline_duration_ms is None and ctx.duration_ms is not None:
            self._baseline_duration_ms = ctx.duration_ms
        if self._baseline_cost_usd is None and ctx.estimated_cost_usd is not None:
            self._baseline_cost_usd = ctx.estimated_cost_usd

    def _record_baseline_from_batch(self, attempt_results: List[OptimizationContext]) -> None:
        """Capture duration/cost baseline as the average across a GT batch.

        Used by the GT optimization loop. The first attempt's N samples form
        the baseline; averaging them gives a more stable reference than a
        single sample and ensures comparisons in subsequent attempts reflect
        the typical performance of the original configuration rather than an
        outlier measurement.

        Called once per run (subsequent calls are no-ops once both values are
        set).

        :param attempt_results: All completed sample contexts from the first
            GT attempt.
        """
        if not attempt_results:
            return
        if self._baseline_duration_ms is None:
            durations = [
                ctx.duration_ms for ctx in attempt_results if ctx.duration_ms is not None
            ]
            if durations:
                self._baseline_duration_ms = sum(durations) / len(durations)
        if self._baseline_cost_usd is None:
            costs = [
                ctx.estimated_cost_usd
                for ctx in attempt_results
                if ctx.estimated_cost_usd is not None
            ]
            if costs:
                self._baseline_cost_usd = sum(costs) / len(costs)

    def _build_agent_config_for_context(
        self, ctx: OptimizationContext, skip_interpolation: bool = False
    ) -> AIAgentConfig:
        """
        Construct an AIAgentConfig that reflects the current optimization iteration.

        Uses the instructions, model, and parameters from the given context so the
        caller receives the variation being evaluated rather than the original base config.
        ``{{placeholder}}`` tokens in the instructions are substituted using
        ctx.current_variables at call time so the stored template is never mutated.

        :param ctx: The OptimizationContext for this iteration
        :param skip_interpolation: When True, skip variable interpolation on the
            instructions. Use this when the instructions are a meta-prompt (e.g. a
            variation-generation prompt) that deliberately contains ``{{key}}`` tokens
            as text for the LLM to read rather than as runtime substitution targets.
        :return: A fresh AIAgentConfig populated from the context's current state
        """
        instructions = (
            interpolate_variables(ctx.current_instructions, ctx.current_variables)
            if ctx.current_variables and not skip_interpolation
            else ctx.current_instructions
        )
        return AIAgentConfig(
            key=self._agent_key,
            enabled=True,
            create_tracker=self._agent_config.create_tracker,
            evaluator=self._agent_config.evaluator,
            model=ModelConfig(
                name=_strip_provider_prefix(ctx.current_model or ""),
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
            accumulated_token_usage=self._total_token_usage if self._total_token_usage > 0 else None,
        )

    @property
    def _judge_call(self) -> HandleJudgeCall:
        """Return the judge callable, falling back to handle_agent_call when not set."""
        return self._options.handle_judge_call or self._options.handle_agent_call

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
        Fetch a judge configuration by evaluating the flag variation directly.

        Bypasses LDAIClient.judge_config to avoid the reserved-variable warnings
        for 'message_history' and 'response_to_evaluate'. Those variables are
        interpolated here with their actual values instead of being neutralised
        by the SDK. If the template contains only a system message, a user turn
        is synthesised from the provided message_history and response_to_evaluate
        so that _evaluate_config_judge always receives a complete conversation.

        :param judge_key: The key for the judge configuration in LaunchDarkly
        :param context: The evaluation context
        :param default: Unused; kept for signature compatibility
        :param variables: Template variables including message_history and response_to_evaluate
        :return: The resolved AIJudgeConfig
        """
        variation: Dict[str, Any] = self._ldClient._client.variation(judge_key, context, {})
        enabled: bool = bool(variation.get("_ldMeta", {}).get("enabled", False))

        all_variables: Dict[str, Any] = {"ldctx": context.to_dict(), **variables}

        messages: List[LDMessage] = []
        raw_messages = variation.get("messages")
        if isinstance(raw_messages, list) and all(isinstance(m, dict) for m in raw_messages):
            messages = [
                LDMessage(
                    role=m["role"],
                    content=_interpolate(m.get("content", ""), all_variables),
                )
                for m in raw_messages
            ]

        # New-style templates only have a system message. Auto-generate a user
        # turn so _evaluate_config_judge always has a complete conversation to split.
        if not any(m.role == "user" for m in messages):
            message_history = variables.get("message_history", "")
            response_to_evaluate = variables.get("response_to_evaluate", "")
            parts: List[str] = []
            if message_history:
                parts.append(str(message_history))
            parts.append(f"Here is the response to evaluate: {response_to_evaluate}")
            messages.append(LDMessage(role="user", content="\n\n".join(parts)))

        model: Optional[ModelConfig] = None
        raw_model = variation.get("model")
        if isinstance(raw_model, dict):
            model = ModelConfig(
                name=raw_model.get("name", ""),
                parameters=raw_model.get("parameters"),
                custom=raw_model.get("custom"),
            )

        return AIJudgeConfig(
            key=judge_key,
            enabled=enabled,
            create_tracker=lambda: None,
            model=model,
            messages=messages,
            evaluation_metric_key=variation.get("evaluationMetricKey"),
        )

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
        agent_usage: Optional[Any] = None,
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
        :param agent_usage: Token usage from the agent call. Forwarded to acceptance judges
            whose statement implies a cost goal so they can mention token usage in their rationale.
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
                        agent_usage=agent_usage,
                    )
                    judge_results[judge_key] = result

                threshold = (
                    optimization_judge.threshold
                    if optimization_judge.threshold is not None
                    else 1.0
                )
                passed = judge_passed(result.score, threshold, optimization_judge.is_inverted)
                logger.debug(
                    "[Iteration %d] -> Judge '%s' scored %.3f (threshold=%.3f, inverted=%s) -> %s%s",
                    iteration,
                    judge_key,
                    result.score,
                    threshold,
                    optimization_judge.is_inverted,
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
        judge_context = self._ld_context or self._options.context_choices[0]
        judge_config = self._judge_config(
            optimization_judge.judge_key,
            judge_context,
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

        # Always use the global judge_model; model parameters (temperature, etc.) from
        # the judge flag are still forwarded, but the model name is never overridden.
        model_name = self._options.judge_model
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
        judge_response: OptimizationResponse = await _invoke_with_retry(
            f"config judge '{judge_key}' (iteration {iteration})",
            lambda: self._judge_call(judge_key, judge_call_config, judge_ctx, True),
        )
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
        agent_usage: Optional[Any] = None,
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
        :param agent_usage: Token usage from the agent call. When the acceptance statement
            implies a cost goal, the judge is instructed to mention token usage and cost.
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
            self._in_cost_latency_phase
            and agent_duration_ms is not None
            and bool(self._options.latency_optimization)
        ):
            baseline_ms = self._baseline_duration_ms
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
                "In your rationale, state the duration and any change from baseline. "
                "If the latency goal is not yet met, include specific, actionable suggestions "
                "for how the model choice or parameters could be changed to reduce "
                "response time — for example: switching to a faster model or reducing max_tokens. "
                "These suggestions will be used directly to generate the next variation."
            )

        if self._in_cost_latency_phase and bool(self._options.token_optimization):
            current_cost = estimate_cost(
                agent_usage,
                _find_model_config(self._current_model or "", self._model_configs),
            )
            baseline_cost = self._baseline_cost_usd
            if current_cost is not None:
                instructions += (
                    f"\n\nThe acceptance criteria for this judge includes a cost/token-usage goal. "
                )
                if agent_usage is not None:
                    instructions += (
                        f"The agent's response used {agent_usage.input} input tokens "
                        f"and {agent_usage.output} output tokens "
                        f"(estimated cost: ${current_cost:.6f}). "
                    )
                if baseline_cost is not None:
                    delta = current_cost - baseline_cost
                    direction = "less" if delta < 0 else "more"
                    instructions += (
                        f"The baseline cost (first iteration) was ${baseline_cost:.6f}. "
                        f"This response cost ${abs(delta):.6f} {direction} than the baseline. "
                    )
                instructions += (
                    "In your rationale, state the token usage and cost, and any change from baseline. "
                    "If the cost goal is not yet met, include specific, actionable suggestions "
                    "for how the model choice or parameters could be changed to reduce "
                    "cost — for example: switching to a cheaper model or reducing max_tokens. "
                    "These suggestions will be used directly to generate the next variation."
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
        judge_response: OptimizationResponse = await _invoke_with_retry(
            f"acceptance judge '{judge_key}' (iteration {iteration})",
            lambda: self._judge_call(judge_key, judge_call_config, judge_ctx, True),
        )
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
        self,
        agent_key: str,
        context: Context,
        variation_key: Optional[str] = None,
        project_key: Optional[str] = None,
        api_client: Optional["LDApiClient"] = None,
        base_url: Optional[str] = None,
    ) -> AIAgentConfig:
        """
        Fetch the agent configuration, replacing the instructions with the raw variation
        template so that {{placeholder}} tokens are preserved for client-side interpolation.

        agent_config() is called normally so we get a fully populated AIAgentConfig
        (including the tracker). When variation_key is set, the specific variation's
        data (instructions, model, tools) is fetched via the REST API and used as the
        base instead of the SDK-evaluated default. Otherwise, variation() is called to
        retrieve the unrendered instruction template for the SDK-evaluated variation.

        When ``variation_key`` is provided the specific variation is fetched via the
        LaunchDarkly REST API instead of using the SDK's default flag evaluation.

        :param agent_key: The key for the agent to get the configuration for
        :param context: The evaluation context
        :param variation_key: If set, fetch this specific variation from the API as the base.
        :param project_key: Required when variation_key is set.
        :param api_client: Optional pre-built LDApiClient to reuse (e.g. from optimize_from_config).
        :param base_url: Optional base URL override for a newly created LDApiClient.
        :return: AIAgentConfig with raw {{placeholder}} instruction templates intact
        """
        try:
            agent_config = self._ldClient.agent_config(agent_key, context)

            if variation_key:
                # Fetch the specific variation from the REST API so instructions,
                # model, and tools all come from the requested base variation rather
                # than whatever the SDK evaluates for the given context.
                client = api_client or LDApiClient(
                    self._api_key,  # type: ignore[arg-type]
                    **({"base_url": base_url} if base_url else {}),
                )
                raw_variation = client.get_ai_config_variation(
                    project_key,  # type: ignore[arg-type]
                    agent_key,
                    variation_key,
                )
                raw_instructions = raw_variation.get("instructions") or ""
                raw_tools = raw_variation.get("tools") or []
                model_config_key = raw_variation.get("modelConfigKey") or ""
                if model_config_key:
                    raw_model_data = raw_variation.get("model")
                    model_parameters = (
                        raw_model_data.get("parameters")
                        if isinstance(raw_model_data, dict)
                        else {}
                    )
                    agent_config = dataclasses.replace(
                        agent_config,
                        model=ModelConfig(
                            name=_strip_provider_prefix(model_config_key),
                            parameters=model_parameters or {},
                        ),
                    )
            else:
                # variation() returns the raw JSON before chevron.render(), so instructions
                # still contain {{placeholder}} tokens rather than empty strings.
                raw_variation = self._ldClient._client.variation(agent_key, context, {})
                raw_instructions = raw_variation.get(
                    "instructions", agent_config.instructions
                )
                raw_tools = raw_variation.get("tools", [])

            if not raw_instructions:
                raise ValueError(
                    f"Agent '{agent_key}' has no instructions configured. "
                    "Ensure the agent flag has instructions set before running an optimization."
                )
            self._initial_instructions = raw_instructions

            self._initial_tool_keys = [
                t["key"]
                for t in raw_tools
                if isinstance(t, dict) and "key" in t
            ]

            raw_model = raw_variation.get("model")
            self._initial_model_custom = (
                raw_model.get("custom") if isinstance(raw_model, dict) else None
            )

            agent_config = dataclasses.replace(
                agent_config, instructions=raw_instructions
            )
            self._initialize_class_members_from_config(agent_config)
            # Merge variation-level tools into _current_parameters so that
            # _extract_agent_tools (which reads parameters["tools"]) finds them
            # during agent turns.  The model._parameters used above only carries
            # model-level parameters; variation tools are a separate top-level
            # field in the REST response and must be injected explicitly.
            # Create a new dict (not an in-place mutation) so agent_config.model
            # stays unaffected.
            if raw_tools and not self._current_parameters.get("tools"):
                self._current_parameters = {**self._current_parameters, "tools": raw_tools}
            return agent_config
        except Exception:
            logger.exception("[Optimization] -> Failed to get agent configuration")
            raise

    def _fetch_model_configs(
        self,
        project_key: Optional[str],
        base_url: Optional[str],
        token_optimization: Optional[bool],
    ) -> None:
        """Populate ``_model_configs`` from the LD API when credentials are available.

        When an API key and project key are both present, fetches the model pricing
        catalogue so that ``estimate_cost`` can produce USD figures and the cost gate
        can make meaningful comparisons.  If either is absent, ``_model_configs`` is
        reset to an empty list and a warning is emitted when token_optimization is
        enabled — cost data will be unavailable and the cost gate will pass unconditionally.

        :param project_key: LaunchDarkly project key, or None if not provided.
        :param base_url: Optional API base URL override.
        :param token_optimization: Whether token/cost optimization is enabled; used only to
            decide whether a cost-related warning is appropriate.
        """
        self._model_configs = []
        if self._has_api_key and project_key:
            assert self._api_key is not None
            try:
                api_client = LDApiClient(
                    self._api_key,
                    **({"base_url": base_url} if base_url else {}),
                )
                self._model_configs = api_client.get_model_configs(project_key)
            except Exception as exc:
                logger.debug("Could not pre-fetch model configs: %s", exc)
        elif token_optimization:
            logger.warning(
                "Token optimization requires LAUNCHDARKLY_API_KEY and project_key to be set; "
                "cost data will not be available and the cost gate will pass unconditionally"
            )

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
        if options.variation_key:
            if not self._has_api_key:
                raise ValueError(
                    "variation_key requires LAUNCHDARKLY_API_KEY to be set"
                )
            if not options.project_key:
                raise ValueError(
                    "variation_key requires project_key to be set on OptimizationOptions"
                )
        self._agent_key = agent_key
        self._fetch_model_configs(options.project_key, options.base_url, options.token_optimization)
        context = random.choice(options.context_choices)
        self._ld_context = context
        agent_config = await self._get_agent_config(
            agent_key,
            context,
            variation_key=options.variation_key,
            project_key=options.project_key,
            base_url=options.base_url,
        )
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
        if options.variation_key:
            if not self._has_api_key:
                raise ValueError(
                    "variation_key requires LAUNCHDARKLY_API_KEY to be set"
                )
            if not options.project_key:
                raise ValueError(
                    "variation_key requires project_key to be set on GroundTruthOptimizationOptions"
                )
        self._agent_key = agent_key
        self._fetch_model_configs(options.project_key, options.base_url, options.token_optimization)
        context = random.choice(options.context_choices)
        self._ld_context = context
        agent_config = await self._get_agent_config(
            agent_key,
            context,
            variation_key=options.variation_key,
            project_key=options.project_key,
            base_url=options.base_url,
        )
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
            token_limit=gt_options.token_limit,
            latency_optimization=gt_options.latency_optimization,
            token_optimization=gt_options.token_optimization,
        )
        self._options = bridge
        self._agent_config = agent_config
        self._last_run_succeeded = False
        self._last_succeeded_context = None
        self._last_optimization_result_id = None
        self._total_token_usage = 0
        self._last_batch_size = 1
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
                self._accumulate_tokens(optimize_context)
                optimize_context = dataclasses.replace(
                    optimize_context, accumulated_token_usage=self._total_token_usage
                )

                # Per-sample pass/fail check — evaluated before the token limit gate so
                # that a sample which passed is not incorrectly stamped as FAILED simply
                # because the budget was exhausted after its scores were computed.
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

                self._safe_status_update("evaluating", optimize_context, linear_iter)

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

                # Persist the completed sample so every API record gets its scores,
                # generation tokens, and accumulated_total — not just the final one.
                self._safe_status_update("turn completed", optimize_context, linear_iter)

                if gt_options.on_sample_result is not None:
                    try:
                        gt_options.on_sample_result(optimize_context)
                    except Exception:
                        logger.exception(
                            "[GT Attempt %d] -> on_sample_result callback failed for sample %d",
                            attempt,
                            i + 1,
                        )

                # Token limit check after pass/fail so the terminal status reflects
                # whether the samples actually passed, not just that the budget was hit.
                # Mark success only when every sample in this attempt was processed and
                # all passed; stopping mid-batch is always a failure even if the
                # partially-processed samples looked good.
                if self._is_token_limit_exceeded():
                    logger.error(
                        "[GT Attempt %d] -> Token limit exceeded on sample %d (total=%d)",
                        attempt,
                        i + 1,
                        self._total_token_usage,
                    )
                    if all_passed and i == n - 1:
                        self._last_run_succeeded = True
                        self._last_succeeded_context = optimize_context
                        self._safe_status_update("success", optimize_context, linear_iter)
                        if self._options.on_passing_result:
                            try:
                                self._options.on_passing_result(optimize_context)
                            except Exception:
                                logger.exception(
                                    "[GT Attempt %d] -> on_passing_result callback failed", attempt
                                )
                    else:
                        self._last_run_succeeded = False
                        self._last_succeeded_context = None
                        self._safe_status_update("failure", optimize_context, linear_iter)
                        if self._options.on_failing_result:
                            try:
                                self._options.on_failing_result(optimize_context)
                            except Exception:
                                logger.exception(
                                    "[GT Attempt %d] -> on_failing_result callback failed", attempt
                                )
                    return attempt_results

            last_ctx = attempt_results[-1]

            if all_passed:
                logger.info(
                    "[GT Attempt %d] -> All %d samples passed — optimization succeeded",
                    attempt,
                    n,
                )
                # Phase 2: optimize model/params on the frozen winning variation.
                if (
                    self._options.latency_optimization
                    or self._options.token_optimization
                ) and not self._is_token_limit_exceeded():
                    # Record Phase 1 success without firing on_passing_result yet;
                    # we fire it once below with the true final winner.
                    self._last_run_succeeded = True
                    self._last_succeeded_context = last_ctx
                    self._safe_status_update("success", last_ctx, last_ctx.iteration)
                    phase1_winner = self._last_succeeded_context
                    # Record baseline from the winning attempt before Phase 2 so the
                    # latency/cost gates have a reference even when GT passes on
                    # attempt 1 and _record_baseline_from_batch was never called.
                    self._record_baseline_from_batch(attempt_results)
                    await self._run_cost_latency_phase(
                        last_ctx,
                        last_ctx.iteration,
                        # Pass the ground-truth expected response for the last sample so
                        # Phase 2 judges can score against the labeled answer, matching
                        # the evaluation context used in Phase 1.
                        expected_response=samples[-1].expected_response,
                    )
                    if self._last_succeeded_context is None:
                        # No Phase 2 candidate won; restore the Phase 1 winner.
                        self._last_run_succeeded = True
                        self._last_succeeded_context = phase1_winner
                else:
                    self._last_run_succeeded = True
                    self._last_succeeded_context = last_ctx
                    self._safe_status_update("success", last_ctx, last_ctx.iteration)

                # Fire on_passing_result exactly once with the true final winner.
                final_winner = self._last_succeeded_context
                if final_winner and self._options.on_passing_result:
                    try:
                        self._options.on_passing_result(final_winner)
                    except Exception:
                        logger.exception(
                            "[GT Attempt %d] -> on_passing_result callback failed", attempt
                        )

                if (
                    self._last_succeeded_context is not None
                    and self._last_succeeded_context is not last_ctx
                ):
                    # Phase 2 selected a better model; return that context so
                    # callers (including auto_commit) see the actual final winner.
                    return [self._last_succeeded_context]
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
            # from all of the previous samples, then trim to one full attempt's worth so
            # judge prompts don't grow unboundedly across many failed attempts.
            if attempt_results:
                self._record_baseline_from_batch(attempt_results)
            self._history.extend(attempt_results)
            self._history = _trim_history(self._history, n)
            # Track batch size so _all_judges_passing checks every sample in this
            # attempt, not just the last one.
            self._last_batch_size = n

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
        validation_errors = validate_variation_response(response_data)
        if validation_errors:
            logger.debug(
                "[Iteration %d] -> Variation response failed validation: %s. "
                "Received fields: %s. Full response_data: %s",
                iteration,
                "; ".join(validation_errors),
                list(response_data.keys()),
                json.dumps(response_data, indent=2),
            )
            raise ValueError(
                f"Variation response failed validation: {'; '.join(validation_errors)}. "
                f"Received fields: {list(response_data.keys())}"
            )

        new_instructions = response_data["current_instructions"]

        if self._in_cost_latency_phase:
            if new_instructions != self._current_instructions:
                logger.warning(
                    "[Iteration %d] -> Phase 2 (cost/latency): LLM attempted to change instructions; "
                    "restoring frozen winning variation instructions to enforce content lock.",
                    iteration,
                )
                new_instructions = self._current_instructions

        self._current_instructions = new_instructions

        # Post-process: replace any leaked variable values back to {{key}} form.
        # This is a deterministic safety net for when the LLM ignores the prompt
        # instructions and hardcodes a concrete value (e.g. "user-123") instead
        # of the placeholder ("{{user_id}}").
        # Only check the variables that were actually used for this invocation so
        # we don't spuriously replace values that happen to appear in other choices.
        active_variables = (
            [variation_ctx.current_variables]
            if variation_ctx.current_variables
            else self._options.variable_choices
        )
        self._current_instructions, placeholder_warnings = restore_variable_placeholders(
            self._current_instructions,
            active_variables,
        )
        for msg in placeholder_warnings:
            logger.debug("[Iteration %d] -> %s", iteration, msg)

        # Merge the LLM's returned parameters into the existing ones so that custom
        # parameters (e.g. response_format, max_tokens, structured-output config)
        # are preserved even when the LLM omits them from its response.
        original_params = self._current_parameters.copy()
        new_params = response_data["current_parameters"]
        merged_params = {**original_params, **new_params}

        # Tools must be returned "unchanged" per the variation prompt. Always restore
        # the original tools so that (a) user-defined tools are never silently dropped
        # and (b) internal framework tools (e.g. structured-output tool injected by
        # the agent SDK) cannot leak in from the LLM's response.
        original_tools = original_params.get("tools")
        if original_tools is not None:
            returned_tools = new_params.get("tools")
            if returned_tools is not None and returned_tools != original_tools:
                logger.warning(
                    "[Iteration %d] -> LLM returned a modified tools list; restoring "
                    "original tools to prevent tool drift or internal-tool leakage. "
                    "Original: %s  Returned: %s",
                    iteration,
                    [t.get("name") if isinstance(t, dict) else getattr(t, "name", t) for t in original_tools],
                    [t.get("name") if isinstance(t, dict) else getattr(t, "name", t) for t in returned_tools],
                )
            merged_params["tools"] = original_tools

        self._current_parameters = merged_params

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

        if self._in_cost_latency_phase:
            instructions = build_token_latency_variation_prompt(
                self._history,
                self._options.model_choices,
                optimize_for_latency=bool(self._options.latency_optimization),
                optimize_for_cost=bool(self._options.token_optimization),
            )
        else:
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
        agent_config = self._build_agent_config_for_context(variation_ctx, skip_interpolation=True)
        response_data = None
        response_str = ""
        for attempt in range(1, _MAX_VARIATION_RETRIES + 1):
            variation_response: OptimizationResponse = await _invoke_with_retry(
                f"variation generation (iteration {iteration}, attempt {attempt})",
                lambda: self._options.handle_agent_call(
                    self._agent_key,
                    agent_config,
                    variation_ctx,
                    False,
                ),
            )
            if variation_response.usage is not None:
                self._total_token_usage += variation_response.usage.total or 0
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
        self._model_configs = model_configs

        context = random.choice(options.context_choices)
        self._ld_context = context
        # _get_agent_config calls _initialize_class_members_from_config internally;
        # _run_optimization calls it again to reset history before the loop starts.
        agent_config = await self._get_agent_config(
            self._agent_key,
            context,
            variation_key=config.get("variationKey"),
            project_key=options.project_key,
            api_client=api_client,
        )

        # Preflight: verify that the API key has write access to the results endpoint
        # before any agent work begins.  A read-only or wrong-project key would
        # otherwise silently 403 on every result POST mid-run, leaving the Results
        # tab empty with no visible error.
        try:
            api_client.verify_write_access(options.project_key, optimization_key)
        except LDApiError as exc:
            raise ValueError(
                f"API key does not have write access to project '{options.project_key}' "
                f"(optimization: '{optimization_key}'). Verify that your LAUNCHDARKLY_API_KEY "
                f"has the 'Writer' role (or equivalent) for this project before running an "
                f"optimization. Original error: {exc}"
            ) from exc

        optimization_options, _log_persist_summary = self._build_options_from_config(
            config, options, api_client, optimization_key, run_id, model_configs
        )
        if isinstance(optimization_options, GroundTruthOptimizationOptions):
            result = await self._run_ground_truth_optimization(agent_config, optimization_options)
        else:
            result = await self._run_optimization(agent_config, optimization_options)

        _log_persist_summary()

        if (optimization_options.auto_commit and options.auto_commit
                and self._last_run_succeeded and self._last_succeeded_context):
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
    ) -> "Tuple[Union[OptimizationOptions, GroundTruthOptimizationOptions], Any]":
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
        :return: Tuple of (OptimizationOptions or GroundTruthOptimizationOptions, summary_fn).
            Call summary_fn() after the optimization loop to log persistence health.
        """
        judges: Dict[str, OptimizationJudge] = {}

        for i, stmt in enumerate(config["acceptanceStatements"]):
            key = f"acceptance-statement-{i}"
            judges[key] = OptimizationJudge(
                threshold=float(stmt.get("threshold", 0.95)),
                acceptance_statement=stmt["statement"],
            )

        for judge in config["judges"]:
            judge_key = judge["key"]
            ai_config = api_client.get_ai_config(options.project_key, judge_key)
            is_inverted = bool(ai_config.get("isInverted", False)) if ai_config else False
            judges[judge_key] = OptimizationJudge(
                threshold=float(judge.get("threshold", 0.95)),
                judge_key=judge_key,
                is_inverted=is_inverted,
            )

        raw_ground_truth: List[str] = config.get("groundTruthResponses") or []
        has_ground_truth = bool(raw_ground_truth)
        if not judges and options.on_turn is None:
            raise ValueError(
                "The optimization config has no acceptance statements or judges, and no on_turn "
                "callback was provided. At least one is required to evaluate optimization results."
            )

        project_key = options.project_key
        config_version: int = config["version"]
        _cached_model_configs: List[Dict[str, Any]] = list(model_configs or [])

        # Warn early about model choices that have no matching model config — these
        # are likely retired or misspelled and will fail at run time with an opaque
        # provider error rather than a useful message.
        if _cached_model_configs:
            for raw_model in config.get("modelChoices") or []:
                if not _find_model_config(raw_model, _cached_model_configs):
                    logger.warning(
                        "Model choice '%s' was not found in the project's model configs — "
                        "it may be retired or misspelled and will likely fail at run time.",
                        raw_model,
                    )

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

        # Persistence health counters — incremented on every failed POST or PATCH
        # so a single summary warning can be emitted at run end.
        _post_failures: int = 0
        _patch_failures: int = 0
        _post_successes: int = 0
        _patch_successes: int = 0

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
                "optimizing cost/latency",
            ],
            ctx: OptimizationContext,
        ) -> None:
            nonlocal _in_validation_phase, _validation_parent_iteration, _last_open_iteration
            nonlocal _post_failures, _patch_failures, _post_successes, _patch_successes
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
                    "userInput": snapshot.user_input or "",
                }
                if snapshot.current_parameters:
                    post_payload["parameters"] = snapshot.current_parameters
                result_id = api_client.post_agent_optimization_result(
                    project_key, optimization_key, post_payload
                )
                if result_id:
                    _iteration_result_ids[logical_iteration] = result_id
                    self._last_optimization_result_id = result_id
                    _last_open_iteration = logical_iteration
                    _post_successes += 1
                else:
                    _post_failures += 1

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
                    gen_tokens: Dict[str, Any] = {
                        "total": snapshot.usage.total,
                        "input": snapshot.usage.input,
                        "output": snapshot.usage.output,
                    }
                    if snapshot.accumulated_token_usage is not None:
                        gen_tokens["accumulated_total"] = snapshot.accumulated_token_usage
                    patch["generationTokens"] = gen_tokens
                elif snapshot.accumulated_token_usage is not None:
                    patch["generationTokens"] = {
                        "accumulated_total": snapshot.accumulated_token_usage
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
                if api_client.patch_agent_optimization_result(
                    project_key, optimization_key, result_id, patch
                ):
                    _patch_successes += 1
                else:
                    _patch_failures += 1
                # When the winning result is marked successful, make sure
                # _last_optimization_result_id tracks it.  Without this the
                # auto-commit PATCH (which attaches createdVariationKey) is
                # sent to the last-posted Phase 2 record rather than to the
                # winner, giving a non-winning RUNNING record the latest
                # updatedAt and causing the backend to report the run as still
                # running even after the optimization completes.
                if status == "success":
                    self._last_optimization_result_id = result_id

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

        def _log_persist_summary() -> None:
            pass

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
                token_limit=config.get("tokenLimit"),
                latency_optimization=config.get("latencyOptimization"),
                token_optimization=config.get("tokenOptimization"),
                auto_commit=config.get("autoCommit", True),
            ), _log_persist_summary

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
            token_limit=config.get("tokenLimit"),
            latency_optimization=config.get("latencyOptimization"),
            token_optimization=config.get("tokenOptimization"),
            auto_commit=config.get("autoCommit", True),
        ), _log_persist_summary

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
            _agent_config = self._build_agent_config_for_context(optimize_context)
            agent_response: OptimizationResponse = await _invoke_with_retry(
                f"agent call (iteration {iteration})",
                lambda: self._options.handle_agent_call(
                    self._agent_key,
                    _agent_config,
                    optimize_context,
                    False,
                ),
            )
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
                agent_usage=agent_response.usage,
            )

        # Build the fully-populated result context before firing the evaluating event so
        # the PATCH includes scores, generationLatency, and completionResponse. This is
        # particularly important for non-final GT samples which receive no further status
        # events — without this, those fields would never be written to their API records.
        agent_cost = estimate_cost(
            agent_response.usage,
            _find_model_config(self._current_model or "", self._model_configs),
        )
        # Build a _meta entry capturing raw cost and latency telemetry for every
        # iteration, regardless of which optimization goals are enabled. This
        # surfaces the measurements in the API/UI even when the gates are inactive.
        result_ctx = dataclasses.replace(
            optimize_context,
            completion_response=completion_response,
            scores={**scores, "_meta": JudgeResult(
                score=0.0,
                duration_ms=agent_duration_ms,
                estimated_cost_usd=agent_cost,
            )},
            duration_ms=agent_duration_ms,
            usage=agent_response.usage,
            estimated_cost_usd=agent_cost,
        )

        if self._options.judges:
            self._safe_status_update("evaluating", result_ctx, iteration)

        return result_ctx

    def _accumulate_tokens(self, optimize_context: OptimizationContext) -> None:
        """Add token usage from a completed turn to the running total.

        Sums the agent's token usage and each judge's token usage from the given
        context and adds them to ``_total_token_usage``.

        :param optimize_context: The completed turn context containing usage data.
        """
        if optimize_context.usage is not None:
            self._total_token_usage += optimize_context.usage.total or 0
        for judge_result in optimize_context.scores.values():
            if judge_result.usage is not None:
                self._total_token_usage += judge_result.usage.total or 0

    def _is_token_limit_exceeded(self) -> bool:
        """Return True if the accumulated token usage has met or exceeded the configured limit.

        Returns False when no token limit is set, or when the limit is 0 (which is
        treated as "no limit" — a sentinel value meaning the field was left unset).

        :return: True if a positive token limit is set and ``_total_token_usage >= token_limit``.
        """
        limit: Optional[int] = getattr(self._options, "token_limit", None)
        return limit is not None and limit > 0 and self._total_token_usage >= limit

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
            if not judge_passed(result.score, threshold, optimization_judge.is_inverted):
                return False

        return True

    def _evaluate_duration(self, optimize_context: OptimizationContext) -> bool:
        """
        Check whether the candidate's duration meets the improvement target vs. the baseline.

        The baseline is the duration_ms from the very first iteration appended to history,
        captured in ``_baseline_duration_ms``. The candidate must be at least
        _DURATION_TOLERANCE faster (default: 20% improvement).

        Returns True without blocking when no baseline is available or when the candidate's
        duration_ms was not captured. This avoids penalising configurations when timing data
        is missing.

        :param optimize_context: The completed turn context containing duration_ms
        :return: True if the duration requirement is met or cannot be checked
        """
        if self._baseline_duration_ms is None:
            return True
        if optimize_context.duration_ms is None:
            return True
        baseline = self._baseline_duration_ms
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

    def _evaluate_cost(self, optimize_context: OptimizationContext) -> bool:
        """
        Check whether the candidate's estimated cost meets the improvement target vs. the baseline.

        The baseline is the estimated_cost_usd from the very first iteration appended to
        history, captured in ``_baseline_cost_usd``. The candidate must be at least
        _COST_TOLERANCE cheaper (default: 10% improvement).

        The cost value is in USD when model pricing data is available, or raw total token
        count as a proxy when pricing is absent. Both are comparable relative to their
        own baselines.

        Returns True without blocking when no baseline is available or when the candidate's
        cost was not captured. This avoids penalising configurations when cost data is missing.

        :param optimize_context: The completed turn context containing estimated_cost_usd
        :return: True if the cost requirement is met or cannot be checked
        """
        if self._baseline_cost_usd is None:
            return True
        if optimize_context.estimated_cost_usd is None:
            return True
        baseline = self._baseline_cost_usd
        passed = optimize_context.estimated_cost_usd < baseline * _COST_TOLERANCE
        if not passed:
            logger.warning(
                "[Iteration %d] -> Cost check failed: %.6f >= baseline %.6f * %.0f%% (%.6f)",
                optimize_context.iteration,
                optimize_context.estimated_cost_usd,
                baseline,
                _COST_TOLERANCE * 100,
                baseline * _COST_TOLERANCE,
            )
        return passed

    def _all_judges_passing(self) -> bool:
        """Return True if every user-configured judge passed in every sample of the most recent batch.

        In ground-truth mode the last ``_last_batch_size`` entries in ``_history``
        correspond to the samples from the latest attempt. All of them must pass;
        checking only the last entry would incorrectly return True when a middle sample
        failed but the final sample passed.

        In single-sample (non-GT) mode ``_last_batch_size`` is 1, so only the most
        recent entry is inspected (original behaviour).

        Synthetic gate entries (keys beginning with ``_``) are skipped.
        Returns False when history is empty or any judge score does not meet its threshold.

        This is used to decide whether variation generation should preserve the current
        behaviour and only optimise for cost, rather than trying to improve quality further.
        """
        if not self._history or not self._options.judges:
            return False
        batch = self._history[-self._last_batch_size:]
        for ctx in batch:
            if not ctx.scores:
                return False
            for key, judge in self._options.judges.items():
                result = ctx.scores.get(key)
                if result is None:
                    return False
                threshold = judge.threshold if judge.threshold is not None else 1.0
                if not judge_passed(result.score, threshold, judge.is_inverted):
                    return False
        return True

    def _apply_duration_gate(
        self, passed_so_far: bool, ctx: OptimizationContext
    ) -> Tuple[bool, OptimizationContext]:
        """Apply the latency improvement gate and record its result in ctx.scores.

        When the gate is active (any acceptance statement implies latency optimization),
        evaluates whether the candidate's duration improved by at least
        _DURATION_TOLERANCE vs the baseline. A synthetic ``_latency_gate`` entry is
        always added to scores with score=1.0 on pass or score=0.0 on fail so the
        outcome is visible in the API result and UI for every iteration.

        The gate score is recorded even when ``passed_so_far`` is False (quality
        judges already failed) so that latency telemetry is visible on all
        iterations, not just passing ones. In that case it is informational only
        and cannot block the iteration further.

        The gate is skipped entirely (no score entry added) only when no acceptance
        statement implies latency optimization.

        :param passed_so_far: Whether all prior checks for this sample passed.
        :param ctx: Current optimization context.
        :return: (passed, updated_ctx) where passed reflects gate outcome.
        """
        if not bool(self._options.latency_optimization):
            return passed_so_far, ctx
        passed = self._evaluate_duration(ctx)
        if passed:
            if self._baseline_duration_ms is not None and ctx.duration_ms is not None:
                rationale = (
                    f"Latency improvement gate passed: {ctx.duration_ms:.0f}ms is at least "
                    f"{int((1 - _DURATION_TOLERANCE) * 100)}% faster than baseline "
                    f"{self._baseline_duration_ms:.0f}ms."
                )
            else:
                rationale = "Latency gate passed (no baseline)."
            score = 1.0
        else:
            baseline_dur = self._baseline_duration_ms or 0.0
            rationale = (
                f"Latency improvement gate failed: {ctx.duration_ms:.0f}ms did not improve "
                f"by {int((1 - _DURATION_TOLERANCE) * 100)}% vs baseline "
                f"{baseline_dur:.0f}ms "
                f"(required < {baseline_dur * _DURATION_TOLERANCE:.0f}ms)."
            )
            score = 0.0
        ctx = dataclasses.replace(
            ctx,
            scores={**ctx.scores, "_latency_gate": JudgeResult(
                score=score,
                rationale=rationale,
                duration_ms=ctx.duration_ms,
            )},
        )
        return passed_so_far and passed, ctx

    def _apply_cost_gate(
        self, passed_so_far: bool, ctx: OptimizationContext
    ) -> Tuple[bool, OptimizationContext]:
        """Apply the cost improvement gate and record its result in ctx.scores.

        When the gate is active (any acceptance statement implies cost optimization),
        evaluates whether the candidate's estimated cost improved by at least
        _COST_TOLERANCE vs the baseline. A synthetic ``_cost_gate`` entry is always
        added to scores with score=1.0 on pass or score=0.0 on fail.

        The gate score is recorded even when ``passed_so_far`` is False (quality
        judges already failed) so that cost telemetry is visible on all iterations,
        not just passing ones. In that case it is informational only and cannot
        block the iteration further.

        The gate is skipped entirely (no score entry added) only when no acceptance
        statement implies cost optimization.

        :param passed_so_far: Whether all prior checks for this sample passed.
        :param ctx: Current optimization context.
        :return: (passed, updated_ctx) where passed reflects gate outcome.
        """
        if not bool(self._options.token_optimization):
            return passed_so_far, ctx
        passed = self._evaluate_cost(ctx)
        if passed:
            if self._baseline_cost_usd is not None and ctx.estimated_cost_usd is not None:
                rationale = (
                    f"Cost improvement gate passed: {ctx.estimated_cost_usd:.6f} is at least "
                    f"{int((1 - _COST_TOLERANCE) * 100)}% cheaper than baseline "
                    f"{self._baseline_cost_usd:.6f}."
                )
            else:
                rationale = "Cost gate passed (no baseline)."
            score = 1.0
        else:
            baseline_cost = self._baseline_cost_usd or 0.0
            rationale = (
                f"Cost improvement gate failed: {ctx.estimated_cost_usd:.6f} did not improve "
                f"by {int((1 - _COST_TOLERANCE) * 100)}% vs baseline "
                f"{baseline_cost:.6f} "
                f"(required < {baseline_cost * _COST_TOLERANCE:.6f})."
            )
            score = 0.0
        ctx = dataclasses.replace(
            ctx,
            scores={**ctx.scores, "_cost_gate": JudgeResult(
                score=score,
                rationale=rationale,
                duration_ms=ctx.duration_ms,
                estimated_cost_usd=ctx.estimated_cost_usd,
            )},
        )
        return passed_so_far and passed, ctx

    def _handle_success(
        self,
        optimize_context: OptimizationContext,
        iteration: int,
        suppress_user_callbacks: bool = False,
    ) -> Any:
        """
        Handle a successful optimization result.

        Fires the "success" status update and (unless suppressed) invokes
        on_passing_result. Pass suppress_user_callbacks=True from Phase 2 so
        the API record is updated without firing on_passing_result a second time
        — the caller is responsible for firing it once with the true final winner.

        :param optimize_context: The context from the passing iteration
        :param iteration: Current iteration number for logging
        :param suppress_user_callbacks: When True, skip on_passing_result.
        :return: The passing OptimizationContext
        """
        logger.info("[Iteration %d] -> Optimization succeeded", iteration)
        self._last_run_succeeded = True
        self._last_succeeded_context = optimize_context
        self._safe_status_update("success", optimize_context, iteration)
        if not suppress_user_callbacks and self._options.on_passing_result:
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

    def _pick_best_candidate(
        self, candidates: List[OptimizationContext]
    ) -> OptimizationContext:
        """Select the best Phase 2 candidate by normalized combined cost/latency score.

        Ranks all candidates using the sum of their normalized metrics:
            score = (duration_ms / baseline_duration_ms) + (estimated_cost_usd / baseline_cost_usd)

        Terms whose baseline or measurement is unavailable are omitted from the sum.
        The candidate with the lowest score wins. If scores are equal, the first
        candidate (earliest iteration) is returned.

        :param candidates: Non-empty list of passing Phase 2 OptimizationContexts.
        :return: The best-scoring candidate.
        """
        def _score(ctx: OptimizationContext) -> float:
            total = 0.0
            if (
                self._options.latency_optimization
                and ctx.duration_ms is not None
                and self._baseline_duration_ms is not None
                and self._baseline_duration_ms > 0
            ):
                total += ctx.duration_ms / self._baseline_duration_ms
            if (
                self._options.token_optimization
                and ctx.estimated_cost_usd is not None
                and self._baseline_cost_usd is not None
                and self._baseline_cost_usd > 0
            ):
                total += ctx.estimated_cost_usd / self._baseline_cost_usd
            return total

        return min(candidates, key=_score)

    async def _run_cost_latency_phase(
        self,
        winning_ctx: OptimizationContext,
        last_iteration: int,
        expected_response: Optional[str] = None,
    ) -> None:
        """Run Phase 2: optimize model and parameters for cost/latency on the frozen winning variation.

        The agent's content (instructions, tools) is frozen from the Phase 1 winner.
        Only model and parameters may be adjusted. One turn is executed per model
        choice (or at least two turns total), collecting passing candidates, then
        the best is selected by combined normalized cost/latency score.

        Phase 2 always uses a single turn per iteration regardless of whether the
        Phase 1 run was in GT mode — the winning user_input and variables from the
        Phase 1 winner are reused for every Phase 2 turn.

        :param winning_ctx: The Phase 1 winning OptimizationContext.
        :param last_iteration: The last iteration number from Phase 1; Phase 2
            continues from last_iteration + 1.
        :param expected_response: Optional ground truth expected response from the
            Phase 1 GT sample that corresponds to the winning context. When provided,
            injected into judge context so quality checks in Phase 2 can score against
            the ground truth just as they did in Phase 1.
        """
        self._in_cost_latency_phase = True
        self._history = [winning_ctx]
        self._current_instructions = winning_ctx.current_instructions
        self._current_parameters = winning_ctx.current_parameters.copy()
        self._current_model = winning_ctx.current_model

        frozen_variables = winning_ctx.current_variables
        frozen_user_input = winning_ctx.user_input

        # Build a deterministic, deduplicated list of models to evaluate:
        # always start from model_choices, skipping the Phase 1 winner so it
        # doesn't appear as an extra input inside the quality iteration in the
        # UI. Fall back to the Phase 1 winner only when no distinct choices
        # are provided.
        phase1_model = winning_ctx.current_model or ""
        seen_models: set = {phase1_model}
        ordered_models: List[str] = []
        for m in self._options.model_choices or []:
            if m not in seen_models:
                seen_models.add(m)
                ordered_models.append(m)
        # Fall back to Phase 1 model only if no distinct alternatives exist.
        if not ordered_models:
            ordered_models.append(phase1_model)
        # Ensure at least 2 iterations
        while len(ordered_models) < 2:
            ordered_models.append(ordered_models[-1])

        candidates: List[OptimizationContext] = []
        non_candidates: List[OptimizationContext] = []
        max_iters = len(ordered_models)
        iteration = last_iteration

        for i in range(max_iters):
            iteration += 1

            # Cycle to the next scheduled model. Instructions and parameters
            # are always reset to the Phase 1 winner's frozen values so only
            # the model varies — Phase 2 verifies the winning content still
            # passes under each candidate model.
            self._current_model = ordered_models[i]
            self._current_parameters = winning_ctx.current_parameters.copy()
            logger.info(
                "[Phase 2 Iter %d] -> Evaluating model '%s' (%d/%d)",
                iteration,
                self._current_model,
                i + 1,
                max_iters,
            )

            ctx = self._create_optimization_context(
                iteration=iteration,
                variables=frozen_variables,
                user_input=frozen_user_input,
            )

            gate_placeholders: Dict[str, JudgeResult] = {}
            if self._options.latency_optimization:
                gate_placeholders["_latency_gate"] = JudgeResult(
                    score=0.0, rationale="evaluating"
                )
            if self._options.token_optimization:
                gate_placeholders["_cost_gate"] = JudgeResult(
                    score=0.0, rationale="evaluating"
                )
            if gate_placeholders:
                ctx = dataclasses.replace(
                    ctx, scores={**ctx.scores, **gate_placeholders}
                )
            self._safe_status_update("generating", ctx, iteration)
            try:
                ctx = await asyncio.wait_for(
                    self._execute_agent_turn(ctx, iteration, expected_response=expected_response),
                    timeout=120,
                )
            except (Exception, asyncio.TimeoutError):
                logger.warning(
                    "[Phase 2 Iter %d] -> Agent call failed or timed out (model=%s); "
                    "skipping this model and trying the next",
                    iteration,
                    self._current_model,
                )
                non_candidates.append(ctx)
                continue
            self._accumulate_tokens(ctx)
            ctx = dataclasses.replace(
                ctx, accumulated_token_usage=self._total_token_usage
            )

            quality_passed = self._evaluate_response(ctx)
            quality_passed, ctx = self._apply_duration_gate(quality_passed, ctx)
            quality_passed, ctx = self._apply_cost_gate(quality_passed, ctx)

            if self._is_token_limit_exceeded():
                logger.warning(
                    "[Phase 2 Iter %d] -> Token limit exceeded; stopping Phase 2 early",
                    iteration,
                )
                non_candidates.append(ctx)
                break

            if quality_passed:
                logger.info(
                    "[Phase 2 Iter %d] -> Passed quality + gates — added to candidates",
                    iteration,
                )
                candidates.append(ctx)
            else:
                logger.info(
                    "[Phase 2 Iter %d] -> Failed quality or gates", iteration
                )
                non_candidates.append(ctx)

            if i < max_iters - 1:
                self._safe_status_update("turn completed", ctx, iteration)

        # Phase 2 is complete.  The `status` field on each result carries the
        # *run-level* outcome (PASSED / FAILED / RUNNING), not the quality of
        # that individual result — the scores already encode individual quality.
        # The backend derives the visible run status from the highest-iteration
        # result, so every Phase 2 result must end with status=PASSED after a
        # successful run; otherwise the highest-numbered result keeps the run in
        # RUNNING indefinitely.  We use _safe_status_update directly (not
        # _handle_failure / _handle_success) for non-winners so that
        # _last_run_succeeded, _last_succeeded_context, and on_failing_result are
        # not corrupted — those are reserved for run-level outcomes.
        for non_candidate_ctx in non_candidates:
            self._safe_status_update("success", non_candidate_ctx, non_candidate_ctx.iteration)

        if candidates:
            best = self._pick_best_candidate(candidates)
            # Non-best candidates: mark PASSED (run succeeded) before the winner
            # so _handle_success remains the very last update, preserving the
            # correct _last_succeeded_context for the caller.
            for other in candidates:
                if other.iteration != best.iteration:
                    self._safe_status_update("success", other, other.iteration)
            # Suppress on_passing_result here — the caller fires it once with the
            # true final winner after Phase 2 returns, so it is never double-fired.
            self._handle_success(best, best.iteration, suppress_user_callbacks=True)
            logger.info(
                "[Phase 2] -> Best candidate selected: model=%s, duration_ms=%s, cost=%s",
                best.current_model,
                f"{best.duration_ms:.0f}ms" if best.duration_ms is not None else "N/A",
                f"${best.estimated_cost_usd:.6f}" if best.estimated_cost_usd is not None else "N/A",
            )
        else:
            logger.info(
                "[Phase 2] -> No candidates passed; keeping Phase 1 winner as final result"
            )

        self._in_cost_latency_phase = False

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

        candidate = output_key if output_key else generate_slug()

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
            configs_to_search = (
                model_configs if model_configs is not None else api_client.get_model_configs(project_key)
            )
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
        if optimize_context.current_parameters:
            payload["parameters"] = optimize_context.current_parameters
        if self._initial_tool_keys:
            payload["toolKeys"] = list(self._initial_tool_keys)
        if self._initial_model_custom:
            payload["model"] = {"custom": self._initial_model_custom}

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
            self._accumulate_tokens(val_ctx)
            val_ctx = dataclasses.replace(
                val_ctx, accumulated_token_usage=self._total_token_usage
            )

            # Evaluate pass/fail before the token limit check so a passing validation
            # sample is not incorrectly treated as a failure due to budget exhaustion.
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

            if self._is_token_limit_exceeded():
                logger.error(
                    "[Validation %d/%d] -> Token limit exceeded (total=%d)",
                    i + 1,
                    validation_count,
                    self._total_token_usage,
                )
                return False, val_ctx

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
        self._total_token_usage = 0
        self._last_batch_size = 1
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
            self._accumulate_tokens(optimize_context)
            optimize_context = dataclasses.replace(
                optimize_context, accumulated_token_usage=self._total_token_usage
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

            # Token limit check after pass/fail evaluation so the persisted record
            # correctly reflects whether the iteration passed before stopping the run.
            if self._is_token_limit_exceeded():
                logger.error(
                    "[Iteration %d] -> Token limit exceeded (total=%d)",
                    iteration,
                    self._total_token_usage,
                )
                return self._handle_failure(optimize_context, iteration)

            if initial_passed:
                all_valid, last_ctx = await self._run_validation_phase(
                    optimize_context, iteration
                )
                if all_valid:
                    # Suppress on_passing_result in _handle_success; we fire it
                    # exactly once below with the true final winner so that Phase 2
                    # (if it runs) cannot cause a double callback.
                    self._handle_success(
                        optimize_context, iteration, suppress_user_callbacks=True
                    )
                    if (
                        self._options.latency_optimization
                        or self._options.token_optimization
                    ) and not self._is_token_limit_exceeded():
                        phase1_winner = self._last_succeeded_context
                        # Record the Phase 1 baseline before Phase 2 so the latency/
                        # cost gates have a reference point even when the first attempt
                        # passes (i.e. _record_baseline was never called in the loop).
                        self._record_baseline(optimize_context)
                        await self._run_cost_latency_phase(optimize_context, iteration)
                        if self._last_succeeded_context is None:
                            self._last_run_succeeded = True
                            self._last_succeeded_context = phase1_winner
                    # Fire on_passing_result exactly once with the true final winner
                    # (Phase 1 winner if Phase 2 was skipped/found nothing better,
                    # or the Phase 2 best candidate otherwise).
                    final_winner = self._last_succeeded_context
                    if final_winner and self._options.on_passing_result:
                        try:
                            self._options.on_passing_result(final_winner)
                        except Exception:
                            logger.exception(
                                "[Iteration %d] -> on_passing_result callback failed",
                                iteration,
                            )
                    return final_winner
                if self._is_token_limit_exceeded():
                    return self._handle_failure(last_ctx, iteration)
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
                # Record baseline from the main iteration context (not the failing
                # validation run) so Phase 2 gates are not skewed by a bad sample.
                self._record_baseline(optimize_context)
                self._history.append(last_ctx)
                self._history = _trim_history(self._history, _MAX_STANDARD_HISTORY_LENGTH)
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
            self._record_baseline(optimize_context)
            self._history.append(optimize_context)
            self._history = _trim_history(self._history, _MAX_STANDARD_HISTORY_LENGTH)
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
