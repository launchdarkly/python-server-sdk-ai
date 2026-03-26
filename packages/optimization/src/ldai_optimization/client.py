"""Client for LaunchDarkly AI agent optimization."""

from typing import Any, Dict, List, Optional
import dataclasses
import os
import logging
import random
import json

from ldai import LDAIClient, AIJudgeConfigDefault, AIAgentConfig

from ldai_optimization.dataclasses import (
    AutoCommitConfig,
    JudgeResult,
    Message,
    OptimizationContext,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
)
from ldai_optimization.util import (
    await_if_needed,
    create_evaluation_tool,
    create_variation_tool,
    extract_json_from_response,
)

logger = logging.getLogger(__name__)


class OptimizationClient:
    _options: OptimizationOptions
    _ldClient: LDAIClient
    _has_api_key: bool
    _api_key: Optional[str]
    _agent_key: str

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
        self._current_parameters: Dict[str, Any] = agent_config.model._parameters or {}
        self._current_model: Optional[str] = (
            agent_config.model.name if agent_config.model else None
        )
        self._history: List[OptimizationContext] = []

    def _create_optimization_context(
        self,
        iteration: int,
        user_input: Optional[str] = None,
        completion_response: str = "",
        scores: Optional[Dict[str, JudgeResult]] = None,
    ) -> OptimizationContext:
        """
        Create an OptimizeContext with current state.

        :param iteration: Current iteration number
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
            current_model=self._current_model,
            user_input=user_input,
            history=tuple(flat_history),
            iteration=iteration,
        )

    def _safe_status_update(
        self, status: str, context: OptimizationContext, iteration: int
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
                    "[Turn %d] -> on_status_update callback failed", iteration
                )

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

        logger.info("[Turn %d] -> Executing evaluation...", iteration)
        reasoning_history = self._build_reasoning_history()
        judge_results: Dict[str, JudgeResult] = {}

        for judge_key, optimization_judge in self._options.judges.items():
            try:
                if optimization_judge.judge_key is not None:
                    result = await self._evaluate_config_judge(
                        judge_key,
                        optimization_judge,
                        completion_response,
                        iteration,
                        reasoning_history,
                    )
                    judge_results[judge_key] = result
                else:
                    result = await self._evaluate_acceptance_judge(
                        judge_key,
                        optimization_judge,
                        completion_response,
                        iteration,
                        reasoning_history,
                    )
                    judge_results[judge_key] = result
            except Exception as e:
                logger.exception(
                    "[Turn %d] -> Judge %s evaluation failed", iteration, judge_key
                )
                judge_results[judge_key] = JudgeResult(score=0.0, rationale=None)

        judge_results_json = self._serialize_scores(judge_results)
        logger.info(
            "[Turn %d] -> Evaluation result: %s",
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
        # Config-type judge: fetch judge config on-demand from LaunchDarkly SDK
        input_text = self._current_instructions or ""
        # Combine current instructions with reasoning history for message_history
        message_history_text = self._build_message_history_text(
            input_text, reasoning_history
        )

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
            logger.warning(
                "[Turn %d] -> Judge %s is disabled",
                iteration,
                optimization_judge.judge_key,
            )
            return JudgeResult(score=0.0, rationale=None)

        if not judge_config.messages:
            logger.warning(
                "[Turn %d] -> Judge %s has no messages",
                iteration,
                optimization_judge.judge_key,
            )
            return JudgeResult(score=0.0, rationale=None)

        # Convert LDMessage to Message objects, appending structured output instruction to system messages
        judge_messages = []
        for msg in judge_config.messages:
            content = msg.content
            if msg.role == "system":
                content += " Use the structured output tool to format your response. You should always return a JSON object with a score and rationale."
            judge_messages.append(Message(role=msg.role, content=content))

        # Build parameters from judge config, hoisting any pre-existing tools so we can append ours
        parameters = {}
        tools = []
        if judge_config.model:
            parameters["model"] = judge_config.model.name
            if judge_config.model._parameters:
                # Extract tools if present
                existing_tools = judge_config.model._parameters.get("tools")
                if existing_tools:
                    tools = (
                        existing_tools
                        if isinstance(existing_tools, list)
                        else [existing_tools]
                    )
                    # Convert to dicts if needed
                    tools = [
                        tool.to_dict() if hasattr(tool, "to_dict") else tool
                        for tool in tools
                    ]
                # Copy parameters excluding tools
                parameters.update(
                    {
                        k: v
                        for k, v in judge_config.model._parameters.items()
                        if k != "tools"
                    }
                )

        # Add structured output tool for score and rationale
        tools.append(create_evaluation_tool().to_dict())

        judge_ctx = OptimizationJudgeContext(
            messages=judge_messages,
            parameters=parameters,
            tools=tools,
        )

        result = self._options.handle_judge_call(self._options.judge_model, judge_ctx)
        judge_response_str = await await_if_needed(result)

        logger.info(
            "[Turn %d] -> Judge response (%s): %s",
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
            logger.error(
                "[Turn %d] -> Judge %s has no acceptance_statement",
                iteration,
                judge_key,
            )
            return JudgeResult(score=0.0, rationale=None)

        # Build message history with reasoning for context
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
        evaluation_tool = create_evaluation_tool()

        judge_ctx = OptimizationJudgeContext(
            messages=judge_messages,
            parameters={"model": self._options.judge_model},
            tools=[evaluation_tool.to_dict()],
        )

        result = self._options.handle_judge_call(self._options.judge_model, judge_ctx)
        judge_response = await await_if_needed(result)

        logger.info(
            "[Turn %d] -> Judge response (%s): %s", iteration, judge_key, judge_response
        )

        # Parse judge response — expect structured JSON output with score and rationale
        return self._parse_judge_response(
            judge_response, judge_key, judge_key, iteration, clamp_score=True
        )

    async def _get_agent_config(self, agent_key: str) -> AIAgentConfig:
        """
        Get the agent configuration from the LaunchDarkly client.

        :param agent_key: The key for the agent to get the configuration for
        :return: The agent configuration
        """
        try:
            agent_config = await self._ldClient.agent_config(agent_key)
            self._initialize_class_members(agent_config)
            return agent_config
        except Exception as e:
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
        agent_config = await self._get_agent_config(agent_key)
        return await self._run_optimization(agent_config, options)

    def _build_new_variation_prompt(
        self, previous_ctx: Optional[OptimizationContext]
    ) -> str:
        """
        Build the LLM prompt for generating an improved agent configuration.

        Constructs a detailed instruction string based on the previous iteration's
        configuration, completion result, and judge scores. When no previous context
        exists (first variation attempt), asks the LLM to improve the current config
        without evaluation feedback.

        :param previous_ctx: The most recent OptimizationContext, or None on the first attempt
        :return: The assembled prompt string
        """
        sections = [
            self._new_variation_prompt_preamble(),
            self._new_variation_prompt_configuration(previous_ctx),
            self._new_variation_prompt_feedback(previous_ctx),
            self._new_variation_prompt_improvement_instructions(previous_ctx),
        ]
        return "\n\n".join(s for s in sections if s)

    def _new_variation_prompt_preamble(self) -> str:
        """Static opening section for the variation generation prompt."""
        return "\n".join([
            "You are an assistant that helps improve agent configurations through iterative optimization.",
            "",
            "Your task is to generate improved agent instructions and parameters based on the feedback provided.",
        ])

    def _new_variation_prompt_configuration(
        self, previous_ctx: Optional[OptimizationContext]
    ) -> str:
        """
        Configuration section of the variation prompt.

        Shows the previous iteration's model, instructions, parameters, and completion
        response when available, or the current instance state on the first attempt.
        """
        if previous_ctx:
            return "\n".join([
                "## Previous Configuration:",
                f"Model: {previous_ctx.current_model}",
                f"Instructions: {previous_ctx.current_instructions}",
                f"Parameters: {previous_ctx.current_parameters}",
                "",
                "## Previous Result:",
                previous_ctx.completion_response,
            ])
        else:
            return "\n".join([
                "## Current Configuration:",
                f"Model: {self._current_model}",
                f"Instructions: {self._current_instructions}",
                f"Parameters: {self._current_parameters}",
            ])

    def _new_variation_prompt_feedback(
        self, previous_ctx: Optional[OptimizationContext]
    ) -> str:
        """
        Evaluation feedback section of the variation prompt.

        Returns an empty string when there are no scores so it is filtered out
        of the assembled prompt entirely.
        """
        if not previous_ctx or not previous_ctx.scores:
            return ""

        lines = ["## Evaluation Feedback:"]
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
                    lines.append(feedback_line)
                else:
                    passed = score >= 1.0
                    status = "PASSED" if passed else "FAILED"
                    feedback_line = f"- {judge_key}: {status}"
                    if result.rationale:
                        feedback_line += f"\n  Reasoning: {result.rationale}"
                    lines.append(feedback_line)
        return "\n".join(lines)

    def _new_variation_prompt_improvement_instructions(
        self, previous_ctx: Optional[OptimizationContext]
    ) -> str:
        """
        Improvement instructions section of the variation prompt.

        Includes model-choice guidance and the required output format schema.
        When previous_ctx is provided, adds feedback-driven improvement directives.
        """
        model_instructions = "\n".join([
            "You may also choose to change the model if you believe that the current model is not performing well or a different model would be better suited for the task. "
            f"Here are the models you may choose from: {self._options.model_choices}. You must always return a model property, even if it's the same as the current model.",
            "When suggesting a new model, you should provide a rationale for why you believe the new model would be better suited for the task.",
        ])

        parameters_instructions = "\n".join([
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
        ])

        if previous_ctx:
            return "\n".join([
                "## Improvement Instructions:",
                "Based on the evaluation feedback above, generate improved agent instructions and parameters.",
                "Focus on addressing the areas where the evaluation failed or scored below threshold.",
                "The new configuration should aim to improve the agent's performance on the evaluation criteria.",
                model_instructions,
                "",
                "Return the improved configuration in a structured format that can be parsed to update:",
                "1. The agent instructions (current_instructions)",
                "2. The agent parameters (current_parameters)",
                "3. The model (model) - you must always return a model, even if it's the same as the current model.",
                parameters_instructions,
            ])
        else:
            return "\n".join([
                "Generate an improved version of this configuration.",
                model_instructions,
                parameters_instructions,
            ])

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
            logger.error(
                "[Turn %d] -> Response missing required fields: %s. Received fields: %s. Full response_data: %s",
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
                "[Turn %d] -> Model field is empty or None in response, keeping current model %s",
                iteration,
                self._current_model,
            )
        elif model_value not in self._options.model_choices:
            logger.warning(
                "[Turn %d] -> Model '%s' not in model_choices %s, keeping current model %s",
                iteration,
                model_value,
                self._options.model_choices,
                self._current_model,
            )
        else:
            old_model = self._current_model
            self._current_model = model_value
            logger.info(
                "[Turn %d] -> Model updated from '%s' to '%s'",
                iteration,
                old_model,
                self._current_model,
            )

        logger.info(
            "[Turn %d] -> New variation generated: instructions='%.100s...', model=%s, parameters=%s",
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
            current_model=self._current_model,
            user_input=None,
            history=variation_ctx.history,
            iteration=variation_ctx.iteration,
            structured_output_tool=variation_ctx.structured_output_tool,
        )

    async def _generate_new_variation(self, iteration: int) -> OptimizationContext:
        """
        Generate new variation for next iteration (auto-path).

        Calls handle_agent_call to generate a new variation and updates current_instructions
        and current_parameters based on the returned OptimizeContext.

        :param iteration: The current iteration number for logging
        """
        logger.info("[Turn %d] -> Generating new variation...", iteration)

        # Create a context for status update before generating the variation
        status_ctx = self._create_optimization_context(iteration=iteration)
        self._safe_status_update("generating variation", status_ctx, iteration)

        # Get the most recent context for previous result and feedback
        previous_ctx = self._history[-1] if self._history else None

        instructions = self._build_new_variation_prompt(previous_ctx)

        # Create structured output tool definition for variation generation
        structured_output_tool = create_variation_tool(self._options.model_choices)

        # Create a flat history list (without nested history) to avoid exponential growth
        flat_history = [prev_ctx.copy_without_history() for prev_ctx in self._history]

        # Create context for variation generation — low temperature for deterministic output
        variation_ctx = OptimizationContext(
            scores={},
            completion_response="",
            current_instructions=instructions,
            current_parameters={"temperature": 0.1},
            current_model=self._current_model,
            user_input=None,
            history=tuple(flat_history),
            iteration=len(self._history) + 1,
            structured_output_tool=structured_output_tool,
        )

        # Call handle_agent_call to generate new variation; expects a JSON string
        # matching the structured output schema (current_instructions, current_parameters, model)
        result = self._options.handle_agent_call(self._agent_key, variation_ctx)
        response_str = await await_if_needed(result)

        # Extract and update current state from the parsed response
        response_data = extract_json_from_response(response_str)
        return self._apply_new_variation_response(
            response_data, variation_ctx, response_str, iteration
        )

    async def optimize_from_config(
        self, agent_key: str, optimization_config_key: str
    ) -> Any:
        """Optimize an agent from a configuration.

        :param agent_key: Identifier of the agent to optimize.
        :param optimization_config_key: Identifier of the optimization configuration to use.
        :return: Optimization result.
        """
        if not self._has_api_key:
            raise ValueError(
                "LAUNCHDARKLY_API_KEY is not set, so optimize_from_config is not available"
            )

        self._agent_key = agent_key
        agent_config = await self._get_agent_config(agent_key)

        raise NotImplementedError

    async def _execute_agent_turn(
        self,
        optimize_context: OptimizationContext,
        iteration: int,
    ) -> OptimizationContext:
        """
        Run the agent call and judge scoring for one optimization turn.

        Returns a new OptimizationContext with completion_response and scores
        populated, leaving the input context unchanged.

        :param optimize_context: The context for this turn (instructions, model, history, etc.)
        :param iteration: Current iteration number for logging and status callbacks
        :return: Updated context with completion_response and scores filled in
        """
        try:
            result = self._options.handle_agent_call(self._agent_key, optimize_context)
            completion_response = await await_if_needed(result)
        except Exception:
            logger.exception("[Turn %d] -> Agent call failed", iteration)
            if self._options.on_failing_result:
                self._options.on_failing_result(optimize_context)
            raise

        scores: Dict[str, JudgeResult] = {}
        if self._options.judges:
            self._safe_status_update("evaluating", optimize_context, iteration)
            scores = await self._call_judges(completion_response, iteration)

        return dataclasses.replace(
            optimize_context,
            completion_response=completion_response,
            scores=scores,
        )

    async def _run_optimization(
        self, agent_config: AIAgentConfig, options: OptimizationOptions
    ) -> Any:
        """Run an optimization on the given agent with the given options.

        :param agent_config: Agent configuration from LaunchDarkly.
        :param options: Optimization options.
        :return: Optimization result.
        """
        self._options = options
        self._initialize_class_members_from_config(agent_config)

        initial_context = self._create_optimization_context(
            iteration=0,
        )

        self._safe_status_update("init", initial_context, 0)

        iteration = 0
        while True:
            iteration += 1
            logger.info("[Turn %d] -> Starting", iteration)
            user_input = None
            if self._options.user_input_options:
                user_input = random.choice(self._options.user_input_options)

            optimize_context = self._create_optimization_context(
                iteration=iteration,
                user_input=user_input,
            )

            self._safe_status_update("generating", optimize_context, iteration)
            optimize_context = await self._execute_agent_turn(optimize_context, iteration)

            # Manual path: on_turn callback gives caller full control over pass/fail
            if self._options.on_turn is not None:
                try:
                    on_turn_result = self._options.on_turn(optimize_context)
                    if on_turn_result:
                        # on_turn returned True — success
                        return self._handle_success(optimize_context, iteration)
                    else:
                        # on_turn returned False — generate new variation and continue
                        if iteration >= self._options.max_attempts:
                            return self._handle_failure(optimize_context, iteration)
                        self._history.append(optimize_context)
                        await self._generate_new_variation(iteration)
                        # Notify before starting next turn
                        self._safe_status_update(
                            "turn completed", optimize_context, iteration
                        )
                        continue
                except Exception as e:
                    logger.exception(
                        "[Turn %d] -> on_turn evaluation failed", iteration
                    )
                    self._history.append(optimize_context)
                    await self._generate_new_variation(iteration)
                    if iteration >= self._options.max_attempts:
                        return self._handle_failure(optimize_context, iteration)
                    self._safe_status_update(
                        "turn completed", optimize_context, iteration
                    )
                    continue
            else:
                # Auto-path: judge scores determine pass/fail via _evaluate_response
                passes = self._evaluate_response(optimize_context)
                if passes:
                    return self._handle_success(optimize_context, iteration)
                else:
                    self._history.append(optimize_context)
                    await self._generate_new_variation(iteration)
                    # Check max_attempts after generating variation
                    if iteration >= self._options.max_attempts:
                        return self._handle_failure(optimize_context, iteration)
                    self._safe_status_update(
                        "turn completed", optimize_context, iteration
                    )
                    continue
