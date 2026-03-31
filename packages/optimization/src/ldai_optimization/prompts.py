"""Prompt-building functions for LaunchDarkly AI optimization."""

from typing import Any, Dict, List, Optional

from ldai_optimization.dataclasses import (
    OptimizationContext,
    OptimizationJudge,
)


def build_message_history_text(
    history: List[OptimizationContext],
    input_text: str,
    reasoning_history: str,
    current_user_input: str,
) -> str:
    """
    Build a formatted message-history string for use as a judge template variable.

    Combines the current instructions (system text), the conversation turns
    recorded in history, the current turn's user question, and the accumulated
    reasoning/score history.

    :param history: All previous OptimizationContexts, oldest first
    :param input_text: Current system instructions (may be empty string)
    :param reasoning_history: Pre-formatted string from build_reasoning_history
    :param current_user_input: The user question for the turn being evaluated.
        Must be passed explicitly because the current turn is not yet in
        history when the judge runs.
    :return: Combined string to substitute into the judge's message_history variable
    """
    turn_messages = []
    for ctx in history:
        if ctx.user_input:
            turn_messages.append(f"User: {ctx.user_input}")
        if ctx.completion_response:
            turn_messages.append(f"Assistant: {ctx.completion_response}")

    # Include the current turn's question so judges see what was actually asked
    turn_messages.append(f"User: {current_user_input}")

    parts = []
    if input_text:
        parts.append(f"System: {input_text}")
    if turn_messages:
        parts.append("\n".join(turn_messages))
    if reasoning_history:
        parts.append(f"Evaluation history:\n{reasoning_history}")

    return "\n\n".join(parts)


def build_reasoning_history(history: List[OptimizationContext]) -> str:
    """
    Build a formatted string of reasoning from previous iterations.

    :param history: All previous OptimizationContexts, oldest first
    :return: Formatted string containing reasoning history
    """
    if not history:
        return ""

    reasoning_parts = []
    for i, prev_ctx in enumerate(history, 1):
        if prev_ctx.scores:
            reasoning_parts.append(f"## Iteration {i} Judge Evaluations:")
            for judge_key, result in prev_ctx.scores.items():
                reasoning_parts.append(f"- {judge_key}: Score {result.score}")
                if result.rationale:
                    reasoning_parts.append(f"  Reasoning: {result.rationale}")
            reasoning_parts.append("")

    return "\n".join(reasoning_parts)


def build_new_variation_prompt(
    history: List[OptimizationContext],
    judges: Optional[Dict[str, OptimizationJudge]],
    current_model: Optional[str],
    current_instructions: str,
    current_parameters: Dict[str, Any],
    model_choices: List[str],
    variable_choices: List[Dict[str, Any]],
    initial_instructions: str,
) -> str:
    """
    Build the LLM prompt for generating an improved agent configuration.

    Constructs a detailed instruction string based on the full optimization
    history, including all previous configurations, completion results, and
    judge scores. When history is empty (first variation attempt), asks the
    LLM to improve the current config without evaluation feedback.

    :param history: All previous OptimizationContexts, oldest first. Empty on the first attempt.
    :param judges: Judge configuration dict from OptimizationOptions
    :param current_model: The model currently in use
    :param current_instructions: The current agent instructions template
    :param current_parameters: The current model parameters dict
    :param model_choices: List of model IDs the LLM may select from
    :param variable_choices: List of variable dicts (used to derive placeholder names)
    :param initial_instructions: The original unmodified instructions template
    :return: The assembled prompt string
    """
    sections = [
        variation_prompt_preamble(),
        variation_prompt_acceptance_criteria(judges),
        variation_prompt_configuration(
            history, current_model, current_instructions, current_parameters
        ),
        variation_prompt_feedback(history, judges),
        variation_prompt_improvement_instructions(
            history, model_choices, variable_choices, initial_instructions
        ),
    ]

    return "\n\n".join(s for s in sections if s)


def variation_prompt_preamble() -> str:
    """Static opening section for the variation generation prompt."""
    return "\n".join(
        [
            "You are an assistant that helps improve agent configurations through iterative optimization.",
            "",
            "Your task is to generate improved agent instructions and parameters based on the feedback provided.",
            "The feedback you provide should guide the LLM to improve the agent instructions "
            "for all possible use cases, not one concrete case.",
            "For example, if the feedback is that the agent is not returning the correct records, "
            "you should improve the agent instructions to return the correct records for all possible use cases. "
            "Not just the one concrete case that was provided in the feedback.",
            "When changing the instructions, keep the original intent in mind "
            "when it comes to things like the use of variables and placeholders.",
            "If the original instructions were to use a placeholder like {{id}}, "
            "you should keep the placeholder in the new instructions, not replace it with the actual value. "
            "This is the case for all parameterized values (all parameters should appear in each new variation).",
            "Pay particular attention to the instructions regarding tools and the rules for variables.",
        ]
    )


def variation_prompt_acceptance_criteria(
    judges: Optional[Dict[str, OptimizationJudge]],
) -> str:
    """
    Acceptance criteria section of the variation prompt.

    Collects every acceptance statement defined across all judges and renders
    them as an emphatic block so the LLM understands exactly what the improved
    configuration must achieve. Returns an empty string when no judges carry
    acceptance statements (e.g. all judges are config-key-only judges).
    """
    if not judges:
        return ""

    statements = [
        (key, judge.acceptance_statement)
        for key, judge in judges.items()
        if judge.acceptance_statement
    ]

    if not statements:
        return ""

    lines = [
        "## *** ACCEPTANCE CRITERIA (MUST BE MET) ***",
        "The improved configuration MUST produce responses that satisfy ALL of the following criteria.",
        "These criteria are non-negotiable — every generated variation will be evaluated against them.",
        "All variables must be used in the new instructions."
        "",
    ]
    for key, statement in statements:
        lines.append(f"- [{key}] {statement}")

    lines += [
        "",
        "When writing new instructions, explicitly address each criterion above.",
        "Do not sacrifice any criterion in favour of another.",
    ]

    return "\n".join(lines)


def variation_prompt_configuration(
    history: List[OptimizationContext],
    current_model: Optional[str],
    current_instructions: str,
    current_parameters: Dict[str, Any],
) -> str:
    """
    Configuration section of the variation prompt.

    Shows the most recent iteration's model, instructions, parameters,
    user input, and completion response when history is available, or the
    current state on the first attempt.
    """
    if history:
        previous_ctx = history[-1]
        lines = [
            "## Most Recent Configuration:",
            f"Model: {previous_ctx.current_model}",
            f"Instructions: {previous_ctx.current_instructions}",
            f"Parameters: {previous_ctx.current_parameters}",
            "",
            "## Most Recent Result:",
        ]
        if previous_ctx.user_input:
            lines.append(f"User question: {previous_ctx.user_input}")
        lines.append(f"Agent response: {previous_ctx.completion_response}")
        return "\n".join(lines)
    else:
        return "\n".join(
            [
                "## Current Configuration:",
                f"Model: {current_model}",
                f"Instructions: {current_instructions}",
                f"Parameters: {current_parameters}",
            ]
        )


def variation_prompt_feedback(
    history: List[OptimizationContext],
    judges: Optional[Dict[str, OptimizationJudge]],
) -> str:
    """
    Evaluation feedback section of the variation prompt.

    Renders all previous iterations' scores in chronological order so the
    LLM can observe trends across the full optimization run. Returns an
    empty string when no history exists or no iteration has scores, so it
    is filtered out of the assembled prompt entirely.
    """
    iterations_with_scores = [ctx for ctx in history if ctx.scores]
    if not iterations_with_scores:
        return ""

    lines = ["## Evaluation History:"]
    for ctx in iterations_with_scores:
        lines.append(f"\n### Iteration {ctx.iteration}:")
        if ctx.user_input:
            lines.append(f"User question: {ctx.user_input}")
        for judge_key, result in ctx.scores.items():
            optimization_judge = judges.get(judge_key) if judges else None
            if optimization_judge:
                score = result.score
                if optimization_judge.threshold is not None:
                    passed = score >= optimization_judge.threshold
                    status = "PASSED" if passed else "FAILED"
                    feedback_line = (
                        f"- {judge_key}: Score {score:.3f}"
                        f" (threshold: {optimization_judge.threshold}) - {status}"
                    )
                else:
                    passed = score >= 1.0
                    status = "PASSED" if passed else "FAILED"
                    feedback_line = f"- {judge_key}: {status}"
                if result.rationale:
                    feedback_line += f"\n  Reasoning: {result.rationale}"
                lines.append(feedback_line)
    return "\n".join(lines)


def variation_prompt_improvement_instructions(
    history: List[OptimizationContext],
    model_choices: List[str],
    variable_choices: List[Dict[str, Any]],
    initial_instructions: str,
) -> str:
    """
    Improvement instructions section of the variation prompt.

    Includes model-choice guidance, prompt variable rules, and the required
    output format schema. When history is non-empty, adds feedback-driven
    improvement directives.
    """
    model_instructions = "\n".join(
        [
            "You may also choose to change the model if you believe that the current model is "
            "not performing well or a different model would be better suited for the task. "
            f"Here are the models you may choose from: {model_choices}. "
            "You must always return a model property, even if it's the same as the current model.",
            "When suggesting a new model, you should provide a rationale for why you believe "
            "the new model would be better suited for the task.",
        ]
    )

    # Collect unique variable keys across all variable_choices entries
    variable_keys: set = set()
    for choice in variable_choices:
        variable_keys.update(choice.keys())
    placeholder_list = ", ".join(f"{{{{{k}}}}}" for k in sorted(variable_keys))

    variable_instructions = "\n".join(
        [
            "## Prompt Variables:",
            "These variables are substituted into the instructions at call time using {{variable_name}} syntax.",
            "Rules:",
            "- If the {{variable_name}} placeholder is not present in the current instructions, "
            "you should include it where logically appropriate.",
            "Here are the original instructions so that you can see how the "
            "placeholders are used and which are available:",
            "\nSTART:" "\n" + initial_instructions + "\n",
            "\nEND OF ORIGINAL INSTRUCTIONS\n",
            "The following prompt variables are available and are the only "
            f"variables that should be used: {placeholder_list}",
            "Here is an example of a good response if an {{id}} placeholder is available: "
            "'Select records matching id {{id}}'",
            "Here is an example of a bad response if an {{id}} placeholder is available: "
            "'Select records matching id 1232'",
            "Here is an example of a good response if a {{resource_id}} and {{resource_type}} "
            "placeholder are available: "
            "'Select records matching id {{resource_id}} and type {{resource_type}}'",
            "Here is an example of a bad response if a {{resource_id}} and {{resource_type}} "
            "placeholder are available: "
            "'Select records matching id 1232 and type {{resource_type}}'",
            "Here is another example of a bad response if a {{resource_id}} and {{resource_type}} "
            "placeholder are available: "
            "'Select records matching id {{resource_id}} and type {{resource-123}}'",
            "The above example is incorrect because the resource-123 is not a valid variable name.",
            "To fix the above example, you would instead use {{resource_type}} and {{resource_id}}",
        ]
    )

    tool_instructions = "\n".join(
        [
            "## Tool Format:",
            'If the current configuration includes tools, you MUST return them '
            'unchanged in current_parameters["tools"].',
            "Do NOT include internal framework tools such as the evaluation tool or structured output tool.",
            "Each tool must follow this exact format:",
            "{",
            '  "name": "tool-name",',
            '  "type": "function",',
            '  "description": "What the tool does",',
            '  "parameters": {',
            '    "type": "object",',
            '    "properties": {',
            '      "param_name": {',
            '        "type": "type of the input parameter",',
            '        "description": "Description of the parameter"',
            "      }",
            "    },",
            '    "required": ["param_name"],',
            '    "additionalProperties": false',
            "  }",
            "}",
            "Example:",
            "{",
            '  "name": "user-preferences-lookup",',
            '  "type": "function",',
            '  "description": "Looks up user preferences by ID",',
            '  "parameters": {',
            '    "type": "object",',
            '    "properties": {',
            '      "user_id": {',
            '        "type": "string",',
            '        "description": "The user id"',
            "      }",
            "    },",
            '    "required": ["user_id"],',
            '    "additionalProperties": false',
            "  }",
            "}"
            "Always call the return_improved_configuration tool to format the response.",
            "Return the response as-is from the return_improved_configuration tool,",
            "do not modify it in any way.",
        ]
    )

    parameters_instructions = "\n".join(
        [
            "Return these values in a JSON object with the following keys: "
            "current_instructions, current_parameters, and model.",
            "Example:",
            "{",
            '  "current_instructions": "...',
            '  "current_parameters": {',
            '    "...": "..."',
            "  },",
            '  "model": "gpt-4o"',
            "}",
            "Parameters should only be things that are directly parseable by an LLM call, "
            "for example, temperature, max_tokens, etc.",
            "Do not include any other parameters that are not directly parseable by an LLM call. "
            "If you want to provide instruction for tone or other attributes, "
            "provide them directly in the instructions.",
        ]
    )

    if history:
        return "\n".join(
            [
                "## Improvement Instructions:",
                "Based on the evaluation history above, generate improved agent instructions and parameters.",
                "Focus on addressing the areas where the evaluation failed or scored below threshold.",
                "The new configuration should aim to improve the agent's performance on the evaluation criteria.",
                model_instructions,
                "",
                variable_instructions,
                "",
                tool_instructions,
                "",
                "Return the improved configuration in a structured format that can be parsed to update:",
                "1. The agent instructions (current_instructions)",
                "2. The agent parameters (current_parameters)",
                "3. The model (model) - you must always return a model, "
                "even if it's the same as the current model.",
                "4. You should return the tools the user has defined, as-is, on the new parameters. "
                "Do not modify them, but make sure you do not include internal tools like "
                "the evaluation tool or structured output tool.",
                parameters_instructions,
            ]
        )
    else:
        return "\n".join(
            [
                "Generate an improved version of this configuration.",
                model_instructions,
                "",
                variable_instructions,
                "",
                tool_instructions,
                "",
                parameters_instructions,
            ]
        )
