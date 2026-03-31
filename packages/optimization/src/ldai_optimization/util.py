"""Utility functions for the LaunchDarkly AI optimization package."""

import inspect
import json
import logging
import re
from typing import Any, Awaitable, Dict, List, Optional, Union

from ldai_optimization.dataclasses import ToolDefinition

logger = logging.getLogger(__name__)


def handle_evaluation_tool_call(score: float, rationale: str) -> str:
    """
    Process the return_evaluation tool call from the judge LLM.

    Serialises the score and rationale to a JSON string. The caller
    (handle_judge_call implementor) should return this string as the result of
    the judge turn; the framework will then parse it via _parse_judge_response
    to extract the score and rationale.

    :param score: The evaluation score (0.0 to 1.0)
    :param rationale: Explanation of the evaluation decision
    :return: JSON string of the score and rationale
    """
    return json.dumps({"score": score, "rationale": rationale})


def handle_variation_tool_call(
    current_instructions: str,
    current_parameters: Dict[str, Any],
    model: str,
) -> str:
    """
    Process the return_improved_configuration tool call from the variation LLM.

    Serialises the improved configuration to a JSON string. The caller
    (handle_agent_call implementor) should return this string as the result of
    the variation agent turn; the framework will then parse it via
    extract_json_from_response and apply it in _apply_new_variation_response.

    :param current_instructions: The improved agent instructions
    :param current_parameters: The improved agent parameters (e.g. temperature, max_tokens)
    :param model: The model to use for the improved agent
    :return: JSON string of the improved configuration
    """
    return json.dumps({
        "current_instructions": current_instructions,
        "current_parameters": current_parameters,
        "model": model,
    })


def interpolate_variables(text: str, variables: Dict[str, Any]) -> str:
    """
    Interpolate ``{{variable}}`` placeholders in text using the provided variables.

    Matches LaunchDarkly's Mustache-style template format so that manually
    generated variation instructions use the same syntax as LD-fetched templates.
    Unrecognised placeholders are left unchanged.

    :param text: Template string potentially containing ``{{key}}`` placeholders
    :param variables: Mapping of variable names to their replacement values
    :return: Text with all recognised placeholders replaced
    """
    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        return str(variables[key]) if key in variables else match.group(0)

    return re.sub(r"\{\{(\w+)\}\}", replace, text)


async def await_if_needed(
    result: Union[str, Awaitable[str]]
) -> str:
    """
    Handle both sync and async callable results.

    :param result: Either a string or an awaitable that returns a string
    :return: The string result
    """
    if isinstance(result, str):
        return result
    return await result


def create_evaluation_tool() -> ToolDefinition:
    """
    Create the structured output tool for judge evaluations.

    :return: A ToolDefinition for evaluation responses
    """
    return ToolDefinition(
        type="function",
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


def create_boolean_tool() -> ToolDefinition:
    """
    Create the structured output tool for acceptance judges.

    :return: A ToolDefinition for boolean evaluation responses
    """
    return ToolDefinition(
        type="function",
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


def create_variation_tool(model_choices: List[str]) -> ToolDefinition:
    """
    Create the structured output tool for variation generation.

    :param model_choices: List of model IDs the LLM may select from
    :return: A ToolDefinition for variation generation responses
    """
    return ToolDefinition(
        type="function",
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
                    "enum": model_choices,
                },
            },
            "required": ["current_instructions", "current_parameters", "model"],
            "additionalProperties": False,
        },
    )


def extract_json_from_response(response_str: str) -> Dict[str, Any]:
    """
    Parse a JSON object from an LLM response string.

    Attempts direct JSON parsing first, then progressively falls back to
    extracting JSON from markdown code blocks and balanced-brace scanning.

    :param response_str: Raw string response from an LLM
    :return: Parsed dictionary
    :raises ValueError: If no valid JSON object can be extracted
    """
    # Try direct parse first
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        pass

    response_data: Optional[Dict[str, Any]] = None

    # Try to extract JSON from markdown code blocks
    code_block_match = re.search(
        r'```(?:json)?\s*(\{.*?\})\s*```',
        response_str,
        re.DOTALL,
    )
    if code_block_match:
        try:
            response_data = json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try balanced-brace scanning
    if response_data is None:
        brace_count = 0
        start_idx = response_str.find('{')
        if start_idx != -1:
            for i in range(start_idx, len(response_str)):
                if response_str[i] == '{':
                    brace_count += 1
                elif response_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_str[start_idx:i + 1]
                        try:
                            response_data = json.loads(json_str)
                            break
                        except json.JSONDecodeError:
                            start_idx = response_str.find('{', start_idx + 1)
                            if start_idx == -1:
                                break
                            brace_count = 0

    # Legacy regex fallback
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
                logger.debug(
                    "Extracted JSON string failed to parse: %s",
                    json_match.group()[:200],
                )
                raise ValueError(
                    "Failed to parse extracted JSON from variation generation response"
                )

    if response_data is None:
        logger.debug(
            "Failed to extract JSON from response. "
            "Response length: %d, response: %s",
            len(response_str),
            response_str,
        )
        raise ValueError(
            "Failed to parse structured output from variation generation. "
            "Expected JSON object with 'current_instructions', 'current_parameters', and 'model' fields."
            "Response length: %d",
            len(response_str),
        )

    return response_data
