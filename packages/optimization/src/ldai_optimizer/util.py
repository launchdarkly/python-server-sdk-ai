"""Utility functions for the LaunchDarkly AI optimization package."""

import inspect
import json
import logging
import random
import re
from typing import Any, Awaitable, Dict, List, Optional, Tuple, TypeVar, Union

from ldai_optimizer._slug_words import _ADJECTIVES, _NOUNS
from ldai_optimizer.dataclasses import ToolDefinition

logger = logging.getLogger(__name__)

# Matches LaunchDarkly API key and SDK key formats:
#   api-<hex/alphanumeric, 16+ chars>
#   sdk-<hex/alphanumeric, 16+ chars>
#   cli-<hex/alphanumeric, 16+ chars>
_KEY_PATTERN = re.compile(r"\b(api|sdk|cli)-[A-Za-z0-9_\-]{16,}\b")


class RedactionFilter(logging.Filter):
    """Logging filter that redacts strings resembling LaunchDarkly API keys.

    Scrubs both the format string (``record.msg``) and each positional argument
    (``record.args``) before the handler formats the final log line, so raw key
    values are never written to any log destination.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _KEY_PATTERN.sub("[REDACTED]", str(record.msg))
        if record.args:
            record.args = tuple(
                _KEY_PATTERN.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
                for a in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
        return True


logger.addFilter(RedactionFilter())


def generate_slug() -> str:
    """Generate a random ``adjective-noun`` slug (e.g. ``blazing-lobster``).

    Produces the same format as ``coolname.generate_slug(2)`` using an
    internal word list, removing the external dependency.

    :return: A hyphen-joined two-word lowercase string.
    """
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


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

    return re.sub(r"\{\{([\w-]+)\}\}", replace, text)


def restore_variable_placeholders(
    text: str,
    variable_choices: List[Dict[str, Any]],
    min_value_length: int = 3,
) -> Tuple[str, List[str]]:
    """
    Scan ``text`` for leaked variable values and restore them to ``{{key}}`` form.

    This is the deterministic inverse of :func:`interpolate_variables`. It acts
    as a post-processing safety net after variation generation: when the LLM
    hardcodes a concrete variable value (e.g. ``user-123``) instead of writing
    the placeholder (``{{user_id}}``), this function replaces the value back so
    subsequent iterations receive correctly templated instructions.

    Values are matched with boundary guards so that a value like ``user-123``
    inside a longer token like ``user-1234`` is not substituted. Multi-line
    values are handled identically to single-line ones — ``re.escape`` produces
    a literal pattern and the lookbehind/lookahead only inspect the character
    immediately adjacent to the match boundary.

    Values shorter than ``min_value_length`` characters are skipped because
    short strings (e.g. ``"en"``, ``"US"``) are too likely to appear
    coincidentally in unrelated prose.

    :param text: The generated instruction string to clean.
    :param variable_choices: All possible variable dicts, used to build the
        reverse value→key map. When the same value appears under multiple keys
        the first key encountered wins.
    :param min_value_length: Minimum character length a value must have before
        it is considered for replacement. Defaults to 3.
    :return: A tuple of ``(cleaned_text, warnings)`` where ``warnings`` is a
        list of human-readable strings describing each replacement made.
    """
    # Build reverse map: string(value) → key. Longest values first so that
    # a longer value like "user-123-admin" is replaced before the shorter
    # "user-123" substring, preventing partial-match corruption.
    value_to_key: Dict[str, str] = {}
    for choice in variable_choices:
        for key, value in choice.items():
            str_value = str(value)
            if str_value not in value_to_key:
                value_to_key[str_value] = key

    sorted_entries = sorted(value_to_key.items(), key=lambda kv: len(kv[0]), reverse=True)

    warnings: List[str] = []
    for value, key in sorted_entries:
        if len(value) < min_value_length:
            continue
        placeholder = f"{{{{{key}}}}}"
        # Skip if the placeholder is already present — nothing to fix.
        if placeholder in text and value not in text:
            continue

        total_count = 0

        # Pass 1: replace {{value}} forms — the LLM used the runtime value as
        # if it were a placeholder key (e.g. {{user-125}} instead of {{user_id}}).
        # This must run before the boundary-guarded pass so that the bare value
        # inside the braces is consumed here rather than matched by pass 2,
        # which would otherwise leave the surrounding braces and produce
        # {{{{user_id}}}}.
        brace_pattern = r'\{\{' + re.escape(value) + r'\}\}'
        new_text, brace_count = re.subn(brace_pattern, placeholder, text, flags=re.DOTALL)
        if brace_count:
            text = new_text
            total_count += brace_count

        # Pass 2: replace bare value occurrences with a boundary guard so that
        # "user-123" inside "user-1234" is not substituted.
        pattern = r'(?<![A-Za-z0-9_\-])' + re.escape(value) + r'(?![A-Za-z0-9_\-])'
        new_text, count = re.subn(pattern, placeholder, text, flags=re.DOTALL)
        if count:
            text = new_text
            total_count += count

        if total_count:
            warnings.append(
                f"Variable value {value!r} found in generated instructions "
                f"— replaced {total_count} occurrence(s) with placeholder {placeholder}"
            )

    return text, warnings


_T = TypeVar("_T")


async def await_if_needed(result: Union[_T, Awaitable[_T]]) -> _T:
    """
    Handle both sync and async callable results.

    :param result: Either a value or an awaitable that returns a value
    :return: The resolved value
    """
    if inspect.isawaitable(result):
        return await result  # type: ignore[return-value]
    return result  # type: ignore[return-value]


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


def validate_variation_response(response_data: Dict[str, Any]) -> List[str]:
    """Validate the shape of a parsed LLM variation response.

    Checks that the three required fields are present and have the expected
    types. An empty ``current_parameters`` dict is acceptable; an empty
    ``current_instructions`` or ``model`` string is flagged as an error
    because downstream code cannot meaningfully use a blank value.

    :param response_data: Parsed dict from the LLM (output of extract_json_from_response).
    :return: List of human-readable error strings. Empty list means the response is valid.
    """
    errors: List[str] = []

    if "current_instructions" not in response_data:
        errors.append("missing required field 'current_instructions'")
    elif not isinstance(response_data["current_instructions"], str):
        errors.append(
            f"'current_instructions' must be a string, "
            f"got {type(response_data['current_instructions']).__name__}"
        )
    elif not response_data["current_instructions"].strip():
        errors.append("'current_instructions' must not be empty")

    if "current_parameters" not in response_data:
        errors.append("missing required field 'current_parameters'")
    elif not isinstance(response_data["current_parameters"], dict):
        errors.append(
            f"'current_parameters' must be a dict, "
            f"got {type(response_data['current_parameters']).__name__}"
        )

    if "model" not in response_data:
        errors.append("missing required field 'model'")
    elif not isinstance(response_data["model"], str):
        errors.append(
            f"'model' must be a string, got {type(response_data['model']).__name__}"
        )

    return errors


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
        start_idx = response_str.find('{')
        if start_idx != -1:
            logger.warning(
                "Direct JSON parse and code-block extraction failed; "
                "falling back to balanced-brace scanner. "
                "Response may be malformed JSON (length: %d).",
                len(response_str),
            )
        while start_idx != -1 and response_data is None:
            brace_count = 0
            i = start_idx
            while i < len(response_str):
                if response_str[i] == '{':
                    brace_count += 1
                elif response_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_str[start_idx:i + 1]
                        try:
                            response_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            start_idx = response_str.find('{', start_idx + 1)
                        break
                i += 1
            else:
                # Exhausted the string without closing the object
                break

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
            "Response length: %d",
            len(response_str),
        )
        raise ValueError(
            "Failed to parse structured output from variation generation. "
            "Expected JSON object with 'current_instructions', 'current_parameters', and 'model' fields. "
            f"Response length: {len(response_str)}"
        )

    return response_data
