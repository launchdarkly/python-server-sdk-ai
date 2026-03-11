"""OpenAI agent runner for LaunchDarkly AI SDK."""

import json
from typing import Any, Dict, List

from ldai import log
from ldai.providers import AgentResult, AgentRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage
from openai import AsyncOpenAI

from ldai_openai.openai_helper import get_ai_metrics_from_response


class OpenAIAgentRunner(AgentRunner):
    """
    AgentRunner implementation for OpenAI.

    Executes a single-agent loop using OpenAI Chat Completions with tool calling.
    Returned by OpenAIRunnerFactory.create_agent(config, tools).
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        parameters: Dict[str, Any],
        instructions: str,
        tool_definitions: List[Dict[str, Any]],
        tools: ToolRegistry,
    ):
        self._client = client
        self._model_name = model_name
        self._parameters = parameters
        self._instructions = instructions
        self._tool_definitions = tool_definitions
        self._tools = tools

    async def run(self, input: Any) -> AgentResult:
        """
        Run the agent with the given input string.

        Executes an agentic loop: calls the model, handles tool calls,
        and continues until the model produces a final response.

        :param input: The user prompt or input to the agent
        :return: AgentResult with output, raw response, and aggregated metrics
        """
        messages: List[Dict[str, Any]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        messages.append({"role": "user", "content": str(input)})

        total_input = 0
        total_output = 0
        raw_response = None

        try:
            while True:
                create_kwargs: Dict[str, Any] = {
                    "model": self._model_name,
                    "messages": messages,
                    **self._parameters,
                }
                openai_tools = self._build_openai_tools()
                if openai_tools:
                    create_kwargs["tools"] = openai_tools
                    create_kwargs["tool_choice"] = "auto"

                response = await self._client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
                raw_response = response
                metrics = get_ai_metrics_from_response(response)

                if metrics.usage:
                    total_input += metrics.usage.input
                    total_output += metrics.usage.output

                if not response.choices:
                    break

                message = response.choices[0].message

                # Add assistant message to history
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                messages.append(assistant_msg)

                if not message.tool_calls:
                    total_tokens = total_input + total_output
                    return AgentResult(
                        output=message.content or "",
                        raw=raw_response,
                        metrics=LDAIMetrics(
                            success=True,
                            usage=TokenUsage(
                                total=total_tokens,
                                input=total_input,
                                output=total_output,
                            ) if total_tokens > 0 else None,
                        ),
                    )

                # Execute tool calls and append results
                for tool_call in message.tool_calls:
                    result = await self._call_tool(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

        except Exception as error:
            log.warning(f"OpenAI agent run failed: {error}")
            return AgentResult(
                output="",
                raw=raw_response,
                metrics=LDAIMetrics(success=False, usage=None),
            )

        return AgentResult(
            output="",
            raw=raw_response,
            metrics=LDAIMetrics(success=False, usage=None),
        )

    async def _call_tool(self, name: str, arguments_json: str) -> str:
        """Execute a tool by name, returning the result as a string."""
        tool_fn = self._tools.get(name)
        if not tool_fn:
            log.warning(f"Tool '{name}' not found in registry")
            return f"Tool '{name}' not found"
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            result = tool_fn(**args)
            if hasattr(result, "__await__"):
                result = await result
            return str(result)
        except Exception as error:
            log.warning(f"Tool '{name}' execution failed: {error}")
            return f"Tool execution failed: {error}"

    def _build_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert LD tool definitions to OpenAI function-calling format."""
        tools = []
        for td in self._tool_definitions:
            if not isinstance(td, dict):
                continue
            if "type" in td:
                # Already in OpenAI format
                tools.append(td)
            elif "name" in td:
                # LD simplified format: {name, description, parameters}
                tools.append({
                    "type": "function",
                    "function": {
                        "name": td["name"],
                        "description": td.get("description", ""),
                        "parameters": td.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
        return tools

    def get_client(self) -> AsyncOpenAI:
        """Return the underlying AsyncOpenAI client."""
        return self._client
