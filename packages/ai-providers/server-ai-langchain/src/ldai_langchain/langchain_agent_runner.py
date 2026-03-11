"""LangChain agent runner for LaunchDarkly AI SDK."""

from typing import Any, Dict, List

from ldai import log
from ldai.providers import AgentResult, AgentRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ldai_langchain.langchain_helper import get_ai_metrics_from_response


class LangChainAgentRunner(AgentRunner):
    """
    AgentRunner implementation for LangChain.

    Executes a single-agent loop using a LangChain BaseChatModel with tool calling.
    Returned by LangChainRunnerFactory.create_agent(config, tools).
    """

    def __init__(
        self,
        llm: Any,
        instructions: str,
        tool_definitions: List[Dict[str, Any]],
        tools: ToolRegistry,
    ):
        self._llm = llm
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
        messages: List[BaseMessage] = []
        if self._instructions:
            messages.append(SystemMessage(content=self._instructions))
        messages.append(HumanMessage(content=str(input)))

        openai_tools = self._build_openai_tools()
        model = self._llm.bind_tools(openai_tools) if openai_tools else self._llm

        raw_response = None

        try:
            while True:
                response: AIMessage = await model.ainvoke(messages)
                raw_response = response
                messages.append(response)

                tool_calls = getattr(response, 'tool_calls', None)

                if not tool_calls:
                    metrics = get_ai_metrics_from_response(response)
                    content = response.content if isinstance(response.content, str) else ""
                    return AgentResult(
                        output=content,
                        raw=raw_response,
                        metrics=metrics,
                    )

                # Execute tool calls and append results
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    tool_fn = self._tools.get(tool_name)
                    if tool_fn:
                        try:
                            result = tool_fn(**tool_args)
                            if hasattr(result, "__await__"):
                                result = await result
                            result_str = str(result)
                        except Exception as error:
                            log.warning(f"Tool '{tool_name}' execution failed: {error}")
                            result_str = f"Tool execution failed: {error}"
                    else:
                        log.warning(f"Tool '{tool_name}' not found in registry")
                        result_str = f"Tool '{tool_name}' not found"

                    messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))

        except Exception as error:
            log.warning(f"LangChain agent run failed: {error}")
            return AgentResult(
                output="",
                raw=raw_response,
                metrics=LDAIMetrics(success=False, usage=None),
            )

    def _build_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert LD tool definitions to OpenAI function-calling format for bind_tools."""
        tools = []
        for td in self._tool_definitions:
            if not isinstance(td, dict):
                continue
            if "type" in td:
                tools.append(td)
            elif "name" in td:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": td["name"],
                        "description": td.get("description", ""),
                        "parameters": td.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
        return tools

    def get_llm(self) -> Any:
        """Return the underlying LangChain LLM."""
        return self._llm
