"""LangChain agent runner for LaunchDarkly AI SDK."""

from typing import Any, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from ldai import log
from ldai.providers import AgentResult, AgentRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics

from ldai_langchain.langchain_helper import get_ai_metrics_from_response


class LangChainAgentRunner(AgentRunner):
    """
    AgentRunner implementation for LangChain.

    Executes a single-agent loop using a LangChain BaseChatModel with tool calling.
    The model is expected to have tools already bound to it.
    Returned by LangChainRunnerFactory.create_agent(config, tools).
    """

    def __init__(
        self,
        llm: Any,
        instructions: str,
        tools: ToolRegistry,
    ):
        self._llm = llm
        self._instructions = instructions
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

        raw_response = None

        try:
            while True:
                response: AIMessage = await self._llm.ainvoke(messages)
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

    def get_llm(self) -> Any:
        """Return the underlying LangChain LLM."""
        return self._llm
