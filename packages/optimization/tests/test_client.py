"""Tests for OptimizationClient."""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ldai import AIAgentConfig, AIJudgeConfig, AIJudgeConfigDefault, LDAIClient
from ldai.models import LDMessage, ModelConfig
from ldclient import Context

from ldai_optimization.client import OptimizationClient
from ldai_optimization.dataclasses import (
    AIJudgeCallConfig,
    JudgeResult,
    OptimizationContext,
    OptimizationFromConfigOptions,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
    ToolDefinition,
)
from ldai_optimization.prompts import (
    build_new_variation_prompt,
    variation_prompt_acceptance_criteria,
)
from ldai_optimization.util import (
    create_evaluation_tool,
    create_variation_tool,
    handle_evaluation_tool_call,
    handle_variation_tool_call,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

LD_CONTEXT = Context.create("test-user")

AGENT_INSTRUCTIONS = "You are a helpful assistant. Answer using {{language}}."
VARIATION_RESPONSE = json.dumps({
    "current_instructions": "You are an improved assistant.",
    "current_parameters": {"temperature": 0.5},
    "model": "gpt-4o",
})
JUDGE_PASS_RESPONSE = json.dumps({"score": 1.0, "rationale": "Perfect answer."})
JUDGE_FAIL_RESPONSE = json.dumps({"score": 0.2, "rationale": "Off topic."})


def _make_agent_config(
    instructions: str = AGENT_INSTRUCTIONS,
    model_name: str = "gpt-4o",
    parameters: Dict[str, Any] | None = None,
) -> AIAgentConfig:
    return AIAgentConfig(
        key="test-agent",
        enabled=True,
        model=ModelConfig(name=model_name, parameters=parameters or {}),
        instructions=instructions,
    )


def _make_ldai_client(agent_config: AIAgentConfig | None = None) -> MagicMock:
    mock = MagicMock(spec=LDAIClient)
    mock.agent_config.return_value = agent_config or _make_agent_config()
    mock._client = MagicMock()
    mock._client.variation.return_value = {"instructions": AGENT_INSTRUCTIONS}
    return mock


def _make_options(
    *,
    handle_agent_call=None,
    handle_judge_call=None,
    judges=None,
    max_attempts: int = 3,
    variable_choices=None,
) -> OptimizationOptions:
    if handle_agent_call is None:
        handle_agent_call = AsyncMock(return_value="The capital of France is Paris.")
    if handle_judge_call is None:
        handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
    if judges is None:
        judges = {
            "accuracy": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be accurate and concise.",
            )
        }
    return OptimizationOptions(
        context_choices=[LD_CONTEXT],
        max_attempts=max_attempts,
        model_choices=["gpt-4o", "gpt-4o-mini"],
        judge_model="gpt-4o",
        variable_choices=variable_choices or [{"language": "English"}],
        handle_agent_call=handle_agent_call,
        handle_judge_call=handle_judge_call,
        judges=judges,
    )


def _make_client(ldai: MagicMock | None = None) -> OptimizationClient:
    client = OptimizationClient(ldai or _make_ldai_client())
    return client


# ---------------------------------------------------------------------------
# Util functions
# ---------------------------------------------------------------------------


class TestHandleEvaluationToolCall:
    def test_returns_json_with_score_and_rationale(self):
        result = handle_evaluation_tool_call(score=0.8, rationale="Good answer.")
        data = json.loads(result)
        assert data["score"] == 0.8
        assert data["rationale"] == "Good answer."

    def test_score_zero_is_valid(self):
        result = handle_evaluation_tool_call(score=0.0, rationale="No match.")
        assert json.loads(result)["score"] == 0.0

    def test_result_is_valid_json_string(self):
        result = handle_evaluation_tool_call(score=0.5, rationale="Partial.")
        assert isinstance(result, str)
        json.loads(result)  # must not raise


class TestHandleVariationToolCall:
    def test_returns_json_with_all_fields(self):
        result = handle_variation_tool_call(
            current_instructions="Do X.",
            current_parameters={"temperature": 0.7},
            model="gpt-4o",
        )
        data = json.loads(result)
        assert data["current_instructions"] == "Do X."
        assert data["current_parameters"] == {"temperature": 0.7}
        assert data["model"] == "gpt-4o"

    def test_result_is_valid_json_string(self):
        result = handle_variation_tool_call(
            current_instructions="Do Y.",
            current_parameters={},
            model="gpt-4o-mini",
        )
        assert isinstance(result, str)
        json.loads(result)


# ---------------------------------------------------------------------------
# _extract_agent_tools
# ---------------------------------------------------------------------------


class TestExtractAgentTools:
    def setup_method(self):
        self.client = _make_client()
        self.client._agent_key = "test-agent"
        self.client._options = _make_options()
        self.client._agent_config = _make_agent_config()
        self.client._initialize_class_members_from_config(_make_agent_config())

    def test_returns_empty_list_when_no_tools(self):
        result = self.client._extract_agent_tools({})
        assert result == []

    def test_returns_empty_list_when_tools_key_is_empty(self):
        result = self.client._extract_agent_tools({"tools": []})
        assert result == []

    def test_returns_structured_output_tool_from_dict(self):
        tool_dict = {
            "name": "lookup",
            "description": "Looks up data",
            "input_schema": {"type": "object", "properties": {}},
        }
        result = self.client._extract_agent_tools({"tools": [tool_dict]})
        assert len(result) == 1
        assert isinstance(result[0], ToolDefinition)
        assert result[0].name == "lookup"

    def test_passes_through_existing_structured_output_tool(self):
        tool = ToolDefinition(
            name="my-tool", description="desc", input_schema={}
        )
        result = self.client._extract_agent_tools({"tools": [tool]})
        assert result == [tool]

    def test_wraps_single_non_list_tool(self):
        tool_dict = {"name": "single", "description": "x", "input_schema": {}}
        result = self.client._extract_agent_tools({"tools": tool_dict})
        assert len(result) == 1
        assert result[0].name == "single"

    def test_converts_object_with_to_dict(self):
        mock_tool = MagicMock()
        mock_tool.to_dict.return_value = {
            "name": "converted",
            "description": "via to_dict",
            "input_schema": {},
        }
        result = self.client._extract_agent_tools({"tools": [mock_tool]})
        assert len(result) == 1
        assert result[0].name == "converted"


# ---------------------------------------------------------------------------
# _evaluate_response
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    def setup_method(self):
        self.client = _make_client()
        self.client._options = _make_options()

    def _ctx_with_scores(self, scores: Dict[str, JudgeResult]) -> OptimizationContext:
        return OptimizationContext(
            scores=scores,
            completion_response="Some response.",
            current_instructions="Do X.",
            current_parameters={},
            current_variables={},
            iteration=1,
        )

    def test_passes_when_all_judges_meet_threshold(self):
        ctx = self._ctx_with_scores({"accuracy": JudgeResult(score=0.9)})
        assert self.client._evaluate_response(ctx) is True

    def test_fails_when_judge_below_threshold(self):
        ctx = self._ctx_with_scores({"accuracy": JudgeResult(score=0.5)})
        assert self.client._evaluate_response(ctx) is False

    def test_fails_when_judge_result_missing(self):
        ctx = self._ctx_with_scores({})
        assert self.client._evaluate_response(ctx) is False

    def test_passes_at_exact_threshold(self):
        ctx = self._ctx_with_scores({"accuracy": JudgeResult(score=0.8)})
        assert self.client._evaluate_response(ctx) is True

    def test_no_judges_always_passes(self):
        options = _make_options(judges=None, handle_agent_call=AsyncMock(return_value="x"))
        # Need on_turn to satisfy validation — inject directly
        options_with_on_turn = OptimizationOptions(
            context_choices=[LD_CONTEXT],
            max_attempts=1,
            model_choices=["gpt-4o"],
            judge_model="gpt-4o",
            variable_choices=[{}],
            handle_agent_call=AsyncMock(return_value="x"),
            handle_judge_call=AsyncMock(return_value=JUDGE_PASS_RESPONSE),
            judges={"j": OptimizationJudge(threshold=1.0, acceptance_statement="x")},
            on_turn=lambda ctx: True,
        )
        self.client._options = options_with_on_turn
        # Without judges, _evaluate_response returns True
        options_no_judges = MagicMock()
        options_no_judges.judges = None
        self.client._options = options_no_judges
        ctx = self._ctx_with_scores({})
        assert self.client._evaluate_response(ctx) is True

    def test_multiple_judges_all_must_pass(self):
        self.client._options = _make_options(
            judges={
                "a": OptimizationJudge(threshold=0.8, acceptance_statement="A"),
                "b": OptimizationJudge(threshold=0.9, acceptance_statement="B"),
            }
        )
        ctx = self._ctx_with_scores({
            "a": JudgeResult(score=0.9),
            "b": JudgeResult(score=0.7),  # fails
        })
        assert self.client._evaluate_response(ctx) is False

    def test_multiple_judges_all_passing(self):
        self.client._options = _make_options(
            judges={
                "a": OptimizationJudge(threshold=0.8, acceptance_statement="A"),
                "b": OptimizationJudge(threshold=0.8, acceptance_statement="B"),
            }
        )
        ctx = self._ctx_with_scores({
            "a": JudgeResult(score=0.9),
            "b": JudgeResult(score=1.0),
        })
        assert self.client._evaluate_response(ctx) is True


# ---------------------------------------------------------------------------
# _builtin_judge_tool_handlers / _builtin_agent_tool_handlers
# ---------------------------------------------------------------------------


class TestBuiltinToolHandlers:
    def setup_method(self):
        self.client = _make_client()
        self.client._options = _make_options()

    def test_judge_handlers_contains_evaluation_tool(self):
        handlers = self.client._builtin_judge_tool_handlers()
        assert create_evaluation_tool().name in handlers

    def test_judge_handler_returns_json(self):
        handlers = self.client._builtin_judge_tool_handlers()
        result = handlers[create_evaluation_tool().name](score=0.7, rationale="ok")
        data = json.loads(result)
        assert data["score"] == 0.7

    def test_agent_handlers_empty_for_regular_turn(self):
        handlers = self.client._builtin_agent_tool_handlers(is_variation=False)
        assert handlers == {}

    def test_agent_handlers_contains_variation_tool_for_variation_turn(self):
        handlers = self.client._builtin_agent_tool_handlers(is_variation=True)
        expected_name = create_variation_tool(self.client._options.model_choices).name
        assert expected_name in handlers

    def test_variation_handler_returns_valid_json(self):
        handlers = self.client._builtin_agent_tool_handlers(is_variation=True)
        name = create_variation_tool(self.client._options.model_choices).name
        result = handlers[name](
            current_instructions="New instructions.",
            current_parameters={"temperature": 0.3},
            model="gpt-4o",
        )
        data = json.loads(result)
        assert data["current_instructions"] == "New instructions."
        assert data["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# _evaluate_acceptance_judge
# ---------------------------------------------------------------------------


class TestEvaluateAcceptanceJudge:
    def setup_method(self):
        self.client = _make_client()
        agent_config = _make_agent_config()
        self.client._agent_key = "test-agent"
        self.client._agent_config = agent_config
        self.client._initialize_class_members_from_config(agent_config)
        self.handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        self.client._options = _make_options(handle_judge_call=self.handle_judge_call)

    async def test_returns_parsed_score_and_rationale(self):
        judge = OptimizationJudge(
            threshold=0.8, acceptance_statement="Must be concise."
        )
        result = await self.client._evaluate_acceptance_judge(
            judge_key="conciseness",
            optimization_judge=judge,
            completion_response="Paris.",
            iteration=1,
            reasoning_history="",
            user_input="What is the capital of France?",
        )
        assert result.score == 1.0
        assert result.rationale == "Perfect answer."

    async def test_handle_judge_call_receives_correct_key_and_config(self):
        judge = OptimizationJudge(
            threshold=0.8, acceptance_statement="Must answer the question."
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="relevance",
            optimization_judge=judge,
            completion_response="Some answer.",
            iteration=1,
            reasoning_history="",
            user_input="What time is it?",
        )
        call_args = self.handle_judge_call.call_args
        key, config, ctx, handlers = call_args.args
        assert key == "relevance"
        assert isinstance(config, AIJudgeCallConfig)
        assert isinstance(ctx, OptimizationJudgeContext)
        assert create_evaluation_tool().name in handlers

    async def test_messages_has_system_and_user_turns(self):
        judge = OptimizationJudge(
            threshold=0.8, acceptance_statement="Must be factual."
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="facts",
            optimization_judge=judge,
            completion_response="The sky is blue.",
            iteration=1,
            reasoning_history="",
            user_input="What colour is the sky?",
        )
        _, config, _, _ = self.handle_judge_call.call_args.args
        roles = [m.role for m in config.messages]
        assert roles == ["system", "user"]

    async def test_messages_system_content_matches_instructions(self):
        judge = OptimizationJudge(
            threshold=0.8, acceptance_statement="Be concise."
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="brevity",
            optimization_judge=judge,
            completion_response="Yes.",
            iteration=1,
            reasoning_history="",
            user_input="Is Paris in France?",
        )
        _, config, _, _ = self.handle_judge_call.call_args.args
        system_msg = next(m for m in config.messages if m.role == "system")
        assert system_msg.content == config.instructions

    async def test_messages_user_content_matches_context_user_input(self):
        judge = OptimizationJudge(
            threshold=0.8, acceptance_statement="Answer directly."
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="directness",
            optimization_judge=judge,
            completion_response="Paris.",
            iteration=1,
            reasoning_history="",
            user_input="Capital of France?",
        )
        _, config, ctx, _ = self.handle_judge_call.call_args.args
        user_msg = next(m for m in config.messages if m.role == "user")
        assert user_msg.content == ctx.user_input

    async def test_acceptance_statement_in_instructions(self):
        statement = "Response must mention the Eiffel Tower."
        judge = OptimizationJudge(threshold=0.8, acceptance_statement=statement)
        await self.client._evaluate_acceptance_judge(
            judge_key="tower",
            optimization_judge=judge,
            completion_response="Paris has the Eiffel Tower.",
            iteration=1,
            reasoning_history="",
            user_input="Tell me about Paris.",
        )
        call_args = self.handle_judge_call.call_args
        _, config, _, _ = call_args.args
        assert statement in config.instructions

    async def test_evaluation_tool_in_config_parameters(self):
        judge = OptimizationJudge(threshold=0.8, acceptance_statement="Be brief.")
        await self.client._evaluate_acceptance_judge(
            judge_key="brevity",
            optimization_judge=judge,
            completion_response="Yes.",
            iteration=1,
            reasoning_history="",
            user_input="Is Paris in France?",
        )
        call_args = self.handle_judge_call.call_args
        _, config, _, _ = call_args.args
        tools = config.model.get_parameter("tools") or []
        tool_names = [t["name"] for t in tools]
        assert create_evaluation_tool().name in tool_names

    async def test_agent_tools_prepended_to_config_tools(self):
        agent_tool = ToolDefinition(
            name="lookup", description="Lookup data", input_schema={}
        )
        judge = OptimizationJudge(threshold=0.8, acceptance_statement="Use tool.")
        await self.client._evaluate_acceptance_judge(
            judge_key="tool-use",
            optimization_judge=judge,
            completion_response="I looked it up.",
            iteration=1,
            reasoning_history="",
            user_input="Find me something.",
            agent_tools=[agent_tool],
        )
        call_args = self.handle_judge_call.call_args
        _, config, _, _ = call_args.args
        tools = config.model.get_parameter("tools") or []
        tool_names = [t["name"] for t in tools]
        assert "lookup" in tool_names
        assert tool_names.index("lookup") < tool_names.index(create_evaluation_tool().name)

    async def test_variables_in_context(self):
        judge = OptimizationJudge(threshold=0.8, acceptance_statement="Be accurate.")
        variables = {"language": "French", "topic": "geography"}
        await self.client._evaluate_acceptance_judge(
            judge_key="accuracy",
            optimization_judge=judge,
            completion_response="Paris.",
            iteration=1,
            reasoning_history="",
            user_input="Capital?",
            variables=variables,
        )
        call_args = self.handle_judge_call.call_args
        _, _, ctx, _ = call_args.args
        assert ctx.variables == variables

    async def test_returns_zero_score_on_missing_acceptance_statement(self):
        judge = OptimizationJudge(threshold=0.8, acceptance_statement=None)
        result = await self.client._evaluate_acceptance_judge(
            judge_key="broken",
            optimization_judge=judge,
            completion_response="Anything.",
            iteration=1,
            reasoning_history="",
            user_input="Hello?",
        )
        assert result.score == 0.0
        self.handle_judge_call.assert_not_called()

    async def test_returns_zero_score_on_parse_failure(self):
        self.handle_judge_call.return_value = "not json at all"
        judge = OptimizationJudge(threshold=0.8, acceptance_statement="Be clear.")
        result = await self.client._evaluate_acceptance_judge(
            judge_key="clarity",
            optimization_judge=judge,
            completion_response="Clear answer.",
            iteration=1,
            reasoning_history="",
            user_input="Explain X.",
        )
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# _evaluate_config_judge
# ---------------------------------------------------------------------------


class TestEvaluateConfigJudge:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()
        self.client = _make_client(self.mock_ldai)
        agent_config = _make_agent_config()
        self.client._agent_key = "test-agent"
        self.client._agent_config = agent_config
        self.client._initialize_class_members_from_config(agent_config)
        self.handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        self.client._options = _make_options(handle_judge_call=self.handle_judge_call)

    def _make_judge_config(self, enabled: bool = True) -> AIJudgeConfig:
        return AIJudgeConfig(
            key="ld-judge-key",
            enabled=enabled,
            model=ModelConfig(name="gpt-4o", parameters={}),
            messages=[
                LDMessage(role="system", content="You are an evaluator."),
                LDMessage(role="user", content="Evaluate this response."),
            ],
        )

    async def test_calls_handle_judge_call_with_correct_config_type(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config()
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Good answer.",
            iteration=1,
            reasoning_history="",
            user_input="What is X?",
        )
        call_args = self.handle_judge_call.call_args
        key, config, ctx, handlers = call_args.args
        assert key == "quality"
        assert isinstance(config, AIJudgeCallConfig)
        assert "You are an evaluator." in config.instructions
        assert isinstance(ctx, OptimizationJudgeContext)

    async def test_messages_has_system_and_user_turns(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config()
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Good answer.",
            iteration=1,
            reasoning_history="",
            user_input="What is X?",
        )
        _, config, _, _ = self.handle_judge_call.call_args.args
        roles = [m.role for m in config.messages]
        assert roles == ["system", "user"]

    async def test_messages_system_content_matches_instructions(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config()
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Good answer.",
            iteration=1,
            reasoning_history="",
            user_input="What is X?",
        )
        _, config, _, _ = self.handle_judge_call.call_args.args
        system_msg = next(m for m in config.messages if m.role == "system")
        assert system_msg.content == config.instructions

    async def test_messages_user_content_matches_context_user_input(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config()
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Good answer.",
            iteration=1,
            reasoning_history="",
            user_input="What is X?",
        )
        _, config, ctx, _ = self.handle_judge_call.call_args.args
        user_msg = next(m for m in config.messages if m.role == "user")
        assert user_msg.content == ctx.user_input

    async def test_messages_user_content_contains_ld_user_message(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config()
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Good answer.",
            iteration=1,
            reasoning_history="",
            user_input="What is X?",
        )
        _, config, _, _ = self.handle_judge_call.call_args.args
        user_msg = next(m for m in config.messages if m.role == "user")
        assert "Evaluate this response." in user_msg.content

    async def test_returns_zero_score_when_judge_disabled(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config(enabled=False)
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        result = await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Some answer.",
            iteration=1,
            reasoning_history="",
            user_input="What?",
        )
        assert result.score == 0.0
        self.handle_judge_call.assert_not_called()

    async def test_returns_zero_score_when_judge_has_no_messages(self):
        judge_config = AIJudgeConfig(
            key="ld-judge-key",
            enabled=True,
            model=ModelConfig(name="gpt-4o", parameters={}),
            messages=None,
        )
        self.mock_ldai.judge_config.return_value = judge_config
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        result = await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Any.",
            iteration=1,
            reasoning_history="",
            user_input="Anything?",
        )
        assert result.score == 0.0
        self.handle_judge_call.assert_not_called()

    async def test_template_variables_merged_into_judge_config_call(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config()
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        variables = {"language": "Spanish"}
        await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Answer.",
            iteration=1,
            reasoning_history="",
            user_input="Q?",
            variables=variables,
        )
        call_kwargs = self.mock_ldai.judge_config.call_args
        passed_vars = call_kwargs.args[3] if call_kwargs.args else call_kwargs.kwargs.get("variables", {})
        assert passed_vars.get("language") == "Spanish"
        assert "message_history" in passed_vars
        assert "response_to_evaluate" in passed_vars

    async def test_agent_tools_prepended_before_evaluation_tool(self):
        self.mock_ldai.judge_config.return_value = self._make_judge_config()
        agent_tool = ToolDefinition(name="search", description="Search", input_schema={})
        judge = OptimizationJudge(threshold=0.8, judge_key="ld-judge-key")
        await self.client._evaluate_config_judge(
            judge_key="quality",
            optimization_judge=judge,
            completion_response="Answer.",
            iteration=1,
            reasoning_history="",
            user_input="Q?",
            agent_tools=[agent_tool],
        )
        _, config, _, _ = self.handle_judge_call.call_args.args
        tools = config.model.get_parameter("tools") or []
        names = [t["name"] for t in tools]
        assert "search" in names
        assert names.index("search") < names.index(create_evaluation_tool().name)


# ---------------------------------------------------------------------------
# _execute_agent_turn
# ---------------------------------------------------------------------------


class TestExecuteAgentTurn:
    def setup_method(self):
        self.agent_response = "Paris is the capital of France."
        self.handle_agent_call = AsyncMock(return_value=self.agent_response)
        self.handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        self.client = _make_client()
        agent_config = _make_agent_config()
        self.client._agent_key = "test-agent"
        self.client._agent_config = agent_config
        self.client._initialize_class_members_from_config(agent_config)
        self.client._options = _make_options(
            handle_agent_call=self.handle_agent_call,
            handle_judge_call=self.handle_judge_call,
        )

    def _make_context(self, user_input: str = "What is the capital of France?") -> OptimizationContext:
        return OptimizationContext(
            scores={},
            completion_response="",
            current_instructions=AGENT_INSTRUCTIONS,
            current_parameters={},
            current_variables={"language": "English"},
            current_model="gpt-4o",
            user_input=user_input,
            iteration=1,
        )

    async def test_calls_handle_agent_call_with_config_and_context(self):
        ctx = self._make_context()
        await self.client._execute_agent_turn(ctx, iteration=1)
        self.handle_agent_call.assert_called_once()
        key, config, passed_ctx, handlers = self.handle_agent_call.call_args.args
        assert key == "test-agent"
        assert isinstance(config, AIAgentConfig)
        assert passed_ctx is ctx
        assert handlers == {}

    async def test_completion_response_stored_in_returned_context(self):
        ctx = self._make_context()
        result = await self.client._execute_agent_turn(ctx, iteration=1)
        assert result.completion_response == self.agent_response

    async def test_judge_scores_stored_in_returned_context(self):
        ctx = self._make_context()
        result = await self.client._execute_agent_turn(ctx, iteration=1)
        assert "accuracy" in result.scores
        assert result.scores["accuracy"].score == 1.0

    async def test_variables_interpolated_into_agent_config_instructions(self):
        ctx = self._make_context()
        await self.client._execute_agent_turn(ctx, iteration=1)
        _, config, _, _ = self.handle_agent_call.call_args.args
        assert "{{language}}" not in config.instructions
        assert "English" in config.instructions

    async def test_raises_on_agent_call_failure(self):
        self.handle_agent_call.side_effect = RuntimeError("LLM unavailable")
        ctx = self._make_context()
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await self.client._execute_agent_turn(ctx, iteration=1)


# ---------------------------------------------------------------------------
# _generate_new_variation
# ---------------------------------------------------------------------------


class TestGenerateNewVariation:
    def setup_method(self):
        self.handle_agent_call = AsyncMock(return_value=VARIATION_RESPONSE)
        self.client = _make_client()
        agent_config = _make_agent_config()
        self.client._agent_key = "test-agent"
        self.client._agent_config = agent_config
        self.client._initial_instructions = AGENT_INSTRUCTIONS
        self.client._initialize_class_members_from_config(agent_config)
        self.client._options = _make_options(handle_agent_call=self.handle_agent_call)

    async def test_updates_current_instructions(self):
        await self.client._generate_new_variation(iteration=1, variables={"language": "English"})
        assert self.client._current_instructions == "You are an improved assistant."

    async def test_updates_current_parameters(self):
        await self.client._generate_new_variation(iteration=1, variables={})
        assert self.client._current_parameters == {"temperature": 0.5}

    async def test_updates_current_model(self):
        await self.client._generate_new_variation(iteration=1, variables={})
        assert self.client._current_model == "gpt-4o"

    async def test_variation_tool_in_agent_config(self):
        await self.client._generate_new_variation(iteration=1, variables={})
        _, config, _, _ = self.handle_agent_call.call_args.args
        tools = config.model.get_parameter("tools") or []
        tool_names = [t["name"] for t in tools]
        assert create_variation_tool(self.client._options.model_choices).name in tool_names

    async def test_builtin_handlers_passed_for_variation(self):
        await self.client._generate_new_variation(iteration=1, variables={})
        _, _, _, handlers = self.handle_agent_call.call_args.args
        expected_name = create_variation_tool(self.client._options.model_choices).name
        assert expected_name in handlers

    async def test_model_not_updated_when_not_in_model_choices(self):
        bad_response = json.dumps({
            "current_instructions": "New instructions.",
            "current_parameters": {},
            "model": "some-unknown-model",
        })
        self.handle_agent_call.return_value = bad_response
        original_model = self.client._current_model
        await self.client._generate_new_variation(iteration=1, variables={})
        assert self.client._current_model == original_model


# ---------------------------------------------------------------------------
# Full optimization loop
# ---------------------------------------------------------------------------


class TestRunOptimization:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()

    async def test_succeeds_on_first_attempt_when_judge_passes(self):
        handle_agent_call = AsyncMock(return_value="The capital of France is Paris.")
        handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
        )
        result = await client.optimize_from_options("test-agent", options)
        assert result.scores["accuracy"].score == 1.0
        handle_agent_call.assert_called_once()

    async def test_generates_variation_when_judge_fails(self):
        agent_responses = [
            "Bad answer.",
            VARIATION_RESPONSE,  # variation generation
            "Better answer.",
        ]
        handle_agent_call = AsyncMock(side_effect=agent_responses)
        judge_responses = [JUDGE_FAIL_RESPONSE, JUDGE_PASS_RESPONSE]
        handle_judge_call = AsyncMock(side_effect=judge_responses)
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
            max_attempts=3,
        )
        result = await client.optimize_from_options("test-agent", options)
        assert result.scores["accuracy"].score == 1.0
        assert handle_agent_call.call_count == 3  # 1 agent + 1 variation + 1 agent

    async def test_returns_last_context_after_max_attempts(self):
        # The max_attempts guard fires before variation on the final iteration,
        # so only iterations 1 and 2 produce a variation call.
        handle_agent_call = AsyncMock(side_effect=[
            "Bad answer.",       # iteration 1: agent
            VARIATION_RESPONSE,  # iteration 1: variation
            "Still bad.",        # iteration 2: agent
            VARIATION_RESPONSE,  # iteration 2: variation
            "Still bad.",        # iteration 3: agent (max_attempts reached — no variation)
        ])
        handle_judge_call = AsyncMock(return_value=JUDGE_FAIL_RESPONSE)
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
            max_attempts=3,
        )
        result = await client.optimize_from_options("test-agent", options)
        assert result.scores["accuracy"].score == 0.2

    async def test_on_passing_result_called_on_success(self):
        on_passing = MagicMock()
        handle_agent_call = AsyncMock(return_value="Great answer.")
        handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
        )
        options.on_passing_result = on_passing
        await client.optimize_from_options("test-agent", options)
        on_passing.assert_called_once()

    async def test_on_failing_result_called_on_max_attempts(self):
        on_failing = MagicMock()
        handle_agent_call = AsyncMock(side_effect=[
            "Bad.",             # iteration 1: agent
            VARIATION_RESPONSE, # iteration 1: variation
            "Still bad.",       # iteration 2: agent (max_attempts reached — no variation)
        ])
        handle_judge_call = AsyncMock(return_value=JUDGE_FAIL_RESPONSE)
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
            max_attempts=2,
        )
        options.on_failing_result = on_failing
        await client.optimize_from_options("test-agent", options)
        on_failing.assert_called_once()

    async def test_on_turn_manual_path_success(self):
        handle_agent_call = AsyncMock(return_value="Answer.")
        handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        client = _make_client(self.mock_ldai)
        options = OptimizationOptions(
            context_choices=[LD_CONTEXT],
            max_attempts=3,
            model_choices=["gpt-4o"],
            judge_model="gpt-4o",
            variable_choices=[{}],
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
            judges={"j": OptimizationJudge(threshold=0.8, acceptance_statement="x")},
            on_turn=lambda ctx: True,
        )
        result = await client.optimize_from_options("test-agent", options)
        assert result.completion_response == "Answer."

    async def test_status_update_callback_called_at_each_stage(self):
        statuses = []
        handle_agent_call = AsyncMock(return_value="Good answer.")
        handle_judge_call = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
        )
        options.on_status_update = lambda status, ctx: statuses.append(status)
        await client.optimize_from_options("test-agent", options)
        assert "init" in statuses
        assert "generating" in statuses
        assert "evaluating" in statuses
        assert "success" in statuses


# ---------------------------------------------------------------------------
# Variation prompt — acceptance criteria section
# ---------------------------------------------------------------------------


class TestVariationPromptAcceptanceCriteria:
    def test_includes_acceptance_statement_in_section(self):
        judges = {
            "quality": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="Responses must be concise and factual.",
            )
        }
        section = variation_prompt_acceptance_criteria(judges)
        assert "Responses must be concise and factual." in section
        assert "quality" in section

    def test_labels_all_judges(self):
        judges = {
            "a": OptimizationJudge(threshold=0.8, acceptance_statement="Must be brief."),
            "b": OptimizationJudge(threshold=0.9, acceptance_statement="Must cite sources."),
        }
        section = variation_prompt_acceptance_criteria(judges)
        assert "[a]" in section
        assert "[b]" in section
        assert "Must be brief." in section
        assert "Must cite sources." in section

    def test_returns_empty_string_when_no_acceptance_statements(self):
        judges = {
            "ld-judge": OptimizationJudge(threshold=0.8, judge_key="some-ld-key"),
        }
        section = variation_prompt_acceptance_criteria(judges)
        assert section == ""

    def test_returns_empty_string_with_no_judges(self):
        section = variation_prompt_acceptance_criteria(None)
        assert section == ""

    def test_section_appears_in_full_prompt(self):
        judges = {
            "accuracy": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="Facts only.",
            )
        }
        options = _make_options(judges=judges)
        prompt = build_new_variation_prompt(
            history=[],
            judges=judges,
            current_model="gpt-4o",
            current_instructions=AGENT_INSTRUCTIONS,
            current_parameters={},
            model_choices=options.model_choices,
            variable_choices=options.variable_choices,
            initial_instructions=AGENT_INSTRUCTIONS,
        )
        assert "Facts only." in prompt
        assert "ACCEPTANCE CRITERIA" in prompt


# ---------------------------------------------------------------------------
# _build_options_from_config helpers
# ---------------------------------------------------------------------------

_API_CONFIG: Dict[str, Any] = {
    "id": "opt-uuid-123",
    "key": "my-optimization",
    "aiConfigKey": "my-agent",
    "maxAttempts": 3,
    "modelChoices": ["gpt-4o", "gpt-4o-mini"],
    "judgeModel": "gpt-4o",
    "variableChoices": [{"language": "English"}],
    "acceptanceStatements": [{"statement": "Be accurate.", "threshold": 0.9}],
    "judges": [],
    "userInputOptions": ["What is 2+2?"],
    "version": 2,
    "createdAt": 1700000000,
}


def _make_from_config_options(**overrides: Any) -> OptimizationFromConfigOptions:
    defaults: Dict[str, Any] = dict(
        project_key="my-project",
        context_choices=[LD_CONTEXT],
        handle_agent_call=AsyncMock(return_value="The answer is 4."),
        handle_judge_call=AsyncMock(return_value=JUDGE_PASS_RESPONSE),
    )
    defaults.update(overrides)
    return OptimizationFromConfigOptions(**defaults)


def _make_mock_api_client() -> MagicMock:
    mock = MagicMock()
    mock.post_agent_optimization_result = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# _build_options_from_config
# ---------------------------------------------------------------------------


class TestBuildOptionsFromConfig:
    def setup_method(self):
        self.client = _make_client()
        self.client._agent_key = "my-agent"
        self.client._initialize_class_members_from_config(_make_agent_config())
        self.client._options = _make_options()
        self.api_client = _make_mock_api_client()

    def _build(self, config=None, options=None) -> OptimizationOptions:
        return self.client._build_options_from_config(
            config or dict(_API_CONFIG),
            options or _make_from_config_options(),
            self.api_client,
            optimization_id="opt-uuid-123",
            run_id="run-uuid-456",
        )

    def test_acceptance_statements_mapped_to_judges(self):
        result = self._build()
        assert "acceptance-statement-0" in result.judges
        judge = result.judges["acceptance-statement-0"]
        assert judge.acceptance_statement == "Be accurate."
        assert judge.threshold == 0.9

    def test_multiple_acceptance_statements_get_indexed_keys(self):
        config = dict(_API_CONFIG, acceptanceStatements=[
            {"statement": "First.", "threshold": 0.8},
            {"statement": "Second.", "threshold": 0.7},
        ])
        result = self._build(config=config)
        assert "acceptance-statement-0" in result.judges
        assert "acceptance-statement-1" in result.judges
        assert result.judges["acceptance-statement-0"].acceptance_statement == "First."
        assert result.judges["acceptance-statement-1"].acceptance_statement == "Second."

    def test_judges_mapped_by_key(self):
        config = dict(_API_CONFIG, acceptanceStatements=[], judges=[
            {"key": "accuracy", "threshold": 0.85},
        ])
        result = self._build(config=config)
        assert "accuracy" in result.judges
        judge = result.judges["accuracy"]
        assert judge.judge_key == "accuracy"
        assert judge.threshold == 0.85

    def test_acceptance_statements_and_judges_merged(self):
        config = dict(_API_CONFIG,
            acceptanceStatements=[{"statement": "Be brief.", "threshold": 0.8}],
            judges=[{"key": "accuracy", "threshold": 0.9}],
        )
        result = self._build(config=config)
        assert "acceptance-statement-0" in result.judges
        assert "accuracy" in result.judges

    def test_raises_when_no_judges_no_ground_truth_no_on_turn(self):
        config = dict(_API_CONFIG, acceptanceStatements=[], judges=[])
        with pytest.raises(ValueError, match="no acceptance statements, judges, or ground truth"):
            self._build(config=config)

    def test_ground_truth_responses_alone_does_not_pass_no_criteria_check(self):
        # groundTruthResponses is not yet implemented as standalone criteria;
        # OptimizationOptions still requires judges or on_turn.
        config = dict(_API_CONFIG, acceptanceStatements=[], judges=[], groundTruthResponses=["4"])
        with pytest.raises((ValueError, Exception)):
            self._build(config=config)

    def test_on_turn_satisfies_no_judges_requirement(self):
        config = dict(_API_CONFIG, acceptanceStatements=[], judges=[])
        options = _make_from_config_options(on_turn=lambda ctx: True)
        result = self._build(config=config, options=options)
        assert result.on_turn is not None

    def test_empty_variable_choices_defaults_to_single_empty_dict(self):
        config = dict(_API_CONFIG, variableChoices=[])
        result = self._build(config=config)
        assert result.variable_choices == [{}]

    def test_non_empty_variable_choices_passed_through(self):
        result = self._build()
        assert result.variable_choices == [{"language": "English"}]

    def test_empty_user_input_options_becomes_none(self):
        config = dict(_API_CONFIG, userInputOptions=[])
        result = self._build(config=config)
        assert result.user_input_options is None

    def test_non_empty_user_input_options_passed_through(self):
        result = self._build()
        assert result.user_input_options == ["What is 2+2?"]

    def test_max_attempts_from_config(self):
        result = self._build()
        assert result.max_attempts == 3

    def test_model_choices_provider_prefix_stripped(self):
        config = dict(_API_CONFIG, modelChoices=["OpenAI.gpt-4o", "Anthropic.claude-opus-4-5"])
        result = self._build(config=config)
        assert result.model_choices == ["gpt-4o", "claude-opus-4-5"]

    def test_judge_model_provider_prefix_stripped(self):
        config = dict(_API_CONFIG, judgeModel="OpenAI.gpt-4o")
        result = self._build(config=config)
        assert result.judge_model == "gpt-4o"

    def test_model_choices_without_prefix_unchanged(self):
        result = self._build()
        assert result.model_choices == ["gpt-4o", "gpt-4o-mini"]

    def test_judge_model_without_prefix_unchanged(self):
        result = self._build()
        assert result.judge_model == "gpt-4o"

    def test_model_with_multiple_dots_only_prefix_stripped(self):
        config = dict(_API_CONFIG, judgeModel="Anthropic.claude-opus-4.6")
        result = self._build(config=config)
        assert result.judge_model == "claude-opus-4.6"

    def test_callbacks_forwarded_from_options(self):
        handle_agent = AsyncMock(return_value="ok")
        handle_judge = AsyncMock(return_value=JUDGE_PASS_RESPONSE)
        options = _make_from_config_options(
            handle_agent_call=handle_agent,
            handle_judge_call=handle_judge,
            on_passing_result=MagicMock(),
            on_failing_result=MagicMock(),
        )
        result = self._build(options=options)
        assert result.handle_agent_call is handle_agent
        assert result.handle_judge_call is handle_judge
        assert result.on_passing_result is options.on_passing_result
        assert result.on_failing_result is options.on_failing_result

    def test_persist_and_forward_posts_result_on_status_update(self):
        result = self._build()
        ctx = OptimizationContext(
            scores={},
            completion_response="The answer is 4.",
            current_instructions="Be helpful.",
            current_parameters={"temperature": 0.7},
            current_variables={"language": "English"},
            current_model="gpt-4o",
            user_input="What is 2+2?",
            iteration=1,
        )
        result.on_status_update("generating", ctx)
        self.api_client.post_agent_optimization_result.assert_called_once()
        call_args = self.api_client.post_agent_optimization_result.call_args
        assert call_args[0][0] == "my-project"
        assert call_args[0][1] == "opt-uuid-123"

    def test_persist_and_forward_payload_has_correct_field_names(self):
        result = self._build()
        ctx = OptimizationContext(
            scores={"j": JudgeResult(score=0.9, rationale="Good.")},
            completion_response="Paris.",
            current_instructions="Be helpful.",
            current_parameters={"temperature": 0.5},
            current_variables={},
            current_model="gpt-4o",
            user_input="Capital of France?",
            iteration=2,
        )
        result.on_status_update("evaluating", ctx)
        payload = self.api_client.post_agent_optimization_result.call_args[0][2]
        assert payload["instructions"] == "Be helpful."
        assert payload["parameters"] == {"temperature": 0.5}
        assert payload["completion_response"] == "Paris."
        assert payload["user_input"] == "Capital of France?"
        assert payload["iteration"] == 2
        assert "j" in payload["scores"]

    def test_persist_and_forward_includes_run_id_and_version(self):
        result = self._build()
        ctx = OptimizationContext(
            scores={}, completion_response="", current_instructions="",
            current_parameters={}, current_variables={}, iteration=1,
        )
        result.on_status_update("generating", ctx)
        payload = self.api_client.post_agent_optimization_result.call_args[0][2]
        assert payload["run_id"] == "run-uuid-456"
        assert payload["config_optimization_version"] == 2

    @pytest.mark.parametrize("sdk_status,expected_status,expected_activity", [
        ("init", "RUNNING", "PENDING"),
        ("generating", "RUNNING", "GENERATING"),
        ("evaluating", "RUNNING", "EVALUATING"),
        ("generating variation", "RUNNING", "GENERATING_VARIATION"),
        ("turn completed", "RUNNING", "COMPLETED"),
        ("success", "PASSED", "COMPLETED"),
        ("failure", "FAILED", "COMPLETED"),
    ])
    def test_status_mapping(self, sdk_status, expected_status, expected_activity):
        result = self._build()
        ctx = OptimizationContext(
            scores={}, completion_response="", current_instructions="",
            current_parameters={}, current_variables={}, iteration=1,
        )
        result.on_status_update(sdk_status, ctx)
        payload = self.api_client.post_agent_optimization_result.call_args[0][2]
        assert payload["status"] == expected_status
        assert payload["activity"] == expected_activity

    def test_user_on_status_update_chained_after_post(self):
        call_order = []
        self.api_client.post_agent_optimization_result.side_effect = (
            lambda *a, **kw: call_order.append("post")
        )
        user_cb = MagicMock(side_effect=lambda s, c: call_order.append("user"))
        options = _make_from_config_options(on_status_update=user_cb)
        result = self._build(options=options)
        ctx = OptimizationContext(
            scores={}, completion_response="", current_instructions="",
            current_parameters={}, current_variables={}, iteration=1,
        )
        result.on_status_update("generating", ctx)
        assert call_order == ["post", "user"]

    def test_user_on_status_update_exception_does_not_propagate(self):
        options = _make_from_config_options(
            on_status_update=MagicMock(side_effect=RuntimeError("cb boom"))
        )
        result = self._build(options=options)
        ctx = OptimizationContext(
            scores={}, completion_response="", current_instructions="",
            current_parameters={}, current_variables={}, iteration=1,
        )
        result.on_status_update("generating", ctx)  # must not raise

    def test_payload_history_not_included(self):
        result = self._build()
        ctx = OptimizationContext(
            scores={}, completion_response="", current_instructions="",
            current_parameters={}, current_variables={}, iteration=1,
        )
        result.on_status_update("generating", ctx)
        payload = self.api_client.post_agent_optimization_result.call_args[0][2]
        assert "history" not in payload


# ---------------------------------------------------------------------------
# optimize_from_config
# ---------------------------------------------------------------------------


class TestOptimizeFromConfig:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()

    def _make_client_with_key(self) -> OptimizationClient:
        with patch.dict("os.environ", {"LAUNCHDARKLY_API_KEY": "test-api-key"}):
            return _make_client(self.mock_ldai)

    def _make_client_without_key(self) -> OptimizationClient:
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("LAUNCHDARKLY_API_KEY", None)
            client = OptimizationClient(self.mock_ldai)
            client._has_api_key = False
            client._api_key = None
            return client

    async def test_raises_without_api_key(self):
        client = self._make_client_without_key()
        options = _make_from_config_options()
        with pytest.raises(ValueError, match="LAUNCHDARKLY_API_KEY is not set"):
            await client.optimize_from_config("my-opt", options)

    async def test_fetches_config_and_uses_ai_config_key(self):
        client = self._make_client_with_key()
        mock_api = _make_mock_api_client()
        mock_api.get_agent_optimization = MagicMock(return_value=dict(_API_CONFIG))

        with patch("ldai_optimization.client.LDApiClient", return_value=mock_api):
            options = _make_from_config_options()
            await client.optimize_from_config("my-opt", options)

        mock_api.get_agent_optimization.assert_called_once_with("my-project", "my-opt")
        assert client._agent_key == "my-agent"

    async def test_posts_result_on_each_status_event(self):
        client = self._make_client_with_key()
        mock_api = _make_mock_api_client()
        mock_api.get_agent_optimization = MagicMock(return_value=dict(_API_CONFIG))

        with patch("ldai_optimization.client.LDApiClient", return_value=mock_api):
            options = _make_from_config_options()
            await client.optimize_from_config("my-opt", options)

        assert mock_api.post_agent_optimization_result.call_count >= 1

    async def test_user_on_status_update_called_during_run(self):
        client = self._make_client_with_key()
        mock_api = _make_mock_api_client()
        mock_api.get_agent_optimization = MagicMock(return_value=dict(_API_CONFIG))
        statuses = []

        with patch("ldai_optimization.client.LDApiClient", return_value=mock_api):
            options = _make_from_config_options(
                on_status_update=lambda status, ctx: statuses.append(status)
            )
            await client.optimize_from_config("my-opt", options)

        assert "generating" in statuses
        assert "success" in statuses

    async def test_custom_base_url_passed_to_api_client(self):
        client = self._make_client_with_key()

        with patch("ldai_optimization.client.LDApiClient") as MockLDApiClient:
            instance = _make_mock_api_client()
            instance.get_agent_optimization = MagicMock(return_value=dict(_API_CONFIG))
            MockLDApiClient.return_value = instance
            options = _make_from_config_options(base_url="https://staging.launchdarkly.com")
            await client.optimize_from_config("my-opt", options)

        MockLDApiClient.assert_called_once_with(
            "test-api-key", base_url="https://staging.launchdarkly.com"
        )

    async def test_no_base_url_does_not_pass_kwarg(self):
        client = self._make_client_with_key()

        with patch("ldai_optimization.client.LDApiClient") as MockLDApiClient:
            instance = _make_mock_api_client()
            instance.get_agent_optimization = MagicMock(return_value=dict(_API_CONFIG))
            MockLDApiClient.return_value = instance
            options = _make_from_config_options()
            await client.optimize_from_config("my-opt", options)

        MockLDApiClient.assert_called_once_with("test-api-key")

    async def test_returns_optimization_context_on_success(self):
        client = self._make_client_with_key()
        mock_api = _make_mock_api_client()
        mock_api.get_agent_optimization = MagicMock(return_value=dict(_API_CONFIG))

        with patch("ldai_optimization.client.LDApiClient", return_value=mock_api):
            options = _make_from_config_options()
            result = await client.optimize_from_config("my-opt", options)

        assert isinstance(result, OptimizationContext)
        assert result.completion_response == "The answer is 4."
