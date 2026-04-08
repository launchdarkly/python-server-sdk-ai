"""Tests for OptimizationClient."""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ldai import AIAgentConfig, AIJudgeConfig, AIJudgeConfigDefault, LDAIClient
from ldai.models import LDMessage, ModelConfig
from ldclient import Context

from ldai_optimization.client import OptimizationClient, _compute_validation_count
from ldai_optimization.dataclasses import (
    AIJudgeCallConfig,
    GroundTruthOptimizationOptions,
    GroundTruthSample,
    JudgeResult,
    OptimizationContext,
    OptimizationFromConfigOptions,
    OptimizationJudge,
    OptimizationJudgeContext,
    OptimizationOptions,
    OptimizationResponse,
    ToolDefinition,
)
from ldai_optimization.prompts import (
    _acceptance_criteria_implies_duration_optimization,
    build_new_variation_prompt,
    variation_prompt_acceptance_criteria,
    variation_prompt_improvement_instructions,
    variation_prompt_overfit_warning,
    variation_prompt_preamble,
)
from ldai_optimization.util import interpolate_variables
from ldai_optimization.util import (
    handle_evaluation_tool_call,
    handle_variation_tool_call,
    restore_variable_placeholders,
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
        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output="The capital of France is Paris."))
    if handle_judge_call is None:
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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
        options = _make_options(judges=None, handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="x")))
        # Need on_turn to satisfy validation — inject directly
        options_with_on_turn = OptimizationOptions(
            context_choices=[LD_CONTEXT],
            max_attempts=1,
            model_choices=["gpt-4o"],
            judge_model="gpt-4o",
            variable_choices=[{}],
            handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="x")),
            handle_judge_call=AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE)),
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
# _evaluate_acceptance_judge
# ---------------------------------------------------------------------------


class TestEvaluateAcceptanceJudge:
    def setup_method(self):
        self.client = _make_client()
        agent_config = _make_agent_config()
        self.client._agent_key = "test-agent"
        self.client._agent_config = agent_config
        self.client._initialize_class_members_from_config(agent_config)
        self.handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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
        key, config, ctx = call_args.args
        assert key == "relevance"
        assert isinstance(config, AIJudgeCallConfig)
        assert isinstance(ctx, OptimizationJudgeContext)

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
        _, config, _ = self.handle_judge_call.call_args.args
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
        _, config, _ = self.handle_judge_call.call_args.args
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
        _, config, ctx = self.handle_judge_call.call_args.args
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
        _, config, _ = call_args.args
        assert statement in config.instructions

    async def test_no_structured_output_tool_in_judge_config(self):
        """Structured output tool must not be injected — judges return plain JSON."""
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
        _, config, _ = call_args.args
        tools = config.model.get_parameter("tools") or []
        assert tools == []

    async def test_agent_tools_included_in_config_tools(self):
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
        _, config, _ = call_args.args
        tools = config.model.get_parameter("tools") or []
        tool_names = [t["name"] for t in tools]
        assert tool_names == ["lookup"]

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
        _, _, ctx = call_args.args
        assert ctx.variables == variables

    async def test_duration_context_added_to_instructions_when_latency_keyword_present(self):
        """When acceptance statement has a latency keyword and agent_duration_ms is provided,
        the instructions mention the duration."""
        judge = OptimizationJudge(
            threshold=0.8,
            acceptance_statement="The response must be fast.",
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="speed",
            optimization_judge=judge,
            completion_response="Here is the answer.",
            iteration=2,
            reasoning_history="",
            user_input="Tell me something.",
            agent_duration_ms=1500.0,
        )
        _, config, _ = self.handle_judge_call.call_args.args
        assert "1500ms" in config.instructions
        assert "mention the duration" in config.instructions

    async def test_duration_context_includes_baseline_comparison_when_history_present(self):
        """When history[0] has a duration, the judge instructions include a baseline comparison."""
        self.client._history = [
            OptimizationContext(
                scores={},
                completion_response="old response",
                current_instructions="Do X.",
                current_parameters={},
                current_variables={},
                iteration=1,
                duration_ms=2000.0,
            )
        ]
        judge = OptimizationJudge(
            threshold=0.8,
            acceptance_statement="Responses should have low latency.",
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="latency",
            optimization_judge=judge,
            completion_response="Here is the answer.",
            iteration=2,
            reasoning_history="",
            user_input="Tell me something.",
            agent_duration_ms=1500.0,
        )
        _, config, _ = self.handle_judge_call.call_args.args
        assert "1500ms" in config.instructions
        assert "2000ms" in config.instructions
        assert "faster" in config.instructions

    async def test_duration_context_says_slower_when_candidate_is_slower(self):
        """When the candidate is slower than baseline, the instructions say 'slower'."""
        self.client._history = [
            OptimizationContext(
                scores={},
                completion_response="old response",
                current_instructions="Do X.",
                current_parameters={},
                current_variables={},
                iteration=1,
                duration_ms=1000.0,
            )
        ]
        judge = OptimizationJudge(
            threshold=0.8,
            acceptance_statement="The response must be fast.",
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="speed",
            optimization_judge=judge,
            completion_response="Here is the answer.",
            iteration=2,
            reasoning_history="",
            user_input="Tell me something.",
            agent_duration_ms=1800.0,
        )
        _, config, _ = self.handle_judge_call.call_args.args
        assert "slower" in config.instructions

    async def test_duration_context_not_added_when_no_latency_keyword(self):
        """When acceptance statement has no latency keyword, duration is not injected."""
        judge = OptimizationJudge(
            threshold=0.8,
            acceptance_statement="The response must be accurate.",
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="accuracy",
            optimization_judge=judge,
            completion_response="Paris.",
            iteration=1,
            reasoning_history="",
            user_input="Capital of France?",
            agent_duration_ms=2000.0,
        )
        _, config, _ = self.handle_judge_call.call_args.args
        assert "2000ms" not in config.instructions
        assert "duration" not in config.instructions.lower() or "acceptance" in config.instructions.lower()

    async def test_duration_context_not_added_when_agent_duration_ms_is_none(self):
        """When agent_duration_ms is None, no duration block is added even if keyword matches."""
        judge = OptimizationJudge(
            threshold=0.8,
            acceptance_statement="The response must be fast.",
        )
        await self.client._evaluate_acceptance_judge(
            judge_key="speed",
            optimization_judge=judge,
            completion_response="Here is the answer.",
            iteration=1,
            reasoning_history="",
            user_input="Tell me something.",
            agent_duration_ms=None,
        )
        _, config, _ = self.handle_judge_call.call_args.args
        assert "mention the duration" not in config.instructions

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
        self.handle_judge_call.return_value = OptimizationResponse(output="not json at all")
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
        self.handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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
        key, config, ctx = call_args.args
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
        _, config, _ = self.handle_judge_call.call_args.args
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
        _, config, _ = self.handle_judge_call.call_args.args
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
        _, config, ctx = self.handle_judge_call.call_args.args
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
        _, config, _ = self.handle_judge_call.call_args.args
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

    async def test_agent_tools_included_without_evaluation_tool(self):
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
        _, config, _ = self.handle_judge_call.call_args.args
        tools = config.model.get_parameter("tools") or []
        names = [t["name"] for t in tools]
        assert names == ["search"]


# ---------------------------------------------------------------------------
# _execute_agent_turn
# ---------------------------------------------------------------------------


class TestExecuteAgentTurn:
    def setup_method(self):
        self.agent_response = "Paris is the capital of France."
        self.handle_agent_call = AsyncMock(return_value=OptimizationResponse(output=self.agent_response))
        self.handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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
        key, config, passed_ctx = self.handle_agent_call.call_args.args
        assert key == "test-agent"
        assert isinstance(config, AIAgentConfig)
        assert passed_ctx is ctx

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
        _, config, _ = self.handle_agent_call.call_args.args
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
        self.handle_agent_call = AsyncMock(return_value=OptimizationResponse(output=VARIATION_RESPONSE))
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

    async def test_no_structured_output_tool_in_variation_config(self):
        """Variation turn must not inject the structured-output tool — prompts use plain JSON."""
        await self.client._generate_new_variation(iteration=1, variables={})
        _, config, _ = self.handle_agent_call.call_args.args
        tools = config.model.get_parameter("tools") or []
        assert tools == []

    async def test_variation_call_uses_three_arg_signature(self):
        """handle_agent_call receives exactly (key, config, context) — no tools arg."""
        await self.client._generate_new_variation(iteration=1, variables={})
        assert len(self.handle_agent_call.call_args.args) == 3

    async def test_model_not_updated_when_not_in_model_choices(self):
        bad_response = json.dumps({
            "current_instructions": "New instructions.",
            "current_parameters": {},
            "model": "some-unknown-model",
        })
        self.handle_agent_call.return_value = OptimizationResponse(output=bad_response)
        original_model = self.client._current_model
        await self.client._generate_new_variation(iteration=1, variables={})
        assert self.client._current_model == original_model

    async def test_retries_on_empty_response_and_succeeds(self):
        """First attempt returns empty string; second returns valid JSON — succeeds."""
        self.handle_agent_call.side_effect = [
            OptimizationResponse(output=""),           # attempt 1: empty
            OptimizationResponse(output=VARIATION_RESPONSE),  # attempt 2: valid
        ]
        await self.client._generate_new_variation(iteration=1, variables={})
        assert self.client._current_instructions == "You are an improved assistant."
        assert self.handle_agent_call.call_count == 2

    async def test_retries_on_unparseable_response_and_succeeds(self):
        """First attempt returns non-JSON text; second returns valid JSON — succeeds."""
        self.handle_agent_call.side_effect = [
            OptimizationResponse(output="Sorry, I cannot do that."),  # attempt 1: not JSON
            OptimizationResponse(output=VARIATION_RESPONSE),           # attempt 2: valid
        ]
        await self.client._generate_new_variation(iteration=1, variables={})
        assert self.client._current_instructions == "You are an improved assistant."
        assert self.handle_agent_call.call_count == 2

    async def test_raises_after_max_retries_exhausted(self):
        """All three attempts return empty strings — ValueError is raised."""
        self.handle_agent_call.side_effect = [
            OptimizationResponse(output=""),
            OptimizationResponse(output=""),
            OptimizationResponse(output=""),
        ]
        with pytest.raises(ValueError, match="Failed to parse structured output"):
            await self.client._generate_new_variation(iteration=1, variables={})
        assert self.handle_agent_call.call_count == 3


# ---------------------------------------------------------------------------
# Full optimization loop
# ---------------------------------------------------------------------------


class TestRunOptimization:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()

    async def test_succeeds_on_first_attempt_when_judge_passes(self):
        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output="The capital of France is Paris."))
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
        )
        result = await client.optimize_from_options("test-agent", options)
        assert result.scores["accuracy"].score == 1.0
        # 1 initial agent call + 1 validation sample (repeated draw — only 1 variable choice)
        assert handle_agent_call.call_count == 2

    async def test_generates_variation_when_judge_fails(self):
        agent_responses = [
            OptimizationResponse(output="Bad answer."),
            OptimizationResponse(output=VARIATION_RESPONSE),  # variation generation
            OptimizationResponse(output="Better answer."),
            OptimizationResponse(output="Better answer."),    # 1 validation sample (repeated draw — only 1 variable choice)
        ]
        handle_agent_call = AsyncMock(side_effect=agent_responses)
        judge_responses = [
            OptimizationResponse(output=JUDGE_FAIL_RESPONSE),
            OptimizationResponse(output=JUDGE_PASS_RESPONSE),
            OptimizationResponse(output=JUDGE_PASS_RESPONSE),
        ]
        handle_judge_call = AsyncMock(side_effect=judge_responses)
        client = _make_client(self.mock_ldai)
        options = _make_options(
            handle_agent_call=handle_agent_call,
            handle_judge_call=handle_judge_call,
            max_attempts=3,
        )
        result = await client.optimize_from_options("test-agent", options)
        assert result.scores["accuracy"].score == 1.0
        # 1 agent + 1 variation + 1 agent + 1 validation sample
        assert handle_agent_call.call_count == 4

    async def test_returns_last_context_after_max_attempts(self):
        # The max_attempts guard fires before variation on the final iteration,
        # so only iterations 1 and 2 produce a variation call.
        handle_agent_call = AsyncMock(side_effect=[
            OptimizationResponse(output="Bad answer."),       # iteration 1: agent
            OptimizationResponse(output=VARIATION_RESPONSE),  # iteration 1: variation
            OptimizationResponse(output="Still bad."),        # iteration 2: agent
            OptimizationResponse(output=VARIATION_RESPONSE),  # iteration 2: variation
            OptimizationResponse(output="Still bad."),        # iteration 3: agent (max_attempts reached — no variation)
        ])
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_FAIL_RESPONSE))
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
        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output="Great answer."))
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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
            OptimizationResponse(output="Bad."),             # iteration 1: agent
            OptimizationResponse(output=VARIATION_RESPONSE), # iteration 1: variation
            OptimizationResponse(output="Still bad."),       # iteration 2: agent (max_attempts reached — no variation)
        ])
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_FAIL_RESPONSE))
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
        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output="Answer."))
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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
        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output="Good answer."))
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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
# _compute_validation_count
# ---------------------------------------------------------------------------


class TestComputeValidationCount:
    def test_pool_of_10_returns_2(self):
        assert _compute_validation_count(10) == 2

    def test_pool_of_20_returns_5(self):
        assert _compute_validation_count(20) == 5

    def test_pool_of_16_returns_4(self):
        assert _compute_validation_count(16) == 4

    def test_small_pool_floors_at_2(self):
        assert _compute_validation_count(1) == 2
        assert _compute_validation_count(3) == 2

    def test_large_pool_caps_at_5(self):
        assert _compute_validation_count(100) == 5

    def test_pool_of_8_returns_2(self):
        assert _compute_validation_count(8) == 2


# ---------------------------------------------------------------------------
# Validation phase (chaos mode)
# ---------------------------------------------------------------------------

# Helper: build OptimizationOptions with multiple variable choices so the
# validation phase has a non-empty distinct pool to sample from.
def _make_multi_options(
    *,
    variable_count: int = 8,
    user_input_options=None,
    on_turn=None,
    handle_agent_call=None,
    handle_judge_call=None,
    on_passing_result=None,
    max_attempts: int = 5,
) -> OptimizationOptions:
    if handle_agent_call is None:
        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output="answer"))
    if handle_judge_call is None:
        handle_judge_call = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
    judges = None if on_turn is not None else {
        "acc": OptimizationJudge(threshold=0.8, acceptance_statement="Be accurate.")
    }
    return OptimizationOptions(
        context_choices=[LD_CONTEXT],
        max_attempts=max_attempts,
        model_choices=["gpt-4o"],
        judge_model="gpt-4o",
        variable_choices=[{"x": i} for i in range(variable_count)],
        user_input_options=user_input_options,
        handle_agent_call=handle_agent_call,
        handle_judge_call=handle_judge_call,
        judges=judges,
        on_turn=on_turn,
        on_passing_result=on_passing_result,
    )


class TestValidationPhase:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()

    def _make_client(self) -> OptimizationClient:
        return _make_client(self.mock_ldai)

    async def test_on_passing_result_fires_only_after_all_validation_passes(self):
        """on_passing_result must not fire until all validation samples pass."""
        on_passing = MagicMock()
        client = self._make_client()
        # 8 variable_choices → validation_count = 2; all judges always pass
        opts = _make_multi_options(on_passing_result=on_passing)
        await client.optimize_from_options("test-agent", opts)
        on_passing.assert_called_once()

    async def test_validation_runs_additional_agent_calls(self):
        """With 8 variable choices, validation runs 2 extra agent calls after the initial pass."""
        call_count = [0]

        async def counting_agent(key, config, ctx):
            call_count[0] += 1
            return OptimizationResponse(output="answer")

        client = self._make_client()
        opts = _make_multi_options(handle_agent_call=counting_agent)
        await client.optimize_from_options("test-agent", opts)
        # 1 initial pass + 2 validation samples
        assert call_count[0] == 3

    async def test_validation_failure_suppresses_on_passing_result_then_retries(self):
        """When a validation sample fails, on_passing_result is not fired and the loop retries."""
        turn_calls = [0]

        def on_turn(ctx):
            turn_calls[0] += 1
            # call 1: initial pass, call 2: first validation FAIL, everything else passes
            return turn_calls[0] != 2

        on_passing = MagicMock()
        client = self._make_client()
        opts = _make_multi_options(
            on_turn=on_turn,
            # 8 items → validation_count = 2
            variable_count=8,
            handle_agent_call=AsyncMock(side_effect=[
                OptimizationResponse(output="iter1"),            # initial turn (passes)
                OptimizationResponse(output="val_iter2"),        # validation sample 1 (fails)
                OptimizationResponse(output=VARIATION_RESPONSE),  # variation generation
                OptimizationResponse(output="iter3"),            # new attempt initial (passes)
                OptimizationResponse(output="val_iter4"),        # new validation sample 1 (passes)
                OptimizationResponse(output="val_iter5"),        # new validation sample 2 (passes)
            ]),
            on_passing_result=on_passing,
            max_attempts=3,
        )
        result = await client.optimize_from_options("test-agent", opts)
        # Eventually succeeds after one failed validation cycle
        on_passing.assert_called_once()
        assert result is not None

    async def test_validation_does_not_reuse_passing_turn_variable(self):
        """The variable set used in the initial passing turn must not appear in validation."""
        seen_variables = []

        async def capture_agent(key, config, ctx):
            seen_variables.append(ctx.current_variables)
            return OptimizationResponse(output="answer")

        client = self._make_client()
        opts = _make_multi_options(handle_agent_call=capture_agent, variable_count=8)
        await client.optimize_from_options("test-agent", opts)

        # First call is the initial passing turn
        initial_vars = seen_variables[0]
        # Remaining calls are validation samples — none should match the initial
        for val_vars in seen_variables[1:]:
            assert val_vars != initial_vars, (
                f"Validation reused the passing turn's variables: {initial_vars}"
            )

    async def test_validation_uses_user_input_options_as_pool_when_provided(self):
        """When user_input_options is provided, validation samples from that pool."""
        seen_inputs = []

        async def capture_agent(key, config, ctx):
            seen_inputs.append(ctx.user_input)
            return OptimizationResponse(output="answer")

        client = self._make_client()
        user_inputs = [f"question {i}" for i in range(8)]
        opts = _make_multi_options(
            handle_agent_call=capture_agent,
            user_input_options=user_inputs,
        )
        await client.optimize_from_options("test-agent", opts)

        # Initial input is at index 0; all validation inputs must be different
        initial_input = seen_inputs[0]
        for val_input in seen_inputs[1:]:
            assert val_input != initial_input, (
                f"Validation reused the passing turn's user_input: {initial_input}"
            )

    async def test_pool_exhaustion_caps_validation_at_available_distinct_items(self):
        """When fewer distinct items remain than validation_count, all available ones are used."""
        call_count = [0]

        async def counting_agent(key, config, ctx):
            call_count[0] += 1
            return OptimizationResponse(output="answer")

        client = self._make_client()
        # 3 variable choices → _compute_validation_count(3) = 2, but only 2 remain after
        # excluding the passing item, so validation_count is still 2 (min of 2 and 2)
        opts = _make_multi_options(handle_agent_call=counting_agent, variable_count=3)
        await client.optimize_from_options("test-agent", opts)
        # 1 initial + 2 validation (uses all remaining distinct items)
        assert call_count[0] == 3

    async def test_single_variable_choice_falls_back_to_repeated_draw(self):
        """With only 1 variable choice validation still runs 1 sample (repeated draw)."""
        call_count = [0]

        async def counting_agent(key, config, ctx):
            call_count[0] += 1
            return OptimizationResponse(output="answer")

        client = self._make_client()
        opts = _make_multi_options(handle_agent_call=counting_agent, variable_count=1)
        await client.optimize_from_options("test-agent", opts)
        # 1 initial pass + 1 validation sample (repeated draw from the only item)
        assert call_count[0] == 2

    async def test_validation_does_not_consume_attempt_budget(self):
        """Validation samples must not count against max_attempts.

        With max_attempts=2 and 8 variable choices (validation_count=2), a failed
        validation on attempt 1 should still leave a full attempt 2 available.
        Without the fix, iteration would be inflated to 3 after validation, which
        exceeds max_attempts=2 and would trigger _handle_failure prematurely.
        """
        turn_calls = [0]

        def on_turn(ctx):
            turn_calls[0] += 1
            # attempt 1 passes initial, validation sample 1 fails
            # attempt 2 passes initial and all validation
            return turn_calls[0] != 2

        on_passing = MagicMock()
        client = self._make_client()
        opts = _make_multi_options(
            on_turn=on_turn,
            variable_count=8,
            handle_agent_call=AsyncMock(side_effect=[
                OptimizationResponse(output="iter1"),            # attempt 1 initial (passes)
                OptimizationResponse(output="val_iter"),         # validation sample 1 (fails)
                OptimizationResponse(output=VARIATION_RESPONSE),  # variation generation
                OptimizationResponse(output="iter2"),            # attempt 2 initial (passes)
                OptimizationResponse(output="val_iter3"),        # validation sample 1 (passes)
                OptimizationResponse(output="val_iter4"),        # validation sample 2 (passes)
            ]),
            on_passing_result=on_passing,
            max_attempts=2,
        )
        result = await client.optimize_from_options("test-agent", opts)
        on_passing.assert_called_once()
        assert result is not None

    async def test_validating_status_emitted(self):
        """The 'validating' status must be emitted when entering the validation phase."""
        statuses = []
        client = self._make_client()
        opts = _make_multi_options()
        opts.on_status_update = lambda s, ctx: statuses.append(s)
        await client.optimize_from_options("test-agent", opts)
        assert "validating" in statuses


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
# Variation prompt — overfitting warning section
# ---------------------------------------------------------------------------


class TestVariationPromptOverfitWarning:
    def _make_ctx(self, user_input=None, variables=None, iteration=1):
        return OptimizationContext(
            iteration=iteration,
            current_instructions=AGENT_INSTRUCTIONS,
            current_parameters={},
            current_model="gpt-4o",
            current_variables=variables or {},
            user_input=user_input,
            completion_response=None,
            scores={},
        )

    def test_returns_empty_string_with_no_history(self):
        assert variation_prompt_overfit_warning([]) == ""

    def test_contains_general_overfitting_reminder(self):
        ctx = self._make_ctx(user_input="What is 2+2?")
        section = variation_prompt_overfit_warning([ctx])
        assert "OVERFITTING" in section.upper()
        assert "generalise" in section.lower() or "generalize" in section.lower() or "generaliz" in section.lower() or "general" in section.lower()

    def test_includes_recent_user_input(self):
        ctx = self._make_ctx(user_input="What is the capital of France?")
        section = variation_prompt_overfit_warning([ctx])
        assert "What is the capital of France?" in section

    def test_includes_recent_variables_as_structured_breakdown(self):
        ctx = self._make_ctx(variables={"language": "English", "tone": "formal"})
        section = variation_prompt_overfit_warning([ctx])
        # Keys (placeholder names) and values must both appear
        assert "{{language}}" in section
        assert '"English"' in section
        assert "{{tone}}" in section
        assert '"formal"' in section

    def test_variables_section_labels_name_vs_value(self):
        ctx = self._make_ctx(variables={"user_id": "user-125"})
        section = variation_prompt_overfit_warning([ctx])
        assert "{{user_id}}" in section
        assert '"user-125"' in section
        assert "placeholder" in section.lower()
        assert "value" in section.lower()
        # Must NOT render as a raw Python dict
        assert "{'user_id': 'user-125'}" not in section

    def test_uses_most_recent_history_entry(self):
        ctx_old = self._make_ctx(user_input="old question", iteration=1)
        ctx_new = self._make_ctx(user_input="new question", iteration=2)
        section = variation_prompt_overfit_warning([ctx_old, ctx_new])
        assert "new question" in section
        assert "old question" not in section

    def test_omits_user_input_line_when_none(self):
        ctx = self._make_ctx(user_input=None, variables={"lang": "en"})
        section = variation_prompt_overfit_warning([ctx])
        assert "User input" not in section
        assert "lang" in section

    def test_omits_variables_line_when_empty(self):
        ctx = self._make_ctx(user_input="hello", variables={})
        section = variation_prompt_overfit_warning([ctx])
        assert "Variables" not in section
        assert "hello" in section

    def test_warning_appears_in_full_prompt_when_history_present(self):
        ctx = self._make_ctx(user_input="test question", variables={"k": "v"})
        prompt = build_new_variation_prompt(
            history=[ctx],
            judges=None,
            current_model="gpt-4o",
            current_instructions=AGENT_INSTRUCTIONS,
            current_parameters={},
            model_choices=["gpt-4o"],
            variable_choices=[{"k": "v"}],
            initial_instructions=AGENT_INSTRUCTIONS,
        )
        assert "OVERFITTING" in prompt.upper()
        assert "test question" in prompt

    def test_warning_absent_from_full_prompt_when_no_history(self):
        prompt = build_new_variation_prompt(
            history=[],
            judges=None,
            current_model="gpt-4o",
            current_instructions=AGENT_INSTRUCTIONS,
            current_parameters={},
            model_choices=["gpt-4o"],
            variable_choices=[{"k": "v"}],
            initial_instructions=AGENT_INSTRUCTIONS,
        )
        assert "OVERFITTING" not in prompt.upper()


# ---------------------------------------------------------------------------
# Variation prompt — preamble key-vs-value note
# ---------------------------------------------------------------------------


class TestVariationPromptPreamble:
    def test_contains_key_vs_value_important_note(self):
        preamble = variation_prompt_preamble()
        assert "IMPORTANT" in preamble
        assert "placeholder" in preamble.lower()
        assert "value" in preamble.lower()

    def test_never_use_value_as_placeholder_name(self):
        preamble = variation_prompt_preamble()
        assert "never" in preamble.lower()


# ---------------------------------------------------------------------------
# Variation prompt — placeholder table
# ---------------------------------------------------------------------------


class TestVariationPromptPlaceholderTable:
    _variable_choices = [
        {"user_id": "user-123", "trip_purpose": "business"},
        {"user_id": "user-125", "trip_purpose": "personal"},
    ]

    def _section(self, variable_choices=None, history=None):
        return variation_prompt_improvement_instructions(
            history=history or [],
            model_choices=["gpt-4o"],
            variable_choices=variable_choices or self._variable_choices,
            initial_instructions=AGENT_INSTRUCTIONS,
        )

    def test_placeholder_names_appear_in_table(self):
        section = self._section()
        assert "{{user_id}}" in section
        assert "{{trip_purpose}}" in section

    def test_example_values_appear_alongside_keys(self):
        section = self._section()
        assert '"user-123"' in section or '"user-125"' in section
        assert '"business"' in section or '"personal"' in section

    def test_keys_and_values_clearly_separated(self):
        section = self._section()
        assert "example values" in section.lower()

    def test_bad_good_counterexamples_use_actual_values(self):
        section = self._section()
        # The bad example must reference a runtime value, good example the key
        assert "BAD" in section
        assert "GOOD" in section
        # At least one of the real values should appear in the bad example
        assert "user-123" in section or "user-125" in section \
            or "business" in section or "personal" in section

    def test_raw_placeholder_list_not_used(self):
        # The old format was a comma-separated list like "{{trip_purpose}}, {{user_id}}"
        # The new format is a structured table; confirm no bare comma-list
        section = self._section()
        assert "{{trip_purpose}}, {{user_id}}" not in section
        assert "{{user_id}}, {{trip_purpose}}" not in section

    def test_single_variable_choice(self):
        section = self._section(variable_choices=[{"lang": "en"}])
        assert "{{lang}}" in section
        assert '"en"' in section

    def test_table_appears_in_full_prompt(self):
        prompt = build_new_variation_prompt(
            history=[],
            judges=None,
            current_model="gpt-4o",
            current_instructions=AGENT_INSTRUCTIONS,
            current_parameters={},
            model_choices=["gpt-4o"],
            variable_choices=self._variable_choices,
            initial_instructions=AGENT_INSTRUCTIONS,
        )
        assert "{{user_id}}" in prompt
        assert "{{trip_purpose}}" in prompt
        assert "example values" in prompt.lower()


# ---------------------------------------------------------------------------
# interpolate_variables — hyphenated key support
# ---------------------------------------------------------------------------


class TestInterpolateVariables:
    def test_substitutes_standard_underscore_key(self):
        result = interpolate_variables("Hello {{user_id}}", {"user_id": "abc"})
        assert result == "Hello abc"

    def test_substitutes_hyphenated_key(self):
        result = interpolate_variables("Hello {{user-id}}", {"user-id": "abc"})
        assert result == "Hello abc"

    def test_leaves_unknown_placeholder_unchanged(self):
        result = interpolate_variables("Hello {{unknown}}", {"user_id": "abc"})
        assert result == "Hello {{unknown}}"

    def test_leaves_unknown_hyphenated_placeholder_unchanged(self):
        result = interpolate_variables("Hello {{bad-125}}", {"user_id": "abc"})
        assert result == "Hello {{bad-125}}"

    def test_mixed_keys_in_same_string(self):
        result = interpolate_variables(
            "{{user-id}} and {{trip_purpose}}",
            {"user-id": "u-1", "trip_purpose": "leisure"},
        )
        assert result == "u-1 and leisure"

    def test_empty_variables_leaves_text_unchanged(self):
        result = interpolate_variables("{{foo}} bar", {})
        assert result == "{{foo}} bar"


# ---------------------------------------------------------------------------
# restore_variable_placeholders
# ---------------------------------------------------------------------------


class TestRestoreVariablePlaceholders:
    _CHOICES = [{"user_id": "user-123", "trip_purpose": "business"}]

    def test_replaces_hardcoded_id_value(self):
        text = "Use the user ID user-123 to look up preferences."
        result, warnings = restore_variable_placeholders(text, self._CHOICES)
        assert "{{user_id}}" in result
        assert "user-123" not in result
        assert len(warnings) == 1
        assert "user-123" in warnings[0]
        assert "{{user_id}}" in warnings[0]

    def test_replaces_multiline_value_verbatim(self):
        multiline_value = "line one\nline two\nline three"
        choices = [{"body_text": multiline_value}]
        text = f"Instructions:\n{multiline_value}\nEnd."
        result, warnings = restore_variable_placeholders(text, choices)
        assert "{{body_text}}" in result
        assert multiline_value not in result
        assert len(warnings) == 1

    def test_skips_value_shorter_than_min_length(self):
        choices = [{"lang": "en"}]  # "en" is only 2 chars
        text = "Use language en for this request."
        result, warnings = restore_variable_placeholders(text, choices, min_value_length=3)
        assert result == text
        assert warnings == []

    def test_does_not_partially_match_longer_token(self):
        """'user-123' must not be replaced inside 'user-1234'."""
        text = "Contact user-1234 for help."
        result, warnings = restore_variable_placeholders(text, self._CHOICES)
        assert "user-1234" in result
        assert warnings == []

    def test_replaces_multiple_variables(self):
        text = "User user-123 is on a business trip."
        result, warnings = restore_variable_placeholders(text, self._CHOICES)
        assert "{{user_id}}" in result
        assert "{{trip_purpose}}" in result
        assert "user-123" not in result
        assert "business" not in result
        assert len(warnings) == 2

    def test_leaves_correct_placeholder_unchanged(self):
        text = "User {{user_id}} is on a {{trip_purpose}} trip."
        result, warnings = restore_variable_placeholders(text, self._CHOICES)
        assert result == text
        assert warnings == []

    def test_replaces_multiple_occurrences_of_same_value(self):
        text = "user-123 and user-123 are duplicates."
        result, warnings = restore_variable_placeholders(text, self._CHOICES)
        assert result == "{{user_id}} and {{user_id}} are duplicates."
        assert "2 occurrence(s)" in warnings[0]

    def test_longer_value_replaced_before_shorter_substring(self):
        """When one value is a prefix of another, the longer one is replaced first."""
        choices = [{"full_id": "user-123-admin", "short_id": "user-123"}]
        text = "Admin is user-123-admin, regular is user-123."
        result, warnings = restore_variable_placeholders(text, choices)
        assert "{{full_id}}" in result
        assert "{{short_id}}" in result
        assert "user-123-admin" not in result
        # The shorter value should not have corrupted the longer replacement
        assert result.count("{{full_id}}") == 1
        assert result.count("{{short_id}}") == 1

    def test_replaces_brace_wrapped_value_without_double_bracketing(self):
        """{{user-125}} must become {{user_id}}, not {{{{user_id}}}}."""
        text = "Fetch preferences for user {{user-123}}."
        result, warnings = restore_variable_placeholders(text, self._CHOICES)
        assert result == "Fetch preferences for user {{user_id}}."
        assert len(warnings) == 1

    def test_empty_variable_choices_returns_text_unchanged(self):
        text = "Some instructions here."
        result, warnings = restore_variable_placeholders(text, [])
        assert result == text
        assert warnings == []

    def test_warning_message_format(self):
        text = "Handle user user-123 carefully."
        _, warnings = restore_variable_placeholders(text, self._CHOICES)
        assert any("user-123" in w for w in warnings)
        assert any("{{user_id}}" in w for w in warnings)

    async def test_apply_variation_response_calls_restore_and_logs_warning(self):
        """_apply_new_variation_response must restore leaked values and log warnings."""
        leaked_instructions = "You serve user user-123 on a business trip."
        variation_response = json.dumps({
            "current_instructions": leaked_instructions,
            "current_parameters": {},
            "model": "gpt-4o",
        })
        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output=variation_response))
        client = _make_client()
        agent_config = _make_agent_config()
        client._agent_key = "test-agent"
        client._agent_config = agent_config
        client._initial_instructions = AGENT_INSTRUCTIONS
        client._initialize_class_members_from_config(agent_config)
        client._options = _make_options(
            handle_agent_call=handle_agent_call,
            variable_choices=[{"user_id": "user-123", "trip_purpose": "business"}],
        )

        with patch("ldai_optimization.client.logger") as mock_logger:
            await client._generate_new_variation(iteration=1, variables={})
            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "user-123" in str(call) or "business" in str(call)
            ]
            assert len(warning_calls) >= 1

        assert "{{user_id}}" in client._current_instructions
        assert "user-123" not in client._current_instructions


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
        handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="The answer is 4.")),
        handle_judge_call=AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE)),
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
        handle_agent = AsyncMock(return_value=OptimizationResponse(output="ok"))
        handle_judge = AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE))
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


# ---------------------------------------------------------------------------
# GroundTruthSample / GroundTruthOptimizationOptions dataclass validation
# ---------------------------------------------------------------------------


class TestGroundTruthSampleDataclass:
    def test_required_fields(self):
        s = GroundTruthSample(user_input="hi", expected_response="hello")
        assert s.user_input == "hi"
        assert s.expected_response == "hello"
        assert s.variables == {}

    def test_variables_populated(self):
        s = GroundTruthSample(user_input="hi", expected_response="hello", variables={"lang": "en"})
        assert s.variables == {"lang": "en"}


class TestGroundTruthOptimizationOptionsValidation:
    def _make(self, **overrides) -> GroundTruthOptimizationOptions:
        defaults = dict(
            context_choices=[LD_CONTEXT],
            ground_truth_responses=[
                GroundTruthSample(user_input="q1", expected_response="a1"),
            ],
            max_attempts=3,
            model_choices=["gpt-4o"],
            judge_model="gpt-4o",
            handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="ans")),
            handle_judge_call=AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE)),
            judges={
                "acc": OptimizationJudge(threshold=0.8, acceptance_statement="Be accurate.")
            },
        )
        defaults.update(overrides)
        return GroundTruthOptimizationOptions(**defaults)

    def test_valid_options_created(self):
        opts = self._make()
        assert len(opts.ground_truth_responses) == 1

    def test_raises_empty_context_choices(self):
        with pytest.raises(ValueError, match="context_choices"):
            self._make(context_choices=[])

    def test_raises_empty_model_choices(self):
        with pytest.raises(ValueError, match="model_choices"):
            self._make(model_choices=[])

    def test_raises_empty_ground_truth_responses(self):
        with pytest.raises(ValueError, match="ground_truth_responses"):
            self._make(ground_truth_responses=[])

    def test_raises_no_judges_and_no_on_turn(self):
        with pytest.raises(ValueError, match="judges or on_turn"):
            self._make(judges=None, on_turn=None)

    def test_on_turn_satisfies_criteria_requirement(self):
        opts = self._make(judges=None, on_turn=lambda ctx: True)
        assert opts.on_turn is not None


# ---------------------------------------------------------------------------
# _run_ground_truth_optimization / optimize_from_ground_truth_options
# ---------------------------------------------------------------------------


def _make_gt_options(**overrides) -> GroundTruthOptimizationOptions:
    defaults = dict(
        context_choices=[LD_CONTEXT],
        ground_truth_responses=[
            GroundTruthSample(user_input="What is 2+2?", expected_response="4", variables={"lang": "English"}),
            GroundTruthSample(user_input="What is 3+3?", expected_response="6", variables={"lang": "English"}),
        ],
        max_attempts=3,
        model_choices=["gpt-4o", "gpt-4o-mini"],
        judge_model="gpt-4o",
        handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="The answer is correct.")),
        handle_judge_call=AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE)),
        judges={
            "acc": OptimizationJudge(threshold=0.8, acceptance_statement="Be accurate.")
        },
    )
    defaults.update(overrides)
    return GroundTruthOptimizationOptions(**defaults)


class TestRunGroundTruthOptimization:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()

    def _make_client(self) -> OptimizationClient:
        return _make_client(self.mock_ldai)

    async def test_returns_list_of_contexts_on_success(self):
        client = self._make_client()
        opts = _make_gt_options()
        results = await client.optimize_from_ground_truth_options("test-agent", opts)
        assert isinstance(results, list)
        assert len(results) == 2
        for ctx in results:
            assert isinstance(ctx, OptimizationContext)

    async def test_each_context_has_correct_user_input(self):
        client = self._make_client()
        opts = _make_gt_options()
        results = await client.optimize_from_ground_truth_options("test-agent", opts)
        assert results[0].user_input == "What is 2+2?"
        assert results[1].user_input == "What is 3+3?"

    async def test_completion_response_set_on_each_context(self):
        client = self._make_client()
        opts = _make_gt_options(handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="42")))
        results = await client.optimize_from_ground_truth_options("test-agent", opts)
        for ctx in results:
            assert ctx.completion_response == "42"

    async def test_on_sample_result_called_per_sample(self):
        client = self._make_client()
        sample_results = []
        opts = _make_gt_options(on_sample_result=lambda ctx: sample_results.append(ctx))
        await client.optimize_from_ground_truth_options("test-agent", opts)
        assert len(sample_results) == 2

    async def test_on_passing_result_called_once_on_success(self):
        client = self._make_client()
        passing_calls = []
        opts = _make_gt_options(on_passing_result=lambda ctx: passing_calls.append(ctx))
        await client.optimize_from_ground_truth_options("test-agent", opts)
        assert len(passing_calls) == 1

    async def test_on_failing_result_called_when_max_attempts_exceeded(self):
        client = self._make_client()
        failing_calls = []
        opts = _make_gt_options(
            handle_judge_call=AsyncMock(return_value=OptimizationResponse(output=JUDGE_FAIL_RESPONSE)),
            max_attempts=2,
            on_failing_result=lambda ctx: failing_calls.append(ctx),
        )
        results = await client.optimize_from_ground_truth_options("test-agent", opts)
        assert isinstance(results, list)
        assert len(failing_calls) == 1

    async def test_generates_variation_when_any_sample_fails(self):
        client = self._make_client()
        judge_responses = [
            JUDGE_PASS_RESPONSE,       # sample 1 attempt 1 — pass
            JUDGE_FAIL_RESPONSE,       # sample 2 attempt 1 — fail → trigger variation
            JUDGE_PASS_RESPONSE,       # sample 1 attempt 2 — pass
            JUDGE_PASS_RESPONSE,       # sample 2 attempt 2 — pass
        ]
        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            resp = judge_responses[call_count]
            call_count += 1
            return OptimizationResponse(output=resp)

        opts = _make_gt_options(
            handle_judge_call=side_effect,
            handle_agent_call=AsyncMock(side_effect=[
                OptimizationResponse(output="ans1"),
                OptimizationResponse(output="ans2"),           # attempt 1 samples
                OptimizationResponse(output=VARIATION_RESPONSE),       # variation generation
                OptimizationResponse(output="ans3"),
                OptimizationResponse(output="ans4"),           # attempt 2 samples
            ]),
            max_attempts=3,
        )
        results = await client.optimize_from_ground_truth_options("test-agent", opts)
        assert isinstance(results, list)
        assert len(results) == 2

    async def test_iteration_numbers_are_linear_and_unique(self):
        client = self._make_client()
        opts = _make_gt_options()
        results = await client.optimize_from_ground_truth_options("test-agent", opts)
        iterations = [ctx.iteration for ctx in results]
        assert len(set(iterations)) == len(iterations)

    async def test_on_sample_result_exception_does_not_abort(self):
        client = self._make_client()

        def bad_callback(ctx):
            raise RuntimeError("boom")

        opts = _make_gt_options(on_sample_result=bad_callback)
        results = await client.optimize_from_ground_truth_options("test-agent", opts)
        assert len(results) == 2

    async def test_variables_from_samples_used_per_evaluation(self):
        client = self._make_client()
        received_contexts = []
        async def capture_agent_call(key, config, ctx):
            received_contexts.append(ctx)
            return OptimizationResponse(output="response")

        opts = _make_gt_options(
            ground_truth_responses=[
                GroundTruthSample(user_input="q1", expected_response="a1", variables={"lang": "English"}),
                GroundTruthSample(user_input="q2", expected_response="a2", variables={"lang": "French"}),
            ],
            handle_agent_call=capture_agent_call,
        )
        await client.optimize_from_ground_truth_options("test-agent", opts)
        assert received_contexts[0].current_variables == {"lang": "English"}
        assert received_contexts[1].current_variables == {"lang": "French"}

    async def test_model_falls_back_to_first_model_choice_when_agent_config_has_no_model(self):
        """When the LD agent config has no model name the first model_choices entry is used."""
        config_without_model = _make_agent_config(model_name="")
        mock_ldai = _make_ldai_client(agent_config=config_without_model)
        client = _make_client(mock_ldai)

        observed_models = []
        async def capture(key, config, ctx):
            observed_models.append(config.model.name if config.model else None)
            return OptimizationResponse(output="answer")

        opts = _make_gt_options(
            handle_agent_call=capture,
            model_choices=["gpt-4o", "gpt-4o-mini"],
        )
        await client.optimize_from_ground_truth_options("test-agent", opts)
        assert all(m == "gpt-4o" for m in observed_models), (
            f"Expected all agent calls to use 'gpt-4o' (fallback), got: {observed_models}"
        )

    async def test_missing_instructions_raises_value_error(self):
        """An agent config with no instructions raises ValueError before the loop starts."""
        config_no_instructions = _make_agent_config(instructions="")
        mock_ldai = _make_ldai_client(agent_config=config_no_instructions)
        # variation() also needs to return no instructions so the fallback doesn't hide the gap.
        mock_ldai._client.variation.return_value = {"instructions": ""}
        client = _make_client(mock_ldai)

        opts = _make_gt_options()
        with pytest.raises(ValueError, match="has no instructions configured"):
            await client.optimize_from_ground_truth_options("test-agent", opts)


# ---------------------------------------------------------------------------
# expected_response in judge evaluation
# ---------------------------------------------------------------------------


class TestExpectedResponseInJudges:
    def setup_method(self):
        self.client = _make_client()
        self.client._agent_key = "test-agent"
        self.client._options = _make_options()
        self.client._agent_config = _make_agent_config()
        self.client._initialize_class_members_from_config(_make_agent_config())

    async def test_expected_response_included_in_acceptance_judge_user_message(self):
        captured_configs = []

        async def capture_judge_call(key, config, ctx):
            captured_configs.append(config)
            return OptimizationResponse(output=JUDGE_PASS_RESPONSE)

        self.client._options = _make_options(
            judges={
                "acc": OptimizationJudge(threshold=0.8, acceptance_statement="Be accurate.")
            },
            handle_judge_call=capture_judge_call,
        )
        await self.client._execute_agent_turn(
            self.client._create_optimization_context(iteration=1, variables={}),
            1,
            expected_response="The expected answer is 42.",
        )
        assert len(captured_configs) == 1
        user_msg = captured_configs[0].messages[-1].content
        assert "The expected answer is 42." in user_msg

    async def test_expected_response_in_acceptance_judge_user_message(self):
        captured_configs = []

        async def capture_judge_call(key, config, ctx):
            captured_configs.append(config)
            return OptimizationResponse(output=JUDGE_PASS_RESPONSE)

        self.client._options = _make_options(
            judges={
                "acc": OptimizationJudge(threshold=0.8, acceptance_statement="Be accurate.")
            },
            handle_judge_call=capture_judge_call,
        )
        await self.client._execute_agent_turn(
            self.client._create_optimization_context(iteration=1, variables={}),
            1,
            expected_response="gold standard",
        )
        user_msg = captured_configs[0].messages[1].content
        assert "gold standard" in user_msg
        assert "expected response" in user_msg.lower()
        # Scoring instructions should now live in the user message, not the system prompt
        system_msg = captured_configs[0].messages[0].content
        assert "gold standard" not in system_msg

    async def test_no_expected_response_leaves_judge_messages_unchanged(self):
        captured_configs = []

        async def capture_judge_call(key, config, ctx):
            captured_configs.append(config)
            return OptimizationResponse(output=JUDGE_PASS_RESPONSE)

        self.client._options = _make_options(
            judges={
                "acc": OptimizationJudge(threshold=0.8, acceptance_statement="Be accurate.")
            },
            handle_judge_call=capture_judge_call,
        )
        await self.client._execute_agent_turn(
            self.client._create_optimization_context(iteration=1, variables={}),
            1,
        )
        user_msg = captured_configs[0].messages[-1].content
        assert "expected response" not in user_msg.lower()


# ---------------------------------------------------------------------------
# _build_options_from_config — ground truth path
# ---------------------------------------------------------------------------


_API_CONFIG_WITH_GT: Dict[str, Any] = {
    "id": "opt-gt-uuid",
    "key": "my-gt-optimization",
    "aiConfigKey": "my-agent",
    "maxAttempts": 3,
    "modelChoices": ["gpt-4o"],
    "judgeModel": "gpt-4o",
    "variableChoices": [{"lang": "English"}, {"lang": "French"}],
    "acceptanceStatements": [{"statement": "Be accurate.", "threshold": 0.9}],
    "judges": [],
    "userInputOptions": ["What is 2+2?", "What is 3+3?"],
    "groundTruthResponses": ["4", "6"],
    "version": 1,
    "createdAt": 1700000000,
}


class TestBuildOptionsFromConfigGroundTruth:
    def setup_method(self):
        self.client = _make_client()
        self.client._agent_key = "my-agent"
        self.client._initialize_class_members_from_config(_make_agent_config())
        self.client._options = _make_options()
        self.api_client = _make_mock_api_client()

    def _build(self, config=None, options=None):
        return self.client._build_options_from_config(
            config or dict(_API_CONFIG_WITH_GT),
            options or _make_from_config_options(),
            self.api_client,
            optimization_id="opt-gt-uuid",
            run_id="run-uuid-789",
        )

    def test_returns_ground_truth_options_when_gt_present(self):
        result = self._build()
        assert isinstance(result, GroundTruthOptimizationOptions)

    def test_samples_zipped_by_index(self):
        result = self._build()
        assert isinstance(result, GroundTruthOptimizationOptions)
        assert len(result.ground_truth_responses) == 2
        s0 = result.ground_truth_responses[0]
        assert s0.user_input == "What is 2+2?"
        assert s0.expected_response == "4"
        assert s0.variables == {"lang": "English"}
        s1 = result.ground_truth_responses[1]
        assert s1.user_input == "What is 3+3?"
        assert s1.expected_response == "6"
        assert s1.variables == {"lang": "French"}

    def test_model_choices_have_prefix_stripped(self):
        config = dict(_API_CONFIG_WITH_GT)
        config["modelChoices"] = ["OpenAI.gpt-4o"]
        result = self._build(config=config)
        assert isinstance(result, GroundTruthOptimizationOptions)
        assert result.model_choices == ["gpt-4o"]

    def test_raises_on_mismatched_lengths(self):
        config = dict(_API_CONFIG_WITH_GT)
        config["userInputOptions"] = ["only one input"]
        with pytest.raises(ValueError, match="same length"):
            self._build(config=config)

    def test_returns_standard_options_when_no_gt(self):
        config = dict(_API_CONFIG)  # no groundTruthResponses
        result = self._build(config=config)
        assert isinstance(result, OptimizationOptions)

    async def test_optimize_from_config_dispatches_to_gt_run(self):
        mock_ldai = _make_ldai_client()
        with patch.dict("os.environ", {"LAUNCHDARKLY_API_KEY": "test-key"}):
            client = _make_client(mock_ldai)
        mock_api = _make_mock_api_client()
        mock_api.get_agent_optimization = MagicMock(return_value=dict(_API_CONFIG_WITH_GT))

        with patch("ldai_optimization.client.LDApiClient", return_value=mock_api):
            options = _make_from_config_options(
                handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="correct answer")),
                handle_judge_call=AsyncMock(return_value=OptimizationResponse(output=JUDGE_PASS_RESPONSE)),
            )
            result = await client.optimize_from_config("my-gt-opt", options)

        assert isinstance(result, list)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _acceptance_criteria_implies_duration_optimization
# ---------------------------------------------------------------------------


class TestAcceptanceCriteriaImpliesDurationOptimization:
    def test_returns_false_when_judges_is_none(self):
        assert _acceptance_criteria_implies_duration_optimization(None) is False

    def test_returns_false_when_judges_is_empty(self):
        assert _acceptance_criteria_implies_duration_optimization({}) is False

    def test_returns_false_when_no_acceptance_statements(self):
        judges = {"quality": OptimizationJudge(threshold=0.8, judge_key="judge-1")}
        assert _acceptance_criteria_implies_duration_optimization(judges) is False

    def test_returns_false_when_acceptance_statement_has_no_latency_keywords(self):
        judges = {
            "accuracy": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be accurate and complete.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is False

    def test_detects_fast_keyword(self):
        judges = {
            "speed": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be fast.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_detects_faster_keyword(self):
        judges = {
            "speed": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The agent should respond faster.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_detects_latency_keyword(self):
        judges = {
            "perf": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The agent must have low latency.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_detects_duration_keyword(self):
        judges = {
            "perf": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="Minimize the duration of each response.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_detects_ms_keyword(self):
        judges = {
            "perf": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="Responses should complete in under 500ms.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_detects_response_time_phrase(self):
        judges = {
            "perf": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response time should be minimized.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_detects_efficient_keyword(self):
        judges = {
            "perf": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The model must be efficient.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_detects_snappy_keyword(self):
        judges = {
            "perf": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="Responses should feel snappy.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_case_insensitive_match(self):
        judges = {
            "perf": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The model must be EFFICIENT and FAST.",
            )
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_returns_true_when_any_judge_matches(self):
        judges = {
            "accuracy": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be accurate.",
            ),
            "speed": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be fast.",
            ),
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is True

    def test_returns_false_when_acceptance_statement_is_none(self):
        judges = {
            "quality": OptimizationJudge(threshold=0.8, acceptance_statement=None)
        }
        assert _acceptance_criteria_implies_duration_optimization(judges) is False


# ---------------------------------------------------------------------------
# _evaluate_duration
# ---------------------------------------------------------------------------


class TestEvaluateDuration:
    def setup_method(self):
        self.client = _make_client()
        self.client._options = _make_options()
        self.client._agent_config = _make_agent_config()
        self.client._initialize_class_members_from_config(_make_agent_config())

    def _ctx(self, duration_ms, iteration=1):
        return OptimizationContext(
            scores={},
            completion_response="response",
            current_instructions="Do X.",
            current_parameters={},
            current_variables={},
            iteration=iteration,
            duration_ms=duration_ms,
        )

    def test_returns_true_when_history_is_empty(self):
        self.client._history = []
        assert self.client._evaluate_duration(self._ctx(5000)) is True

    def test_returns_true_when_baseline_duration_is_none(self):
        self.client._history = [self._ctx(None, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(5000, iteration=2)) is True

    def test_returns_true_when_candidate_duration_is_none(self):
        self.client._history = [self._ctx(2000, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(None, iteration=2)) is True

    def test_passes_when_candidate_is_more_than_20_percent_faster(self):
        # baseline=2000ms, threshold=1600ms, candidate=1500ms → 1500 < 1600 → pass
        self.client._history = [self._ctx(2000, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(1500, iteration=2)) is True

    def test_fails_when_candidate_is_exactly_at_threshold(self):
        # baseline=2000ms, threshold=1600ms, candidate=1600ms → not strictly less → fail
        self.client._history = [self._ctx(2000, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(1600, iteration=2)) is False

    def test_fails_when_improvement_is_less_than_20_percent(self):
        # baseline=2000ms, threshold=1600ms, candidate=1800ms → 1800 >= 1600 → fail
        self.client._history = [self._ctx(2000, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(1800, iteration=2)) is False

    def test_fails_when_candidate_matches_baseline(self):
        self.client._history = [self._ctx(2000, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(2000, iteration=2)) is False

    def test_fails_when_candidate_is_slower_than_baseline(self):
        self.client._history = [self._ctx(2000, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(2500, iteration=2)) is False

    def test_uses_history_index_zero_as_baseline_not_last(self):
        # history[0] is 2000ms (baseline), history[-1] is 500ms (fast, but not the baseline)
        first = self._ctx(2000, iteration=1)
        later = self._ctx(500, iteration=2)
        self.client._history = [first, later]
        # candidate=1500ms < 2000 * 0.80 = 1600ms → pass (uses history[0], not history[-1])
        assert self.client._evaluate_duration(self._ctx(1500, iteration=3)) is True

    def test_pass_boundary_just_below_threshold(self):
        # baseline=1000ms, threshold=800ms, candidate=799ms → pass
        self.client._history = [self._ctx(1000, iteration=1)]
        assert self.client._evaluate_duration(self._ctx(799, iteration=2)) is True


# ---------------------------------------------------------------------------
# Duration optimization — chaos mode wiring
# ---------------------------------------------------------------------------


class TestDurationOptimizationChaosMode:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()

    def _duration_judges(self, statement="The response must be fast."):
        return {
            "speed": OptimizationJudge(
                threshold=0.8,
                acceptance_statement=statement,
            )
        }

    def _ctx_with(self, duration_ms, score=1.0, iteration=1):
        return OptimizationContext(
            scores={"speed": JudgeResult(score=score)},
            completion_response="answer",
            current_instructions="Do X.",
            current_parameters={},
            current_variables={"language": "English"},
            iteration=iteration,
            duration_ms=duration_ms,
        )

    async def test_duration_gate_triggers_variation_when_not_fast_enough(self):
        """Judge passes but duration fails threshold → variation generated → second attempt succeeds."""
        client = _make_client(self.mock_ldai)

        # Iter 1: judge fails → history[0].duration_ms = 2000
        # Iter 2: judge passes, duration 1800ms ≥ 2000 * 0.80 = 1600ms → duration fails → variation
        # Iter 3: judge passes, duration 1500ms < 1600ms → passes → validation → success
        execute_side_effects = [
            self._ctx_with(duration_ms=2000, score=0.2, iteration=1),   # iter 1: judge fails
            self._ctx_with(duration_ms=1800, score=1.0, iteration=2),   # iter 2: judge passes, duration fails
            self._ctx_with(duration_ms=1500, score=1.0, iteration=3),   # iter 3: both pass
            self._ctx_with(duration_ms=1500, score=1.0, iteration=4),   # validation
        ]

        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output=VARIATION_RESPONSE))
        opts = _make_options(
            handle_agent_call=handle_agent_call,
            judges=self._duration_judges(),
            max_attempts=5,
        )

        with patch.object(client, "_execute_agent_turn", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = execute_side_effects
            result = await client.optimize_from_options("test-agent", opts)

        assert result.duration_ms == 1500
        # 2 variations generated (after iter 1 judge fail, after iter 2 duration fail)
        assert handle_agent_call.call_count == 2
        assert mock_execute.call_count == 4

    async def test_duration_check_skipped_on_first_iteration_no_baseline(self):
        """First iteration has no history → duration check always skipped → succeeds even if slow."""
        client = _make_client(self.mock_ldai)

        # Iter 1 (no history): judge passes, duration check skipped → validation
        # Validation: judge passes, duration check still uses history[0] = None since nothing appended yet
        execute_side_effects = [
            self._ctx_with(duration_ms=9999, score=1.0, iteration=1),   # iter 1: would fail if checked
            self._ctx_with(duration_ms=9999, score=1.0, iteration=2),   # validation
        ]

        opts = _make_options(
            handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="answer")),
            judges=self._duration_judges(),
            max_attempts=3,
        )

        with patch.object(client, "_execute_agent_turn", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = execute_side_effects
            result = await client.optimize_from_options("test-agent", opts)

        # Succeeds because history is empty and duration check is skipped
        assert result.duration_ms == 9999

    async def test_no_duration_gate_when_acceptance_criteria_has_no_latency_keywords(self):
        """Acceptance statement with no latency keywords → duration gate never applied."""
        client = _make_client(self.mock_ldai)

        # Judge passes on first try; duration would fail if gate were applied (same as baseline)
        # but since acceptance criteria has no latency keywords, it should succeed anyway
        execute_side_effects = [
            self._ctx_with(duration_ms=2000, score=1.0, iteration=1),
            self._ctx_with(duration_ms=2000, score=1.0, iteration=2),   # validation
        ]

        non_latency_judges = {
            "accuracy": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be accurate and complete.",
            )
        }
        opts = _make_options(
            handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="answer")),
            judges=non_latency_judges,
            max_attempts=3,
        )

        with patch.object(client, "_execute_agent_turn", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = execute_side_effects
            # Manually seed history so _evaluate_duration would fire if incorrectly triggered
            client._history = [self._ctx_with(duration_ms=2000, iteration=0)]
            result = await client.optimize_from_options("test-agent", opts)

        assert result is not None

    async def test_evaluate_duration_called_in_validation_phase(self):
        """Duration gate also runs on validation samples, not just the primary turn."""
        client = _make_client(self.mock_ldai)

        # Iter 1: judge fails → history[0].duration_ms = 2000
        # Iter 2: judge passes, duration 1500ms → primary passes
        # Validation sample: judge passes, duration 1800ms ≥ 1600ms → validation fails → variation
        # Iter 3: judge passes, duration 1500ms → primary passes
        # Validation: judge passes, duration 1500ms → validation passes → success
        execute_side_effects = [
            self._ctx_with(duration_ms=2000, score=0.2, iteration=1),   # iter 1: judge fails
            self._ctx_with(duration_ms=1500, score=1.0, iteration=2),   # iter 2: passes
            self._ctx_with(duration_ms=1800, score=1.0, iteration=3),   # validation: duration fails
            self._ctx_with(duration_ms=1500, score=1.0, iteration=4),   # iter 3: passes
            self._ctx_with(duration_ms=1500, score=1.0, iteration=5),   # validation: passes
        ]

        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output=VARIATION_RESPONSE))
        opts = _make_options(
            handle_agent_call=handle_agent_call,
            judges=self._duration_judges(),
            max_attempts=5,
        )

        with patch.object(client, "_execute_agent_turn", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = execute_side_effects
            result = await client.optimize_from_options("test-agent", opts)

        assert result.duration_ms == 1500
        assert mock_execute.call_count == 5


# ---------------------------------------------------------------------------
# Duration optimization — ground truth mode wiring
# ---------------------------------------------------------------------------


class TestDurationOptimizationGroundTruthMode:
    def setup_method(self):
        self.mock_ldai = _make_ldai_client()

    def _duration_judges(self):
        return {
            "speed": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be fast.",
            )
        }

    def _gt_ctx(self, duration_ms, score=1.0, iteration=1, user_input="q"):
        return OptimizationContext(
            scores={"speed": JudgeResult(score=score)},
            completion_response="answer",
            current_instructions="Do X.",
            current_parameters={},
            current_variables={},
            iteration=iteration,
            duration_ms=duration_ms,
            user_input=user_input,
        )

    async def test_duration_gate_applied_per_sample_in_ground_truth_mode(self):
        """In GT mode, the duration check fires per sample, not just once per attempt."""
        client = _make_client(self.mock_ldai)

        # Attempt 1:
        #   Sample 1: judge fails (score 0.2) → all_passed = False
        #   Sample 2: judge passes → duration skipped (history empty for sample 2)
        #   → history extended with attempt 1 results → variation generated
        # Attempt 2:
        #   Sample 1: judge passes, duration 1800ms vs baseline history[0].duration_ms = 2000ms
        #             → 1800 >= 1600 → duration fails → sample_passed = False → all_passed = False
        #   (attempt 2 fails due to duration on sample 1)
        #   → variation generated
        # Attempt 3:
        #   Sample 1: judge passes, duration 1500ms < 1600ms → passes
        #   Sample 2: judge passes, duration 1500ms (history[0] still 2000ms) → passes
        #   → all_passed = True → success
        execute_side_effects = [
            # Attempt 1
            self._gt_ctx(duration_ms=2000, score=0.2, iteration=1, user_input="q1"),
            self._gt_ctx(duration_ms=2000, score=1.0, iteration=2, user_input="q2"),
            # Variation (not from _execute_agent_turn, from handle_agent_call)
            # Attempt 2
            self._gt_ctx(duration_ms=1800, score=1.0, iteration=3, user_input="q1"),
            self._gt_ctx(duration_ms=1800, score=1.0, iteration=4, user_input="q2"),
            # Variation
            # Attempt 3
            self._gt_ctx(duration_ms=1500, score=1.0, iteration=5, user_input="q1"),
            self._gt_ctx(duration_ms=1500, score=1.0, iteration=6, user_input="q2"),
        ]

        handle_agent_call = AsyncMock(return_value=OptimizationResponse(output=VARIATION_RESPONSE))
        opts = _make_gt_options(
            handle_agent_call=handle_agent_call,
            judges=self._duration_judges(),
            max_attempts=5,
        )

        with patch.object(client, "_execute_agent_turn", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = execute_side_effects
            results = await client.optimize_from_ground_truth_options("test-agent", opts)

        assert isinstance(results, list)
        for ctx in results:
            assert ctx.duration_ms == 1500
        # 2 variations generated
        assert handle_agent_call.call_count == 2
        assert mock_execute.call_count == 6

    async def test_no_duration_gate_in_gt_mode_when_no_latency_keywords(self):
        """In GT mode, duration gate is not applied when acceptance criteria has no latency keywords."""
        client = _make_client(self.mock_ldai)

        execute_side_effects = [
            self._gt_ctx(duration_ms=5000, score=1.0, iteration=1, user_input="q1"),
            self._gt_ctx(duration_ms=5000, score=1.0, iteration=2, user_input="q2"),
        ]

        non_latency_judges = {
            "accuracy": OptimizationJudge(
                threshold=0.8,
                acceptance_statement="The response must be accurate.",
            )
        }
        opts = _make_gt_options(
            handle_agent_call=AsyncMock(return_value=OptimizationResponse(output="answer")),
            judges=non_latency_judges,
            max_attempts=3,
        )

        with patch.object(client, "_execute_agent_turn", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = execute_side_effects
            results = await client.optimize_from_ground_truth_options("test-agent", opts)

        # Succeeds on first attempt even with slow duration (no latency keyword → no gate)
        assert isinstance(results, list)
        assert mock_execute.call_count == 2
