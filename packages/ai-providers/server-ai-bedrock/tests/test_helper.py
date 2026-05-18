"""Tests for ldai_bedrock.bedrock_helper."""

import pytest
from ldai import LDMessage

from ldai_bedrock import (
    convert_messages_to_bedrock,
    convert_tools_to_bedrock,
    get_ai_metrics_from_response,
    get_ai_usage_from_response,
    map_provider,
)


class TestMapProvider:
    """Tests for map_provider."""

    def test_maps_bedrock_to_bedrock(self):
        assert map_provider('bedrock') == 'bedrock'

    def test_maps_bedrock_with_family_prefix(self):
        assert map_provider('Bedrock:Anthropic') == 'bedrock'
        assert map_provider('bedrock:amazon') == 'bedrock'

    def test_passes_through_unrelated_provider(self):
        assert map_provider('openai') == 'openai'


class TestConvertMessagesToBedrock:
    """Tests for convert_messages_to_bedrock."""

    def test_splits_system_and_conversation_messages(self):
        result = convert_messages_to_bedrock([
            LDMessage(role='system', content='You are helpful.'),
            LDMessage(role='user', content='Hello!'),
            LDMessage(role='assistant', content='Hi there.'),
        ])
        assert result == {
            'system': [{'text': 'You are helpful.'}],
            'messages': [
                {'role': 'user', 'content': [{'text': 'Hello!'}]},
                {'role': 'assistant', 'content': [{'text': 'Hi there.'}]},
            ],
        }

    def test_omits_system_key_when_no_system_messages(self):
        result = convert_messages_to_bedrock([
            LDMessage(role='user', content='Hello!'),
        ])
        assert 'system' not in result
        assert result['messages'] == [
            {'role': 'user', 'content': [{'text': 'Hello!'}]},
        ]

    def test_aggregates_multiple_system_messages(self):
        result = convert_messages_to_bedrock([
            LDMessage(role='system', content='First.'),
            LDMessage(role='system', content='Second.'),
            LDMessage(role='user', content='Go.'),
        ])
        assert result['system'] == [
            {'text': 'First.'},
            {'text': 'Second.'},
        ]

    def test_raises_on_unsupported_role(self):
        with pytest.raises(ValueError):
            convert_messages_to_bedrock([
                LDMessage(role='function', content='Nope.'),
            ])


class TestConvertToolsToBedrock:
    """Tests for convert_tools_to_bedrock."""

    def test_wraps_tools_in_tool_spec_envelopes(self):
        result = convert_tools_to_bedrock([
            {
                'name': 'get_weather',
                'description': 'Look up the weather for a location.',
                'parameters': {
                    'type': 'object',
                    'properties': {'location': {'type': 'string'}},
                    'required': ['location'],
                },
            },
        ])
        assert result == {
            'tools': [
                {
                    'toolSpec': {
                        'name': 'get_weather',
                        'description': 'Look up the weather for a location.',
                        'inputSchema': {
                            'json': {
                                'type': 'object',
                                'properties': {'location': {'type': 'string'}},
                                'required': ['location'],
                            },
                        },
                    },
                },
            ],
        }

    def test_returns_none_for_empty_tools(self):
        assert convert_tools_to_bedrock([]) is None

    def test_skips_non_dict_and_unnamed_entries(self):
        assert convert_tools_to_bedrock(['not-a-dict', {'description': 'no name'}]) is None


class TestGetAIUsageFromResponse:
    """Tests for get_ai_usage_from_response."""

    def test_extracts_token_counts(self):
        usage = get_ai_usage_from_response({
            'usage': {'totalTokens': 100, 'inputTokens': 60, 'outputTokens': 40},
        })
        assert usage is not None
        assert usage.total == 100
        assert usage.input == 60
        assert usage.output == 40

    def test_returns_none_when_usage_absent(self):
        assert get_ai_usage_from_response({}) is None

    def test_returns_none_when_all_counts_zero(self):
        assert get_ai_usage_from_response({
            'usage': {'totalTokens': 0, 'inputTokens': 0, 'outputTokens': 0},
        }) is None


class TestGetAIMetricsFromResponse:
    """Tests for get_ai_metrics_from_response."""

    def test_success_when_status_200(self):
        result = get_ai_metrics_from_response({
            'ResponseMetadata': {'HTTPStatusCode': 200},
            'usage': {'totalTokens': 5, 'inputTokens': 3, 'outputTokens': 2},
            'metrics': {'latencyMs': 42},
        })
        assert result.success is True
        assert result.tokens is not None
        assert result.tokens.total == 5
        assert result.duration_ms == 42
        assert result.tool_calls is None

    def test_failure_when_status_not_200(self):
        result = get_ai_metrics_from_response({
            'ResponseMetadata': {'HTTPStatusCode': 500},
            'usage': {'totalTokens': 5, 'inputTokens': 3, 'outputTokens': 2},
            'metrics': {'latencyMs': 42},
        })
        assert result.success is False
        assert result.tokens is not None
        assert result.duration_ms == 42

    def test_records_observed_tool_call_names(self):
        result = get_ai_metrics_from_response({
            'ResponseMetadata': {'HTTPStatusCode': 200},
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'toolUse': {'name': 'get_weather', 'input': {}}},
                        {'text': 'Looking that up.'},
                    ],
                },
            },
        })
        assert result.success is True
        assert result.tool_calls == ['get_weather']

    def test_handles_missing_metadata(self):
        result = get_ai_metrics_from_response({})
        assert result.success is False
        assert result.tokens is None
        assert result.duration_ms is None

    def test_returns_failure_for_non_dict(self):
        result = get_ai_metrics_from_response(None)
        assert result.success is False
