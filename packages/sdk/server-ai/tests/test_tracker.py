from time import sleep
from unittest.mock import MagicMock, call

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai.providers.types import LDAIMetrics
from ldai.tracker import AIGraphTracker, FeedbackKind, LDAIConfigTracker, TokenUsage


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag("model-config")
        .variations(
            {
                "model": {
                    "name": "fakeModel",
                    "parameters": {"temperature": 0.5, "maxTokens": 4096},
                    "custom": {"extra-attribute": "value"},
                },
                "provider": {"name": "fakeProvider"},
                "messages": [{"role": "system", "content": "Hello, {{name}}!"}],
                "_ldMeta": {"enabled": True, "variationKey": "abcd", "version": 1},
            },
            "green",
        )
        .variation_for_all(0)
    )

    return td


@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config("sdk-key", update_processor_class=td, send_events=False)
    client = LDClient(config=config)
    client.track = MagicMock()  # type: ignore
    return client


def test_summary_starts_empty(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 1, "fakeModel", "fakeProvider", context)

    assert tracker.get_summary().duration is None
    assert tracker.get_summary().feedback is None
    assert tracker.get_summary().success is None
    assert tracker.get_summary().usage is None


def test_tracks_duration(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)
    tracker.track_duration(100)

    client.track.assert_called_with(  # type: ignore
        "$ld:ai:duration:total",
        context,
        {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
        100,
    )

    assert tracker.get_summary().duration == 100


def test_tracks_duration_of(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)
    tracker.track_duration_of(lambda: sleep(0.01))

    calls = client.track.mock_calls  # type: ignore

    assert len(calls) == 1
    assert calls[0].args[0] == "$ld:ai:duration:total"
    assert calls[0].args[1] == context
    assert calls[0].args[2] == {
        "variationKey": "variation-key",
        "configKey": "config-key",
        "version": 3,
        "modelName": "fakeModel",
        "providerName": "fakeProvider",
    }
    assert calls[0].args[3] == pytest.approx(10, rel=10)


def test_tracks_time_to_first_token(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)
    tracker.track_time_to_first_token(100)

    client.track.assert_called_with(  # type: ignore
        "$ld:ai:tokens:ttf",
        context,
        {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
        100,
    )

    assert tracker.get_summary().time_to_first_token == 100


def test_tracks_duration_of_with_exception(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)

    def sleep_and_throw():
        sleep(0.01)
        raise ValueError("Something went wrong")

    try:
        tracker.track_duration_of(sleep_and_throw)
        assert False, "Should have thrown an exception"
    except ValueError:
        pass

    calls = client.track.mock_calls  # type: ignore

    assert len(calls) == 1
    assert calls[0].args[0] == "$ld:ai:duration:total"
    assert calls[0].args[1] == context
    assert calls[0].args[2] == {
        "variationKey": "variation-key",
        "configKey": "config-key",
        "version": 3,
        "modelName": "fakeModel",
        "providerName": "fakeProvider",
    }
    assert calls[0].args[3] == pytest.approx(10, rel=10)


def test_tracks_token_usage(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)

    tokens = TokenUsage(300, 200, 100)
    tracker.track_tokens(tokens)

    calls = [
        call(
            "$ld:ai:tokens:total",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            300,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            200,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            100,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().usage == tokens


def test_tracks_bedrock_metrics(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)

    bedrock_result = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "usage": {
            "inputTokens": 220,
            "outputTokens": 110,
            "totalTokens": 330,
        },
        "metrics": {
            "latencyMs": 50,
        },
    }
    tracker.track_bedrock_converse_metrics(bedrock_result)

    calls = [
        call(
            "$ld:ai:generation:success",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
        call(
            "$ld:ai:duration:total",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            50,
        ),
        call(
            "$ld:ai:tokens:total",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            330,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            220,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            110,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is True
    assert tracker.get_summary().duration == 50
    assert tracker.get_summary().usage == TokenUsage(330, 220, 110)


def test_tracks_bedrock_metrics_with_error(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)

    bedrock_result = {
        "ResponseMetadata": {"HTTPStatusCode": 500},
        "usage": {
            "totalTokens": 330,
            "inputTokens": 220,
            "outputTokens": 110,
        },
        "metrics": {
            "latencyMs": 50,
        },
    }
    tracker.track_bedrock_converse_metrics(bedrock_result)

    calls = [
        call(
            "$ld:ai:generation:error",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
        call(
            "$ld:ai:duration:total",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            50,
        ),
        call(
            "$ld:ai:tokens:total",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            330,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            220,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            110,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is False
    assert tracker.get_summary().duration == 50
    assert tracker.get_summary().usage == TokenUsage(330, 220, 110)


def test_tracks_openai_metrics(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)

    class Result:
        def __init__(self):
            self.usage = Usage()

    class Usage:
        def to_dict(self):
            return {
                "total_tokens": 330,
                "prompt_tokens": 220,
                "completion_tokens": 110,
            }

    def get_result():
        return Result()

    tracker.track_openai_metrics(get_result)

    calls = [
        call(
            "$ld:ai:generation:success",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
        call(
            "$ld:ai:tokens:total",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            330,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            220,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            110,
        ),
    ]

    client.track.assert_has_calls(calls, any_order=False)  # type: ignore

    assert tracker.get_summary().usage == TokenUsage(330, 220, 110)


def test_tracks_openai_metrics_with_exception(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)

    def raise_exception():
        raise ValueError("Something went wrong")

    try:
        tracker.track_openai_metrics(raise_exception)
        assert False, "Should have thrown an exception"
    except ValueError:
        pass

    calls = [
        call(
            "$ld:ai:generation:error",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
    ]

    client.track.assert_has_calls(calls, any_order=False)  # type: ignore

    assert tracker.get_summary().usage is None


@pytest.mark.parametrize(
    "kind,label",
    [
        pytest.param(FeedbackKind.Positive, "positive", id="positive"),
        pytest.param(FeedbackKind.Negative, "negative", id="negative"),
    ],
)
def test_tracks_feedback(client: LDClient, kind: FeedbackKind, label: str):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)

    tracker.track_feedback({"kind": kind})

    client.track.assert_called_with(  # type: ignore
        f"$ld:ai:feedback:user:{label}",
        context,
        {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
        1,
    )
    assert tracker.get_summary().feedback == {"kind": kind}


def test_tracks_success(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)
    tracker.track_success()

    calls = [
        call(
            "$ld:ai:generation:success",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is True


def test_tracks_error(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)
    tracker.track_error()

    calls = [
        call(
            "$ld:ai:generation:error",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is False


def test_error_overwrites_success(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context)
    tracker.track_success()
    tracker.track_error()

    calls = [
        call(
            "$ld:ai:generation:success",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
        call(
            "$ld:ai:generation:error",
            context,
            {"variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is False


def _base_td() -> dict:
    return {
        "variationKey": "variation-key",
        "configKey": "config-key",
        "version": 3,
        "modelName": "fakeModel",
        "providerName": "fakeProvider",
    }


def test_config_tracker_includes_graph_key_when_provided(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )
    expected = {**_base_td(), "graphKey": "my-graph"}
    tracker.track_success(graph_key="my-graph")
    client.track.assert_called_with("$ld:ai:generation:success", context, expected, 1)  # type: ignore


def test_config_tracker_track_tokens_with_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )
    tokens = TokenUsage(10, 4, 6)
    expected = {**_base_td(), "graphKey": "g1"}
    tracker.track_tokens(tokens, graph_key="g1")
    client.track.assert_any_call("$ld:ai:tokens:total", context, expected, 10)  # type: ignore


def test_config_tracker_track_feedback_with_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )
    expected = {**_base_td(), "graphKey": "gx"}
    tracker.track_feedback({"kind": FeedbackKind.Positive}, graph_key="gx")
    client.track.assert_called_with(
        "$ld:ai:feedback:user:positive", context, expected, 1
    )  # type: ignore


def test_config_tracker_track_tool_call(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )
    expected = {**_base_td(), "toolKey": "search"}
    tracker.track_tool_call("search")
    client.track.assert_called_with("$ld:ai:tool_call", context, expected, 1)  # type: ignore


def test_config_tracker_track_tool_call_with_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )
    expected = {**_base_td(), "graphKey": "my-graph", "toolKey": "calc"}
    tracker.track_tool_call("calc", graph_key="my-graph")
    client.track.assert_called_with("$ld:ai:tool_call", context, expected, 1)  # type: ignore


def test_config_tracker_track_tool_calls(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )
    tracker.track_tool_calls(["a", "b"], graph_key="g")
    assert client.track.call_count == 2  # type: ignore
    client.track.assert_any_call(
        "$ld:ai:tool_call",
        context,
        {**_base_td(), "graphKey": "g", "toolKey": "a"},
        1,
    )  # type: ignore
    client.track.assert_any_call(
        "$ld:ai:tool_call",
        context,
        {**_base_td(), "graphKey": "g", "toolKey": "b"},
        1,
    )  # type: ignore


def test_config_tracker_track_metrics_of(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )

    def fn():
        return "done"

    def extract(r):
        return LDAIMetrics(success=True, usage=TokenUsage(5, 2, 3))

    out = tracker.track_metrics_of(fn, extract)
    assert out == "done"
    calls = client.track.mock_calls  # type: ignore
    assert any(c.args[0] == "$ld:ai:generation:success" for c in calls)
    assert any(c.args[0] == "$ld:ai:tokens:total" and c.args[3] == 5 for c in calls)


@pytest.mark.asyncio
async def test_config_tracker_track_metrics_of_async_passes_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        client, "variation-key", "config-key", 3, "fakeModel", "fakeProvider", context
    )

    async def fn():
        return "ok"

    def extract(r):
        return LDAIMetrics(success=True, usage=TokenUsage(5, 2, 3))

    await tracker.track_metrics_of_async(fn, extract, graph_key="gg")
    gk_td = {**_base_td(), "graphKey": "gg"}
    calls = client.track.mock_calls  # type: ignore
    assert any(
        c.args[0] == "$ld:ai:generation:success" and c.args[2] == gk_td for c in calls
    )


def test_ai_graph_tracker_graph_key_property(client: LDClient):
    context = Context.create("user-key")
    g = AIGraphTracker(client, "variation-key", "graph-key", 2, context)
    assert g.graph_key == "graph-key"


def test_ai_graph_tracker_track_total_tokens_skips_none_and_nonpositive(client: LDClient):
    context = Context.create("user-key")
    g = AIGraphTracker(client, "variation-key", "graph-key", 2, context)
    g.track_total_tokens(None)
    g.track_total_tokens(TokenUsage(0, 0, 0))
    client.track.assert_not_called()  # type: ignore


def test_ai_graph_tracker_track_total_tokens_tracks_when_positive(client: LDClient):
    context = Context.create("user-key")
    g = AIGraphTracker(client, "variation-key", "graph-key", 2, context)
    g.track_total_tokens(TokenUsage(42, 30, 12))
    client.track.assert_called_with(  # type: ignore
        "$ld:ai:graph:total_tokens",
        context,
        {"variationKey": "variation-key", "graphKey": "graph-key", "version": 2},
        42,
    )
