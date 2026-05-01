from time import sleep
from unittest.mock import ANY, MagicMock, call

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
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=1, model_name="fakeModel", provider_name="fakeProvider", context=context)

    assert tracker.get_summary().duration is None
    assert tracker.get_summary().feedback is None
    assert tracker.get_summary().success is None
    assert tracker.get_summary().usage is None


def test_tracks_duration(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_duration(100)

    client.track.assert_called_with(  # type: ignore
        "$ld:ai:duration:total",
        context,
        {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
        100,
    )

    assert tracker.get_summary().duration == 100


def test_tracks_duration_of(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_duration_of(lambda: sleep(0.01))

    calls = client.track.mock_calls  # type: ignore

    assert len(calls) == 1
    assert calls[0].args[0] == "$ld:ai:duration:total"
    assert calls[0].args[1] == context
    assert calls[0].args[2]["variationKey"] == "variation-key"
    assert calls[0].args[2]["configKey"] == "config-key"
    assert calls[0].args[2]["version"] == 3
    assert calls[0].args[2]["modelName"] == "fakeModel"
    assert calls[0].args[2]["providerName"] == "fakeProvider"
    assert "runId" in calls[0].args[2]
    assert calls[0].args[3] == pytest.approx(10, rel=10)


def test_tracks_time_to_first_token(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_time_to_first_token(100)

    client.track.assert_called_with(  # type: ignore
        "$ld:ai:tokens:ttf",
        context,
        {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
        100,
    )

    assert tracker.get_summary().time_to_first_token == 100


def test_tracks_duration_of_with_exception(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)

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
    assert calls[0].args[2]["variationKey"] == "variation-key"
    assert calls[0].args[2]["configKey"] == "config-key"
    assert calls[0].args[2]["version"] == 3
    assert calls[0].args[2]["modelName"] == "fakeModel"
    assert calls[0].args[2]["providerName"] == "fakeProvider"
    assert "runId" in calls[0].args[2]
    assert calls[0].args[3] == pytest.approx(10, rel=10)


def test_tracks_token_usage(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)

    tokens = TokenUsage(300, 200, 100)
    tracker.track_tokens(tokens)

    calls = [
        call(
            "$ld:ai:tokens:total",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            300,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            200,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            100,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().usage == tokens


def test_tracks_bedrock_metrics(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)

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
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
        call(
            "$ld:ai:duration:total",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            50,
        ),
        call(
            "$ld:ai:tokens:total",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            330,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            220,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            110,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is True
    assert tracker.get_summary().duration == 50
    assert tracker.get_summary().usage == TokenUsage(330, 220, 110)


def test_tracks_bedrock_metrics_with_error(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)

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
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
        call(
            "$ld:ai:duration:total",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            50,
        ),
        call(
            "$ld:ai:tokens:total",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            330,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            220,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            110,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is False
    assert tracker.get_summary().duration == 50
    assert tracker.get_summary().usage == TokenUsage(330, 220, 110)


def test_tracks_openai_metrics(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)

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

    with pytest.warns(DeprecationWarning, match="track_openai_metrics is deprecated"):
        tracker.track_openai_metrics(get_result)

    calls = [
        call(
            "$ld:ai:generation:success",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
        call(
            "$ld:ai:tokens:total",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            330,
        ),
        call(
            "$ld:ai:tokens:input",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            220,
        ),
        call(
            "$ld:ai:tokens:output",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            110,
        ),
    ]

    client.track.assert_has_calls(calls, any_order=False)  # type: ignore

    assert tracker.get_summary().usage == TokenUsage(330, 220, 110)


def test_tracks_openai_metrics_with_exception(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)

    def raise_exception():
        raise ValueError("Something went wrong")

    with pytest.warns(DeprecationWarning, match="track_openai_metrics is deprecated"):
        try:
            tracker.track_openai_metrics(raise_exception)
            assert False, "Should have thrown an exception"
        except ValueError:
            pass

    calls = [
        call(
            "$ld:ai:generation:error",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
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
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)

    tracker.track_feedback({"kind": kind})

    client.track.assert_called_with(  # type: ignore
        f"$ld:ai:feedback:user:{label}",
        context,
        {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
        1,
    )
    assert tracker.get_summary().feedback == {"kind": kind}


def test_tracks_success(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_success()

    calls = [
        call(
            "$ld:ai:generation:success",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is True


def test_tracks_error(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_error()

    calls = [
        call(
            "$ld:ai:generation:error",
            context,
            {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
            1,
        ),
    ]

    client.track.assert_has_calls(calls)  # type: ignore

    assert tracker.get_summary().success is False


def test_error_after_success_is_blocked(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_success()
    tracker.track_error()

    # Only the first call (success) should go through; error is blocked by at-most-once guard
    client.track.assert_called_once_with(  # type: ignore
        "$ld:ai:generation:success",
        context,
        {"runId": ANY, "variationKey": "variation-key", "configKey": "config-key", "version": 3, "modelName": "fakeModel", "providerName": "fakeProvider"},
        1,
    )

    assert tracker.get_summary().success is True


def _base_td() -> dict:
    return {
        "runId": ANY,
        "variationKey": "variation-key",
        "configKey": "config-key",
        "version": 3,
        "modelName": "fakeModel",
        "providerName": "fakeProvider",
    }


def test_config_tracker_includes_graph_key_when_provided(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context, graph_key="my-graph"
    )
    expected = {**_base_td(), "graphKey": "my-graph"}
    tracker.track_success()
    client.track.assert_called_with("$ld:ai:generation:success", context, expected, 1)  # type: ignore


def test_config_tracker_track_tokens_with_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context, graph_key="g1"
    )
    tokens = TokenUsage(10, 4, 6)
    expected = {**_base_td(), "graphKey": "g1"}
    tracker.track_tokens(tokens)
    client.track.assert_any_call("$ld:ai:tokens:total", context, expected, 10)  # type: ignore


def test_config_tracker_track_feedback_with_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context, graph_key="gx"
    )
    expected = {**_base_td(), "graphKey": "gx"}
    tracker.track_feedback({"kind": FeedbackKind.Positive})
    client.track.assert_called_with(
        "$ld:ai:feedback:user:positive", context, expected, 1
    )  # type: ignore


def test_config_tracker_track_tool_call(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key",
        version=3, model_name="fakeModel", provider_name="fakeProvider", context=context,
    )
    expected = {**_base_td(), "toolKey": "search"}
    tracker.track_tool_call("search")
    client.track.assert_called_with("$ld:ai:tool_call", context, expected, 1)  # type: ignore


def test_config_tracker_track_tool_call_with_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context, graph_key="my-graph"
    )
    expected = {**_base_td(), "graphKey": "my-graph", "toolKey": "calc"}
    tracker.track_tool_call("calc")
    client.track.assert_called_with("$ld:ai:tool_call", context, expected, 1)  # type: ignore


def test_config_tracker_track_tool_calls(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context, graph_key="g"
    )
    tracker.track_tool_calls(["a", "b"])
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
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key",
        version=3, model_name="fakeModel", provider_name="fakeProvider", context=context,
    )

    def fn():
        return "done"

    def extract(r):
        return LDAIMetrics(success=True, usage=TokenUsage(5, 2, 3))

    out = tracker.track_metrics_of(extract, fn)
    assert out == "done"
    calls = client.track.mock_calls  # type: ignore
    assert any(c.args[0] == "$ld:ai:generation:success" for c in calls)
    assert any(c.args[0] == "$ld:ai:tokens:total" and c.args[3] == 5 for c in calls)


@pytest.mark.asyncio
async def test_config_tracker_track_metrics_of_async_passes_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context, graph_key="gg"
    )

    async def fn():
        return "ok"

    def extract(r):
        return LDAIMetrics(success=True, usage=TokenUsage(5, 2, 3))

    await tracker.track_metrics_of_async(extract, fn)
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


# --- At-most-once guard tests ---


def test_duplicate_track_duration_is_ignored(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_duration(100)
    tracker.track_duration(200)

    assert client.track.call_count == 1  # type: ignore
    assert tracker.get_summary().duration == 100


def test_duplicate_track_time_to_first_token_is_ignored(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_time_to_first_token(50)
    tracker.track_time_to_first_token(75)

    assert client.track.call_count == 1  # type: ignore
    assert tracker.get_summary().time_to_first_token == 50


def test_duplicate_track_tokens_is_ignored(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tokens1 = TokenUsage(300, 200, 100)
    tokens2 = TokenUsage(600, 400, 200)
    tracker.track_tokens(tokens1)
    tracker.track_tokens(tokens2)

    # 3 track calls for total/input/output from the first call only
    assert client.track.call_count == 3  # type: ignore
    assert tracker.get_summary().usage == tokens1


def test_duplicate_track_success_is_ignored(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_success()
    tracker.track_success()

    assert client.track.call_count == 1  # type: ignore
    assert tracker.get_summary().success is True


def test_duplicate_track_error_is_ignored(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_error()
    tracker.track_error()

    assert client.track.call_count == 1  # type: ignore
    assert tracker.get_summary().success is False


def test_duplicate_track_feedback_is_ignored(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_feedback({"kind": FeedbackKind.Positive})
    tracker.track_feedback({"kind": FeedbackKind.Negative})

    assert client.track.call_count == 1  # type: ignore
    assert tracker.get_summary().feedback == {"kind": FeedbackKind.Positive}


def test_track_data_includes_run_id(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="my-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_success()

    track_data = client.track.call_args[0][2]  # type: ignore
    assert track_data["runId"] == "my-run-id"


def test_run_id_is_consistent_across_track_calls(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="config-key", variation_key="variation-key", version=3, model_name="fakeModel", provider_name="fakeProvider", context=context)
    tracker.track_success()
    tracker.track_duration(100)

    calls = client.track.mock_calls  # type: ignore
    run_id_1 = calls[0].args[2]["runId"]
    run_id_2 = calls[1].args[2]["runId"]
    assert run_id_1 == run_id_2


# --- Resumption token tests ---


def test_resumption_token_round_trip(client: LDClient):
    import base64
    import json

    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="cfg-key", variation_key="var-key", version=5, model_name="gpt-4", provider_name="openai", context=context)

    token = tracker.resumption_token
    # Token has no padding — add it back before decoding
    padded = token + "=" * (-len(token) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))

    assert decoded["runId"] == tracker._run_id
    assert decoded["configKey"] == "cfg-key"
    assert decoded["variationKey"] == "var-key"
    assert decoded["version"] == 5
    # modelName and providerName should NOT be in the token
    assert "modelName" not in decoded
    assert "providerName" not in decoded


def test_resumption_token_has_no_padding(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="cfg-key", variation_key="var-key", version=1, model_name="model", provider_name="provider", context=context)

    token = tracker.resumption_token
    assert "=" not in token


def test_resumption_token_is_url_safe_base64(client: LDClient):
    import base64

    context = Context.create("user-key")
    tracker = LDAIConfigTracker(ld_client=client, run_id="test-run-id", config_key="cfg-key", variation_key="var-key", version=1, model_name="model", provider_name="provider", context=context)

    token = tracker.resumption_token
    # Should decode without error using urlsafe variant (with padding restored)
    padded = token + "=" * (-len(token) % 4)
    base64.urlsafe_b64decode(padded.encode("utf-8"))


def test_resumption_token_omits_variation_key_when_empty(client: LDClient):
    import base64
    import json

    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="cfg-key",
        variation_key="", version=1, context=context,
        model_name="model", provider_name="provider",
    )

    token = tracker.resumption_token
    padded = token + "=" * (-len(token) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))

    assert "variationKey" not in decoded
    assert decoded["runId"] == "test-run-id"
    assert decoded["configKey"] == "cfg-key"
    assert decoded["version"] == 1


def test_resumption_token_includes_graph_key_when_set(client: LDClient):
    import base64
    import json

    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="cfg-key",
        variation_key="var-key", version=2, context=context,
        model_name="model", provider_name="provider", graph_key="my-graph",
    )

    token = tracker.resumption_token
    padded = token + "=" * (-len(token) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))

    assert decoded["runId"] == "test-run-id"
    assert decoded["configKey"] == "cfg-key"
    assert decoded["variationKey"] == "var-key"
    assert decoded["version"] == 2
    assert decoded["graphKey"] == "my-graph"
    # Key order: runId, configKey, variationKey, version, graphKey
    assert list(decoded.keys()) == ["runId", "configKey", "variationKey", "version", "graphKey"]


def test_resumption_token_omits_graph_key_when_not_set(client: LDClient):
    import base64
    import json

    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="cfg-key",
        variation_key="var-key", version=1, context=context,
        model_name="model", provider_name="provider",
    )

    token = tracker.resumption_token
    padded = token + "=" * (-len(token) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))

    assert "graphKey" not in decoded


def test_resumption_token_round_trip_with_graph_key(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="cfg-key",
        variation_key="var-key", version=3, context=context,
        model_name="model", provider_name="provider", graph_key="my-graph",
    )

    token = tracker.resumption_token
    result = LDAIConfigTracker.from_resumption_token(token, client, context)
    assert result.is_success()
    restored = result.value

    assert restored._run_id == "test-run-id"
    assert restored._config_key == "cfg-key"
    assert restored._variation_key == "var-key"
    assert restored._version == 3
    assert restored._graph_key == "my-graph"


def test_tracker_with_explicit_run_id(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="custom-run-id-123", config_key="cfg-key",
        variation_key="var-key", version=1, model_name="model",
        provider_name="provider", context=context,
    )
    tracker.track_success()

    track_data = client.track.call_args[0][2]  # type: ignore
    assert track_data["runId"] == "custom-run-id-123"


def test_client_create_tracker_from_resumption_token():
    from unittest.mock import Mock

    from ldai.client import LDAIClient

    mock_client = Mock()
    ai_client = LDAIClient(mock_client)
    context = Context.create("feedback-user")

    # Create an original tracker and get its token
    original = LDAIConfigTracker(
        ld_client=mock_client, run_id="original-run-id-123",
        config_key="my-config", variation_key="var-abc", version=7,
        model_name="gpt-4", provider_name="openai",
        context=Context.create("original-user"),
    )
    token = original.resumption_token

    # Reconstruct from token
    result = ai_client.create_tracker(token, context)
    assert result.is_success()
    restored = result.value

    # The restored tracker should use the same runId
    restored.track_feedback({"kind": FeedbackKind.Positive})

    feedback_calls = [
        c for c in mock_client.track.call_args_list
        if c.args[0] == "$ld:ai:feedback:user:positive"
    ]
    assert len(feedback_calls) == 1
    track_data = feedback_calls[0].args[2]
    assert track_data["runId"] == original._run_id
    assert track_data["configKey"] == "my-config"
    assert track_data["variationKey"] == "var-abc"
    assert track_data["version"] == 7
    # modelName and providerName are empty when reconstructed from token
    assert track_data["modelName"] == ""
    assert track_data["providerName"] == ""
    # Context should be the new one, not the original
    assert feedback_calls[0].args[1] == context


def test_client_create_tracker_fails_on_invalid_base64():
    from unittest.mock import Mock

    from ldai.client import LDAIClient

    mock_client = Mock()
    ai_client = LDAIClient(mock_client)
    context = Context.create("user-key")

    result = ai_client.create_tracker("not-valid-base64!!!", context)
    assert not result.is_success()
    assert "Invalid resumption token" in result.error


def test_client_create_tracker_fails_on_missing_fields():
    import base64
    import json

    from unittest.mock import Mock

    from ldai.client import LDAIClient

    mock_client = Mock()
    ai_client = LDAIClient(mock_client)
    context = Context.create("user-key")

    # Token missing runId
    incomplete = base64.urlsafe_b64encode(
        json.dumps({"configKey": "k", "version": 1}).encode()
    ).rstrip(b"=").decode()

    result = ai_client.create_tracker(incomplete, context)
    assert not result.is_success()
    assert "missing required field 'runId'" in result.error


def test_client_create_tracker_fails_on_invalid_json():
    import base64

    from unittest.mock import Mock

    from ldai.client import LDAIClient

    mock_client = Mock()
    ai_client = LDAIClient(mock_client)
    context = Context.create("user-key")

    bad_token = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()

    result = ai_client.create_tracker(bad_token, context)
    assert not result.is_success()
    assert "Invalid resumption token" in result.error


def test_ldai_metrics_to_dict_includes_tool_calls_and_duration_ms():
    metrics = LDAIMetrics(
        success=True,
        usage=TokenUsage(total=10, input=4, output=6),
        tool_calls=["search", "lookup"],
        duration_ms=123,
    )
    d = metrics.to_dict()
    assert d["success"] is True
    assert d["usage"] == {"total": 10, "input": 4, "output": 6}
    assert d["toolCalls"] == ["search", "lookup"]
    assert d["durationMs"] == 123


def test_ldai_metrics_to_dict_omits_optional_fields_when_none():
    metrics = LDAIMetrics(success=False)
    d = metrics.to_dict()
    assert d == {"success": False}


def test_track_metrics_of_uses_metrics_duration_ms_when_set(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key",
        variation_key="variation-key", version=3, model_name="m",
        provider_name="p", context=context,
    )

    def fn():
        return "done"

    def extract(_r):
        return LDAIMetrics(success=True, duration_ms=999)

    tracker.track_metrics_of(extract, fn)
    assert tracker.get_summary().duration_ms == 999


@pytest.mark.asyncio
async def test_track_metrics_of_async_uses_metrics_duration_ms_when_set(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key",
        variation_key="variation-key", version=3, model_name="m",
        provider_name="p", context=context,
    )

    async def fn():
        return "done"

    def extract(_r):
        return LDAIMetrics(success=True, duration_ms=42)

    await tracker.track_metrics_of_async(extract, fn)
    assert tracker.get_summary().duration_ms == 42


def test_track_metrics_of_calls_track_tool_calls_when_present(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key",
        variation_key="variation-key", version=3, model_name="m",
        provider_name="p", context=context,
    )

    def fn():
        return "done"

    def extract(_r):
        return LDAIMetrics(success=True, tool_calls=["foo", "bar"])

    tracker.track_metrics_of(extract, fn)
    summary = tracker.get_summary()
    assert summary.tool_calls == ["foo", "bar"]
    # One $ld:ai:tool_call event per tool key.
    tool_call_events = [
        c for c in client.track.mock_calls  # type: ignore
        if c.args[0] == "$ld:ai:tool_call"
    ]
    assert len(tool_call_events) == 2


def test_track_metrics_of_skips_track_tool_calls_when_absent(client: LDClient):
    context = Context.create("user-key")
    tracker = LDAIConfigTracker(
        ld_client=client, run_id="test-run-id", config_key="config-key",
        variation_key="variation-key", version=3, model_name="m",
        provider_name="p", context=context,
    )

    def fn():
        return "done"

    def extract(_r):
        return LDAIMetrics(success=True, usage=None)

    tracker.track_metrics_of(extract, fn)
    assert tracker.get_summary().tool_calls is None
    tool_call_events = [
        c for c in client.track.mock_calls  # type: ignore
        if c.args[0] == "$ld:ai:tool_call"
    ]
    assert tool_call_events == []
