from unittest.mock import MagicMock

import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai.tracker import FeedbackKind, LDAIConfigTracker


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    td.update(
        td.flag('model-config')
        .variations(
            {
                'model': {'name': 'fakeModel', 'parameters': {'temperature': 0.5, 'maxTokens': 4096}, 'custom': {'extra-attribute': 'value'}},
                'provider': {'name': 'fakeProvider'},
                'messages': [{'role': 'system', 'content': 'Hello, {{name}}!'}],
                '_ldMeta': {'enabled': True, 'variationKey': 'abcd'},
            },
            "green",
        )
        .variation_for_all(0)
    )

    return td


@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config('sdk-key', update_processor_class=td, send_events=False)
    client = LDClient(config=config)
    client.track = MagicMock()  # type: ignore
    return client


def test_summary_starts_empty(client: LDClient):
    context = Context.create('user-key')
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", context)

    assert tracker.get_summary().duration is None
    assert tracker.get_summary().feedback is None
    assert tracker.get_summary().success is None
    assert tracker.get_summary().usage is None


def test_tracks_duration(client: LDClient):
    context = Context.create('user-key')
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", context)
    tracker.track_duration(100)

    client.track.assert_called_with(  # type: ignore
        '$ld:ai:duration:total',
        context,
        {'variationKey': 'variation-key', 'configKey': 'config-key'},
        100
    )

    assert tracker.get_summary().duration == 100


@pytest.mark.parametrize(
    "kind,label",
    [
        pytest.param(FeedbackKind.Positive, "positive", id="positive"),
        pytest.param(FeedbackKind.Negative, "negative", id="negative"),
    ],
)
def test_tracks_feedback(client: LDClient, kind: FeedbackKind, label: str):
    context = Context.create('user-key')
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", context)

    tracker.track_feedback({'kind': kind})

    client.track.assert_called_with(  # type: ignore
        f'$ld:ai:feedback:user:{label}',
        context,
        {'variationKey': 'variation-key', 'configKey': 'config-key'},
        1
    )
    assert tracker.get_summary().feedback == {'kind': kind}


def test_tracks_success(client: LDClient):
    context = Context.create('user-key')
    tracker = LDAIConfigTracker(client, "variation-key", "config-key", context)
    tracker.track_success()

    client.track.assert_called_with(  # type: ignore
        '$ld:ai:generation',
        context,
        {'variationKey': 'variation-key', 'configKey': 'config-key'},
        1
    )

    assert tracker.get_summary().success is True
