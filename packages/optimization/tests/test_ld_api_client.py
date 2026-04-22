"""Tests for ldai_optimizer.ld_api_client."""

import json
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from ldai_optimizer.ld_api_client import (
    AgentOptimizationConfig,
    AgentOptimizationResultPost as OptimizationResultPayload,
    LDApiClient,
    LDApiError,
    _parse_agent_optimization,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG: Dict[str, Any] = {
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
    "version": 1,
    "createdAt": 1700000000,
}


def _make_config(**overrides: Any) -> Dict[str, Any]:
    return {**_BASE_CONFIG, **overrides}


def _mock_urlopen(response_data: Any, status: int = 200) -> MagicMock:
    """Return a context-manager mock whose .read() returns JSON-encoded response_data."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# _parse_agent_optimization
# ---------------------------------------------------------------------------


class TestParseAgentOptimization:
    def test_valid_config_is_returned_unchanged(self):
        config = _make_config()
        result = _parse_agent_optimization(config)
        assert result["id"] == "opt-uuid-123"
        assert result["aiConfigKey"] == "my-agent"

    def test_optional_fields_not_required(self):
        config = _make_config()
        # groundTruthResponses and metricKey are optional — should not raise
        assert "groundTruthResponses" not in config
        assert "metricKey" not in config
        _parse_agent_optimization(config)  # must not raise

    def test_raises_on_non_dict_input(self):
        with pytest.raises(ValueError, match="Expected a JSON object"):
            _parse_agent_optimization(["not", "a", "dict"])

    def test_raises_on_none_input(self):
        with pytest.raises(ValueError, match="Expected a JSON object"):
            _parse_agent_optimization(None)

    @pytest.mark.parametrize("field", ["id", "key", "aiConfigKey", "judgeModel"])
    def test_raises_on_missing_required_string_field(self, field: str):
        config = _make_config()
        del config[field]
        with pytest.raises(ValueError, match=f"missing required field '{field}'"):
            _parse_agent_optimization(config)

    @pytest.mark.parametrize("field", ["maxAttempts", "version", "createdAt"])
    def test_raises_on_missing_required_int_field(self, field: str):
        config = _make_config()
        del config[field]
        with pytest.raises(ValueError, match=f"missing required field '{field}'"):
            _parse_agent_optimization(config)

    @pytest.mark.parametrize(
        "field",
        ["modelChoices", "variableChoices", "acceptanceStatements", "judges", "userInputOptions"],
    )
    def test_raises_on_missing_required_list_field(self, field: str):
        config = _make_config()
        del config[field]
        with pytest.raises(ValueError, match=f"missing required field '{field}'"):
            _parse_agent_optimization(config)

    def test_raises_on_wrong_type_for_string_field(self):
        config = _make_config(aiConfigKey=123)
        with pytest.raises(ValueError, match="field 'aiConfigKey' must be a string"):
            _parse_agent_optimization(config)

    def test_raises_on_wrong_type_for_int_field(self):
        config = _make_config(maxAttempts="three")
        with pytest.raises(ValueError, match="field 'maxAttempts' must be an integer"):
            _parse_agent_optimization(config)

    def test_raises_on_wrong_type_for_list_field(self):
        config = _make_config(modelChoices="gpt-4o")
        with pytest.raises(ValueError, match="field 'modelChoices' must be a list"):
            _parse_agent_optimization(config)

    def test_raises_when_model_choices_is_empty(self):
        config = _make_config(modelChoices=[])
        with pytest.raises(ValueError, match="at least 1 entry"):
            _parse_agent_optimization(config)

    def test_collects_multiple_errors_in_one_raise(self):
        config = _make_config()
        del config["id"]
        del config["maxAttempts"]
        config["modelChoices"] = "bad"
        with pytest.raises(ValueError) as exc_info:
            _parse_agent_optimization(config)
        msg = str(exc_info.value)
        assert "id" in msg
        assert "maxAttempts" in msg
        assert "modelChoices" in msg


# ---------------------------------------------------------------------------
# LDApiClient._request
# ---------------------------------------------------------------------------


class TestLDApiClientRequest:
    def test_get_does_not_send_content_type(self):
        client = LDApiClient("test-key")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
            client._request("GET", "/some/path")
            req: urllib.request.Request = mock_open.call_args[0][0]
            assert "Content-Type" not in req.headers

    def test_post_sends_content_type(self):
        client = LDApiClient("test-key")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
            client._request("POST", "/some/path", body={"key": "value"})
            req: urllib.request.Request = mock_open.call_args[0][0]
            assert req.get_header("Content-type") == "application/json"

    def test_authorization_header_always_sent(self):
        client = LDApiClient("my-api-key")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
            client._request("GET", "/path")
            req: urllib.request.Request = mock_open.call_args[0][0]
            assert req.get_header("Authorization") == "my-api-key"

    def test_raises_ld_api_error_on_http_error(self):
        client = LDApiClient("test-key")
        http_error = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=BytesIO(b"not found body")
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(LDApiError) as exc_info:
                client._request("GET", "/missing")
        assert exc_info.value.status_code == 404
        assert "404" in str(exc_info.value)

    def test_raises_ld_api_error_on_url_error(self):
        client = LDApiClient("test-key")
        url_error = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=url_error):
            with pytest.raises(LDApiError) as exc_info:
                client._request("GET", "/path")
        assert exc_info.value.status_code is None
        assert "Connection refused" in str(exc_info.value)

    def test_401_error_includes_api_key_hint(self):
        client = LDApiClient("test-key")
        http_error = urllib.error.HTTPError(
            url="http://x", code=401, msg="Unauthorized", hdrs=MagicMock(), fp=BytesIO(b"")
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(LDApiError, match="LAUNCHDARKLY_API_KEY"):
                client._request("GET", "/path")

    def test_404_error_includes_key_hint(self):
        client = LDApiClient("test-key")
        http_error = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=BytesIO(b"")
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(LDApiError, match="project key"):
                client._request("GET", "/path")

    def test_custom_base_url_used_in_request(self):
        client = LDApiClient("test-key", base_url="https://staging.launchdarkly.com")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
            client._request("GET", "/api/v2/test")
            req: urllib.request.Request = mock_open.call_args[0][0]
            assert req.full_url.startswith("https://staging.launchdarkly.com")

    def test_trailing_slash_stripped_from_base_url(self):
        client = LDApiClient("test-key", base_url="https://app.launchdarkly.com/")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
            client._request("GET", "/api/v2/test")
            req: urllib.request.Request = mock_open.call_args[0][0]
            assert "//" not in req.full_url.replace("https://", "")


# ---------------------------------------------------------------------------
# LDApiClient.get_agent_optimization
# ---------------------------------------------------------------------------


class TestGetAgentOptimization:
    def test_requests_correct_path(self):
        client = LDApiClient("test-key")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_make_config())) as mock_open:
            client.get_agent_optimization("my-project", "my-opt-key")
            req: urllib.request.Request = mock_open.call_args[0][0]
            assert "/api/v2/projects/my-project/agent-optimizations/my-opt-key" in req.full_url

    def test_returns_validated_config(self):
        client = LDApiClient("test-key")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_make_config())):
            result = client.get_agent_optimization("proj", "opt")
        assert result["aiConfigKey"] == "my-agent"
        assert result["maxAttempts"] == 3

    def test_raises_on_invalid_response(self):
        client = LDApiClient("test-key")
        bad_response = {"id": "x"}  # missing many required fields
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(bad_response)):
            with pytest.raises(ValueError, match="Invalid AgentOptimization response"):
                client.get_agent_optimization("proj", "opt")

    def test_raises_ld_api_error_on_http_404(self):
        client = LDApiClient("test-key")
        http_error = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=BytesIO(b"not found")
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(LDApiError) as exc_info:
                client.get_agent_optimization("proj", "missing-key")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# LDApiClient.post_agent_optimization_result
# ---------------------------------------------------------------------------


class TestPostAgentOptimizationResult:
    def _make_payload(self) -> OptimizationResultPayload:
        return {
            "run_id": "run-abc",
            "config_optimization_version": 1,
            "status": "RUNNING",
            "activity": "GENERATING",
            "iteration": 1,
            "instructions": "You are a helpful assistant.",
            "parameters": {"temperature": 0.7},
            "completion_response": "The answer is 4.",
            "scores": {},
        }

    def test_requests_correct_path(self):
        client = LDApiClient("test-key")
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
            client.post_agent_optimization_result("my-project", "opt-uuid", self._make_payload())
            req: urllib.request.Request = mock_open.call_args[0][0]
            assert "/api/v2/projects/my-project/agent-optimizations/opt-uuid/results" in req.full_url

    def test_sends_payload_as_json_body(self):
        client = LDApiClient("test-key")
        payload = self._make_payload()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
            client.post_agent_optimization_result("proj", "opt-id", payload)
            req: urllib.request.Request = mock_open.call_args[0][0]
            sent = json.loads(req.data.decode())
            assert sent["run_id"] == "run-abc"
            assert sent["status"] == "RUNNING"
            assert sent["instructions"] == "You are a helpful assistant."

    def test_swallows_http_errors_without_raising(self):
        client = LDApiClient("test-key")
        http_error = urllib.error.HTTPError(
            url="http://x", code=500, msg="Server Error", hdrs=MagicMock(), fp=BytesIO(b"err")
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            # must not raise
            client.post_agent_optimization_result("proj", "opt-id", self._make_payload())

    def test_swallows_url_errors_without_raising(self):
        client = LDApiClient("test-key")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            client.post_agent_optimization_result("proj", "opt-id", self._make_payload())
