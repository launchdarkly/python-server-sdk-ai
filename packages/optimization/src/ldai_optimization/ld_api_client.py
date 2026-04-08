"""Internal LaunchDarkly REST API client for the optimization package."""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

_BASE_URL = "https://app.launchdarkly.com"


class LDApiError(Exception):
    """Raised when the LaunchDarkly REST API returns an error or is unreachable.

    Attributes:
        status_code: HTTP status code, or None for network-level failures.
        path: The API path that was requested.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, path: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.path = path


_HTTP_ERROR_HINTS: Dict[int, str] = {
    401: "Authentication failed — check that LAUNCHDARKLY_API_KEY is set correctly.",
    403: "Authorization failed — check that your API key has the required permissions.",
    404: "Resource not found — check that the project key and optimization config key are correct.",
    429: "Rate limit exceeded — too many requests to the LaunchDarkly API.",
}

_REQUIRED_STRING_FIELDS = ("id", "key", "aiConfigKey", "judgeModel")
_REQUIRED_INT_FIELDS = ("maxAttempts", "version", "createdAt")
_REQUIRED_LIST_FIELDS = (
    "modelChoices",
    "variableChoices",
    "acceptanceStatements",
    "judges",
    "userInputOptions",
)


# ---------------------------------------------------------------------------
# API response shapes
# ---------------------------------------------------------------------------

class _AcceptanceStatement(TypedDict):
    statement: str
    threshold: float


class _AgentOptimizationJudge(TypedDict):
    key: str
    threshold: float


class _AgentOptimizationConfigRequired(TypedDict):
    id: str
    key: str
    aiConfigKey: str
    maxAttempts: int
    modelChoices: List[str]
    judgeModel: str
    variableChoices: List[Dict[str, Any]]
    acceptanceStatements: List[_AcceptanceStatement]
    judges: List[_AgentOptimizationJudge]
    userInputOptions: List[str]
    version: int
    createdAt: int


class AgentOptimizationConfig(_AgentOptimizationConfigRequired, total=False):
    """Typed representation of the AgentOptimization API response."""

    groundTruthResponses: List[str]
    metricKey: str


# ---------------------------------------------------------------------------
# POST payload shape
# ---------------------------------------------------------------------------

class _OptimizationResultPayloadRequired(TypedDict):
    run_id: str
    config_optimization_version: int
    status: str
    activity: str
    iteration: int
    instructions: str
    parameters: Dict[str, Any]
    completion_response: str
    scores: Dict[str, Any]


class OptimizationResultPayload(_OptimizationResultPayloadRequired, total=False):
    """Typed payload for a single agent_optimization_result POST request.

    Required fields are always sent. Optional fields are omitted when not
    available.

    created_variation_key is only present on the final result record of a
    successful run, populated once a winning variation is committed to LD.
    """

    user_input: Optional[str]
    created_variation_key: str
    generation_latency: float
    generation_tokens: Dict[str, int]
    evaluation_latencies: Dict[str, float]
    evaluation_tokens: Dict[str, Dict[str, int]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _parse_agent_optimization(data: Any) -> AgentOptimizationConfig:
    """Validate and cast a raw API response dict to AgentOptimizationConfig.

    :param data: Parsed JSON response from the GET endpoint.
    :return: The same dict narrowed to AgentOptimizationConfig.
    :raises ValueError: If required fields are missing or have wrong types.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object from AgentOptimization API, got {type(data).__name__}"
        )

    errors: List[str] = []

    for field in _REQUIRED_STRING_FIELDS:
        if field not in data:
            errors.append(f"missing required field '{field}'")
        elif not isinstance(data[field], str):
            errors.append(
                f"field '{field}' must be a string, got {type(data[field]).__name__}"
            )

    for field in _REQUIRED_INT_FIELDS:
        if field not in data:
            errors.append(f"missing required field '{field}'")
        elif not isinstance(data[field], int):
            errors.append(
                f"field '{field}' must be an integer, got {type(data[field]).__name__}"
            )

    for field in _REQUIRED_LIST_FIELDS:
        if field not in data:
            errors.append(f"missing required field '{field}'")
        elif not isinstance(data[field], list):
            errors.append(
                f"field '{field}' must be a list, got {type(data[field]).__name__}"
            )

    if not errors and "modelChoices" in data and isinstance(data["modelChoices"], list):
        if len(data["modelChoices"]) < 1:
            errors.append("field 'modelChoices' must have at least 1 entry")

    if errors:
        raise ValueError(
            f"Invalid AgentOptimization response: {'; '.join(errors)}"
        )

    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LDApiClient:
    """Thin wrapper around the LaunchDarkly REST API for agent-optimization endpoints."""

    def __init__(self, api_key: str, base_url: str = _BASE_URL) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": self._api_key}

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = self._auth_headers()
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            body_excerpt = exc.read(500).decode(errors="replace")
            hint = _HTTP_ERROR_HINTS.get(exc.code, "")
            detail = f"{hint} (API response: {body_excerpt})" if hint else f"API response: {body_excerpt}"
            raise LDApiError(
                f"LaunchDarkly API error {exc.code} {exc.msg} for {method} {path}. {detail}",
                status_code=exc.code,
                path=path,
            ) from exc
        except urllib.error.URLError as exc:
            raise LDApiError(
                f"Could not reach LaunchDarkly API at {url}: {exc.reason}. "
                "Check your network connection and the base_url setting.",
                path=path,
            ) from exc

    def get_agent_optimization(
        self, project_key: str, optimization_key: str
    ) -> AgentOptimizationConfig:
        """Fetch and validate a single agent optimization config by key.

        :param project_key: LaunchDarkly project key.
        :param optimization_key: Key of the agent optimization config.
        :return: Validated AgentOptimizationConfig.
        :raises LDApiError: On non-200 HTTP responses or network errors.
        :raises ValueError: If the response is missing required fields.
        """
        path = f"/api/v2/projects/{project_key}/agent-optimizations/{optimization_key}"
        raw = self._request("GET", path)
        return _parse_agent_optimization(raw)

    def post_agent_optimization_result(
        self, project_key: str, optimization_id: str, payload: OptimizationResultPayload
    ) -> None:
        """Persist an iteration result record for the given optimization run.

        Errors are caught and logged rather than raised so that persistence
        failures never abort an in-progress optimization run.

        :param project_key: LaunchDarkly project key.
        :param optimization_id: UUID id of the parent agent_optimization record.
        :param payload: Typed result payload for this iteration.
        """
        path = f"/api/v2/projects/{project_key}/agent-optimizations/{optimization_id}/results"
        try:
            self._request("POST", path, body=payload)
        except LDApiError as exc:
            logger.debug(
                "Failed to persist optimization result (optimization_id=%s, iteration=%s): %s",
                optimization_id,
                payload.get("iteration"),
                exc,
            )
        except Exception as exc:
            logger.debug(
                "Unexpected error persisting optimization result (optimization_id=%s, iteration=%s): %s",
                optimization_id,
                payload.get("iteration"),
                exc,
            )
