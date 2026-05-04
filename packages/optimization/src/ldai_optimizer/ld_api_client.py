"""Internal LaunchDarkly REST API client for the optimization package."""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, TypedDict

from ldai_optimizer.util import RedactionFilter

logger = logging.getLogger(__name__)
logger.addFilter(RedactionFilter())

_BASE_URL = "https://app.launchdarkly.com"

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds; doubles on each attempt (1s, 2s, 4s)

# Status codes that warrant a retry.  Everything else (including 400, 401, 403,
# 404) is a permanent or auth failure — retrying would not help and could lead
# to corrupted optimization results if some requests succeed and others fail.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


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
    tokenLimit: int
    variationKey: str


# ---------------------------------------------------------------------------
# Result payload shapes
# ---------------------------------------------------------------------------

class _AgentOptimizationResultPostRequired(TypedDict):
    runId: str
    agentOptimizationVersion: int
    iteration: int
    instructions: str


class AgentOptimizationResultPost(_AgentOptimizationResultPostRequired, total=False):
    """Payload for POST /agent-optimizations/{key}/results — creates a new result record."""

    userInput: str
    parameters: Dict[str, Any]


class AgentOptimizationResultPatch(TypedDict, total=False):
    """Payload for PATCH /agent-optimizations/{key}/results/{id} — updates a result record."""

    status: str
    activity: str
    completionResponse: str
    scores: Dict[str, Any]
    generationLatency: int
    generationTokens: Dict[str, int]
    evaluationLatencies: Dict[str, float]
    evaluationTokens: Dict[str, Dict[str, int]]
    variation: Dict[str, Any]
    createdVariationKey: str


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

    def __repr__(self) -> str:
        return f"LDApiClient(base_url={self._base_url!r})"

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": self._api_key}

    def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Execute an HTTP request with automatic retry and exponential backoff.

        Retries up to ``_MAX_RETRIES`` times for transient errors (429, 5xx,
        network failures) with exponential backoff starting at ``_INITIAL_BACKOFF``
        seconds.  Non-retryable status codes (400, 401, 403, 404, …) are raised
        immediately without retrying.

        :param method: HTTP method (GET, POST, PATCH, …).
        :param path: API path, appended to ``self._base_url``.
        :param body: Optional request body; serialised to JSON.
        :param extra_headers: Additional headers merged with the auth header.
        :raises LDApiError: After all retry attempts are exhausted, or immediately
            for non-retryable status codes.
        """
        url = f"{self._base_url}{path}"
        headers = {**self._auth_headers(), **(extra_headers or {})}
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"

        last_exc: Optional[LDApiError] = None
        for attempt in range(_MAX_RETRIES + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                body_excerpt = exc.read(500).decode(errors="replace")
                hint = _HTTP_ERROR_HINTS.get(exc.code, "")
                detail = f"{hint} (API response: {body_excerpt})" if hint else f"API response: {body_excerpt}"
                api_error = LDApiError(
                    f"LaunchDarkly API error {exc.code} {exc.msg} for {method} {path}. {detail}",
                    status_code=exc.code,
                    path=path,
                )
                if exc.code not in _RETRYABLE_STATUS_CODES:
                    raise api_error from exc
                last_exc = api_error
            except urllib.error.URLError as exc:
                last_exc = LDApiError(
                    f"Could not reach LaunchDarkly API at {url}: {exc.reason}. "
                    "Check your network connection and the base_url setting.",
                    path=path,
                )

            if attempt < _MAX_RETRIES:
                delay = _INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "LaunchDarkly API request failed (attempt %d/%d, path=%s), "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    path,
                    delay,
                    last_exc,
                )
                time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    def get_model_configs(self, project_key: str) -> List[Dict[str, Any]]:
        """Fetch all AI model configs for a project.

        :param project_key: LaunchDarkly project key.
        :return: List of model config dicts (each has at minimum ``id`` and ``key``).
        :raises LDApiError: On non-200 HTTP responses or network errors.
        """
        path = f"/api/v2/projects/{project_key}/ai-configs/model-configs"
        result = self._request("GET", path, extra_headers={"LD-API-Version": "beta"})
        return result if isinstance(result, list) else []

    def get_ai_config(self, project_key: str, config_key: str) -> Any:
        """Fetch a single AI Config by key, including its variations.

        :param project_key: LaunchDarkly project key.
        :param config_key: Key of the AI Config (aiConfigKey).
        :return: Raw AI Config dict with a ``variations`` list.
        :raises LDApiError: On non-200 HTTP responses or network errors.
        """
        path = f"/api/v2/projects/{project_key}/ai-configs/{config_key}"
        return self._request("GET", path, extra_headers={"LD-API-Version": "beta"})

    def get_ai_config_variation(
        self, project_key: str, config_key: str, variation_key: str
    ) -> Dict[str, Any]:
        """Fetch a specific variation of an AI config by key.

        Returns the first (latest) item from the variations response.

        :param project_key: LaunchDarkly project key.
        :param config_key: Key of the AI Config (aiConfigKey).
        :param variation_key: Key of the specific variation to fetch.
        :return: The variation dict (first item from the ``items`` array).
        :raises LDApiError: If the variation is not found or the request fails.
        """
        path = f"/api/v2/projects/{project_key}/ai-configs/{config_key}/variations/{variation_key}"
        result = self._request("GET", path, extra_headers={"LD-API-Version": "beta"})
        items = result.get("items") if isinstance(result, dict) else None
        if not items:
            raise LDApiError(
                f"Variation '{variation_key}' not found for AI config '{config_key}'.",
                path=path,
            )
        return items[0]

    def create_ai_config_variation(
        self, project_key: str, config_key: str, payload: Dict[str, Any]
    ) -> Any:
        """Create a new variation on an AI Config.

        :param project_key: LaunchDarkly project key.
        :param config_key: Key of the AI Config.
        :param payload: Variation payload (key, name, mode, instructions, model).
        :return: Created AIConfigVariation dict.
        :raises LDApiError: On non-200 HTTP responses or network errors.
        """
        path = f"/api/v2/projects/{project_key}/ai-configs/{config_key}/variations"
        return self._request("POST", path, body=payload, extra_headers={"LD-API-Version": "beta"})

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
        self, project_key: str, optimization_key: str, payload: AgentOptimizationResultPost
    ) -> Optional[str]:
        """Create an iteration result record for the given optimization run.

        Errors are caught and logged rather than raised so that persistence
        failures never abort an in-progress optimization run.

        :param project_key: LaunchDarkly project key.
        :param optimization_key: String key of the parent agent_optimization record.
        :param payload: POST payload for this iteration.
        :return: The ``id`` of the newly created result record, or None on failure.
        """
        path = f"/api/v2/projects/{project_key}/agent-optimizations/{optimization_key}/results"
        try:
            result = self._request("POST", path, body=payload)
            return result.get("id") if isinstance(result, dict) else None
        except LDApiError as exc:
            logger.debug(
                "Failed to persist optimization result (optimization_key=%s, iteration=%s): %s",
                optimization_key,
                payload.get("iteration"),
                exc,
            )
            return None
        except Exception as exc:
            logger.debug(
                "Unexpected error persisting optimization result (optimization_key=%s, iteration=%s): %s",
                optimization_key,
                payload.get("iteration"),
                exc,
            )
            return None

    def patch_agent_optimization_result(
        self, project_key: str, optimization_key: str, result_id: str, payload: AgentOptimizationResultPatch
    ) -> None:
        """Update an existing iteration result record.

        Errors are caught and logged rather than raised so that persistence
        failures never abort an in-progress optimization run.

        :param project_key: LaunchDarkly project key.
        :param optimization_key: String key of the parent agent_optimization record.
        :param result_id: ID of the result record to update.
        :param payload: PATCH payload with fields to update.
        """
        path = f"/api/v2/projects/{project_key}/agent-optimizations/{optimization_key}/results/{result_id}"
        try:
            self._request("PATCH", path, body=payload)
        except LDApiError as exc:
            logger.debug(
                "Failed to update optimization result (result_id=%s): %s",
                result_id,
                exc,
            )
        except Exception as exc:
            logger.debug(
                "Unexpected error updating optimization result (result_id=%s): %s",
                result_id,
                exc,
            )
