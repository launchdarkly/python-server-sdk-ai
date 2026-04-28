import base64
import json
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

from ldclient import Context, LDClient, Result

from ldai import log


class FeedbackKind(Enum):
    """
    Types of feedback that can be provided for AI operations.
    """

    Positive = "positive"
    Negative = "negative"


@dataclass
class TokenUsage:
    """
    Tracks token usage for AI operations.

    :param total: Total number of tokens used.
    :param input: Number of tokens in the prompt.
    :param output: Number of tokens in the completion.
    """

    total: int
    input: int
    output: int


class LDAIMetricSummary:
    """
    Summary of metrics which have been tracked.
    """

    def __init__(self):
        self._duration_ms: Optional[int] = None
        self._success: Optional[bool] = None
        self._feedback: Optional[Dict[str, FeedbackKind]] = None
        self._usage: Optional[TokenUsage] = None
        self._time_to_first_token: Optional[int] = None
        self._tool_calls: Optional[List[str]] = None
        self._resumption_token: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[int]:
        """Duration of the AI operation in milliseconds."""
        return self._duration_ms

    @property
    def duration(self) -> Optional[int]:
        """
        .. deprecated::
            Use :attr:`duration_ms` instead.
        """
        warnings.warn(
            "LDAIMetricSummary.duration is deprecated. Use duration_ms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._duration_ms

    @property
    def success(self) -> Optional[bool]:
        return self._success

    @property
    def feedback(self) -> Optional[Dict[str, FeedbackKind]]:
        return self._feedback

    @property
    def usage(self) -> Optional[TokenUsage]:
        return self._usage

    @property
    def time_to_first_token(self) -> Optional[int]:
        return self._time_to_first_token

    @property
    def tool_calls(self) -> Optional[List[str]]:
        """List of tool keys that were invoked during this operation."""
        return self._tool_calls

    @property
    def resumption_token(self) -> Optional[str]:
        """
        URL-safe Base64-encoded resumption token captured at tracker
        instantiation. Useful for deferred feedback flows where a downstream
        process needs to associate events with the original execution.
        """
        return self._resumption_token


class LDAIConfigTracker:
    """
    Tracks configuration and usage metrics for LaunchDarkly AI operations.
    """

    def __init__(
        self,
        ld_client: LDClient,
        run_id: str,
        config_key: str,
        variation_key: str,
        version: int,
        context: Context,
        model_name: str,
        provider_name: str,
        graph_key: Optional[str] = None,
    ):
        """
        Initialize an AI Config tracker.

        :param ld_client: LaunchDarkly client instance.
        :param run_id: Unique identifier for this execution.
        :param config_key: Configuration key for tracking.
        :param variation_key: Variation key for tracking.
        :param version: Version of the variation.
        :param context: Context for evaluation.
        :param model_name: Name of the model used.
        :param provider_name: Name of the provider used.
        :param graph_key: When set, include ``graphKey`` in all event payloads
            (e.g. config-level metrics inside a graph).
        """
        self._ld_client = ld_client
        self._variation_key = variation_key
        self._config_key = config_key
        self._version = version
        self._model_name = model_name
        self._provider_name = provider_name
        self._context = context
        self._graph_key = graph_key
        self._run_id = run_id
        self._summary = LDAIMetricSummary()
        # Capture resumption_token immediately so it's available on the summary at instantiation.
        self._summary._resumption_token = self.resumption_token

    @property
    def resumption_token(self) -> str:
        """
        A URL-safe Base64-encoded JSON string that can be used to reconstruct
        a tracker in a different process (e.g. for deferred feedback).

        The token contains ``runId``, ``configKey``, ``version``, and
        optionally ``variationKey`` and ``graphKey`` (omitted when empty).
        ``modelName`` and ``providerName`` are **not** included.
        """
        data: dict = {
            "runId": self._run_id,
            "configKey": self._config_key,
        }
        if self._variation_key:
            data["variationKey"] = self._variation_key
        data["version"] = self._version
        if self._graph_key:
            data["graphKey"] = self._graph_key
        payload = json.dumps(data)
        return base64.urlsafe_b64encode(payload.encode("utf-8")).rstrip(b"=").decode("utf-8")

    @classmethod
    def from_resumption_token(cls, token: str, ld_client: LDClient, context: Context) -> Result:
        """
        Reconstruct a tracker from a resumption token.

        This is used for cross-process scenarios such as deferred feedback,
        where a different service needs to associate tracking events with the
        original execution's ``runId``.

        :param token: A URL-safe Base64-encoded resumption token obtained from
            :attr:`resumption_token`.
        :param ld_client: LaunchDarkly client instance.
        :param context: The context to use for track events.
        :return: A :class:`Result` whose ``value`` is a new
            :class:`LDAIConfigTracker` bound to the original ``runId`` from the
            token on success, or whose ``error`` describes the problem on failure.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            payload = json.loads(
                base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
            )
        except Exception as e:
            return Result.fail(f"Invalid resumption token: {e}", e)

        for field in ("runId", "configKey", "version"):
            if field not in payload:
                return Result.fail(
                    f"Invalid resumption token: missing required field '{field}'"
                )

        return Result.success(cls(
            ld_client=ld_client,
            run_id=payload["runId"],
            config_key=payload["configKey"],
            variation_key=payload.get("variationKey") or "",
            version=payload["version"],
            context=context,
            model_name="",
            provider_name="",
            graph_key=payload.get("graphKey"),
        ))

    def __get_track_data(self) -> dict:
        """
        Get tracking data for events.

        :return: Dictionary containing variation and config keys.
        """
        data = {
            "runId": self._run_id,
            "configKey": self._config_key,
            "version": self._version,
            "modelName": self._model_name,
            "providerName": self._provider_name,
        }
        if self._variation_key:
            data["variationKey"] = self._variation_key
        if self._graph_key:
            data['graphKey'] = self._graph_key
        return data

    def track_duration(self, duration: int) -> None:
        """
        Manually track the duration of an AI operation.

        :param duration: Duration in milliseconds.
        """
        if self._summary.duration_ms is not None:
            log.warning("Duration has already been tracked for this execution. %s", self.__get_track_data())
            return
        self._summary._duration_ms = duration
        self._ld_client.track(
            "$ld:ai:duration:total", self._context, self.__get_track_data(), duration
        )

    def track_time_to_first_token(self, time_to_first_token: int) -> None:
        """
        Manually track the time to first token of an AI operation.

        :param time_to_first_token: Time to first token in milliseconds.
        """
        if self._summary.time_to_first_token is not None:
            log.warning(
                "Time to first token has already been tracked for this execution. %s",
                self.__get_track_data(),
            )
            return
        self._summary._time_to_first_token = time_to_first_token
        self._ld_client.track(
            "$ld:ai:tokens:ttf",
            self._context,
            self.__get_track_data(),
            time_to_first_token,
        )

    def track_duration_of(self, func):
        """
        Automatically track the duration of an AI operation.

        An exception occurring during the execution of the function will still
        track the duration. The exception will be re-thrown.

        :param func: Function to track (synchronous only).
        :return: Result of the tracked function.
        """
        start_ns = time.perf_counter_ns()
        try:
            result = func()
        finally:
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000  # duration in milliseconds
            self.track_duration(duration)

        return result

    def _track_from_metrics_extractor(
        self,
        result: Any,
        metrics_extractor: Callable[[Any], Any],
    ) -> Any:
        metrics = metrics_extractor(result)
        if metrics.success:
            self.track_success()
        else:
            self.track_error()
        if metrics.usage:
            self.track_tokens(metrics.usage)
        if getattr(metrics, 'tool_calls', None):
            self.track_tool_calls(metrics.tool_calls)
        return result

    def track_metrics_of(
        self,
        metrics_extractor: Callable[[Any], Any],
        func: Callable[[], Any],
    ) -> Any:
        """
        Track metrics for a synchronous AI operation.

        This function will track the duration of the operation, extract metrics using the provided
        metrics extractor function, and track success or error status accordingly.

        If the provided function throws, then this method will also throw.
        In the case the provided function throws, this function will record the duration and an error.
        A failed operation will not have any token usage data.

        For async operations, use :meth:`track_metrics_of_async`.

        When the extracted :class:`~ldai.providers.types.LDAIMetrics` object has a
        non-``None`` ``duration_ms`` field, that value is used as the measured duration
        instead of the wall-clock elapsed time.

        :param metrics_extractor: Function that extracts LDAIMetrics from the operation result
        :param func: Synchronous callable that runs the operation
        :return: The result of the operation
        """
        start_ns = time.perf_counter_ns()
        try:
            result = func()
        except Exception as err:
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000
            self.track_duration(duration)
            self.track_error()
            raise err

        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
        metrics = metrics_extractor(result)
        reported_ms = getattr(metrics, 'duration_ms', None) if metrics else None
        self.track_duration(reported_ms if reported_ms is not None else elapsed_ms)
        return self._track_from_metrics_extractor(result, metrics_extractor)

    async def track_metrics_of_async(self, metrics_extractor, func):
        """
        Track metrics for an async AI operation (``func`` is awaited).

        Same event semantics as :meth:`track_metrics_of`.

        When the extracted :class:`~ldai.providers.types.LDAIMetrics` object has a
        non-``None`` ``duration_ms`` field, that value is used as the measured duration
        instead of the wall-clock elapsed time.

        :param metrics_extractor: Function that extracts LDAIMetrics from the operation result
        :param func: Async callable or zero-arg callable that returns an awaitable when called
        :return: The result of the operation
        """
        start_ns = time.perf_counter_ns()
        result = None
        try:
            result = await func()
        except Exception as err:
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000
            self.track_duration(duration)
            self.track_error()
            raise err

        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
        metrics = metrics_extractor(result)
        reported_ms = getattr(metrics, 'duration_ms', None) if metrics else None
        self.track_duration(reported_ms if reported_ms is not None else elapsed_ms)
        return self._track_from_metrics_extractor(result, metrics_extractor)

    def track_judge_result(self, judge_result: Any) -> None:
        """
        Track a judge result, including the evaluation score with judge config key.

        :param judge_result: JudgeResult object containing score, metric key, and success status
        """
        if not judge_result.sampled:
            return

        if judge_result.success and judge_result.metric_key:
            track_data = self.__get_track_data()
            if judge_result.judge_config_key:
                track_data = {**track_data, 'judgeConfigKey': judge_result.judge_config_key}
            self._ld_client.track(
                judge_result.metric_key,
                self._context,
                track_data,
                judge_result.score,
            )

    def track_feedback(self, feedback: Dict[str, FeedbackKind]) -> None:
        """
        Track user feedback for an AI operation.

        :param feedback: Dictionary containing feedback kind.
        """
        if self._summary.feedback is not None:
            log.warning("Feedback has already been tracked for this execution. %s", self.__get_track_data())
            return
        self._summary._feedback = feedback
        if feedback["kind"] == FeedbackKind.Positive:
            self._ld_client.track(
                "$ld:ai:feedback:user:positive",
                self._context,
                self.__get_track_data(),
                1,
            )
        elif feedback["kind"] == FeedbackKind.Negative:
            self._ld_client.track(
                "$ld:ai:feedback:user:negative",
                self._context,
                self.__get_track_data(),
                1,
            )

    def track_tool_calls(self, tool_calls: List[str]) -> None:
        """
        Track the tool calls made during an AI operation.

        :param tool_calls: List of tool call names.
        """
        if self._summary.tool_calls is not None:
            log.warning("Tool calls have already been tracked for this execution. %s", self.__get_track_data())
            return
        self._summary._tool_calls = list(tool_calls)

    def track_success(self) -> None:
        """
        Track a successful AI generation.
        """
        if self._summary.success is not None:
            log.warning("Success has already been tracked for this execution. %s", self.__get_track_data())
            return
        self._summary._success = True
        self._ld_client.track(
            "$ld:ai:generation:success", self._context, self.__get_track_data(), 1
        )

    def track_error(self) -> None:
        """
        Track an unsuccessful AI generation attempt.
        """
        if self._summary.success is not None:
            log.warning("Success has already been tracked for this execution. %s", self.__get_track_data())
            return
        self._summary._success = False
        self._ld_client.track(
            "$ld:ai:generation:error", self._context, self.__get_track_data(), 1
        )

    def track_openai_metrics(self, func):
        """
        Track OpenAI-specific operations.

        .. deprecated:: Use :meth:`track_metrics_of` with ``get_ai_metrics_from_response``
            from ``ldai_openai`` instead. This method will be removed in a future version.

        This function will track the duration of the operation, the token
        usage, and the success or error status.

        If the provided function throws, then this method will also throw.

        In the case the provided function throws, this function will record the
        duration and an error.

        A failed operation will not have any token usage data.

        :param func: Function to track.
        :return: Result of the tracked function.
        """
        warnings.warn(
            "track_openai_metrics is deprecated. Use track_metrics_of with "
            "get_ai_metrics_from_response from ldai_openai instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        start_ns = time.perf_counter_ns()
        try:
            result = func()
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000
            self.track_duration(duration)
            self.track_success()
            if hasattr(result, "usage") and hasattr(result.usage, "to_dict"):
                self.track_tokens(_openai_to_token_usage(result.usage.to_dict()))
        except Exception:
            duration = (time.perf_counter_ns() - start_ns) // 1_000_000
            self.track_duration(duration)
            self.track_error()
            raise

        return result

    def track_bedrock_converse_metrics(self, res: dict) -> dict:
        """
        Track AWS Bedrock conversation operations.


        This function will track the duration of the operation, the token
        usage, and the success or error status.

        :param res: Response dictionary from Bedrock.
        :return: The original response dictionary.
        """
        status_code = res.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if status_code == 200:
            self.track_success()
        elif status_code >= 400:
            self.track_error()
        if res.get("metrics", {}).get("latencyMs"):
            self.track_duration(res["metrics"]["latencyMs"])
        if res.get("usage"):
            self.track_tokens(_bedrock_to_token_usage(res["usage"]))
        return res

    def track_tokens(self, tokens: TokenUsage) -> None:
        """
        Track token usage metrics.

        :param tokens: Token usage data from either custom, OpenAI, or Bedrock sources.
        """
        if self._summary.usage is not None:
            log.warning("Tokens have already been tracked for this execution. %s", self.__get_track_data())
            return
        self._summary._usage = tokens
        td = self.__get_track_data()
        if tokens.total > 0:
            self._ld_client.track(
                "$ld:ai:tokens:total",
                self._context,
                td,
                tokens.total,
            )
        if tokens.input > 0:
            self._ld_client.track(
                "$ld:ai:tokens:input",
                self._context,
                td,
                tokens.input,
            )
        if tokens.output > 0:
            self._ld_client.track(
                "$ld:ai:tokens:output",
                self._context,
                td,
                tokens.output,
            )

    def track_tool_call(self, tool_key: str) -> None:
        """
        Track a tool invocation for this configuration (standalone or within a graph).

        :param tool_key: Identifier of the tool that was invoked.
        """
        track_data = {**self.__get_track_data(), "toolKey": tool_key}
        self._ld_client.track(
            "$ld:ai:tool_call",
            self._context,
            track_data,
            1,
        )

    def track_tool_calls(self, tool_keys: Iterable[str]) -> None:
        """
        Track multiple tool invocations for this configuration.

        :param tool_keys: Tool identifiers (e.g. from a model response).
        """
        for tool_key in tool_keys:
            self.track_tool_call(tool_key)

    def get_summary(self) -> LDAIMetricSummary:
        """
        Get the current summary of AI metrics.

        :return: Summary of AI metrics.
        """
        return self._summary


def _bedrock_to_token_usage(data: dict) -> TokenUsage:
    """
    Convert a Bedrock usage dictionary to a TokenUsage object.

    :param data: Dictionary containing Bedrock usage data.
    :return: TokenUsage object containing usage data.
    """
    return TokenUsage(
        total=data.get("totalTokens", 0),
        input=data.get("inputTokens", 0),
        output=data.get("outputTokens", 0),
    )


def _openai_to_token_usage(data: dict) -> TokenUsage:
    """
    Convert an OpenAI usage dictionary to a TokenUsage object.

    :param data: Dictionary containing OpenAI usage data.
    :return: TokenUsage object containing usage data.
    """
    return TokenUsage(
        total=data.get("total_tokens", 0),
        input=data.get("prompt_tokens", 0),
        output=data.get("completion_tokens", 0),
    )


class AIGraphTracker:
    """
    Tracks graph-level, node-level, and edge-level metrics for AI agent graph operations.
    """

    def __init__(
        self,
        ld_client: LDClient,
        variation_key: str,
        graph_key: str,
        version: int,
        context: Context,
    ):
        """
        Initialize an AI Graph tracker.

        :param ld_client: LaunchDarkly client instance.
        :param variation_key: Variation key for tracking.
        :param graph_key: Graph configuration key for tracking.
        :param version: Version of the variation.
        :param context: Context for evaluation.
        """
        self._ld_client = ld_client
        self._variation_key = variation_key
        self._graph_key = graph_key
        self._version = version
        self._context = context

    @property
    def graph_key(self) -> str:
        """Graph configuration key used in tracking payloads."""
        return self._graph_key

    def __get_track_data(self):
        """
        Get tracking data for events.

        :return: Dictionary containing variation, graph key, and version.
        """
        track_data = {
            "variationKey": self._variation_key,
            "graphKey": self._graph_key,
            "version": self._version,
        }
        return track_data

    def track_invocation_success(self) -> None:
        """
        Track a successful graph invocation.
        """
        self._ld_client.track(
            "$ld:ai:graph:invocation_success",
            self._context,
            self.__get_track_data(),
            1,
        )

    def track_invocation_failure(self) -> None:
        """
        Track an unsuccessful graph invocation.
        """
        self._ld_client.track(
            "$ld:ai:graph:invocation_failure",
            self._context,
            self.__get_track_data(),
            1,
        )

    def track_duration(self, duration: int) -> None:
        """
        Track the total duration of graph execution.

        :param duration: Duration in milliseconds.
        """
        self._ld_client.track(
            "$ld:ai:graph:duration:total",
            self._context,
            self.__get_track_data(),
            duration,
        )

    def track_total_tokens(self, tokens: Optional[TokenUsage] = None) -> None:
        """
        Track aggregated token usage across the entire graph invocation.

        :param tokens: Token usage data, or ``None`` when usage is unknown.
        """
        if tokens is None or tokens.total <= 0:
            return
        self._ld_client.track(
            "$ld:ai:graph:total_tokens",
            self._context,
            self.__get_track_data(),
            tokens.total,
        )

    def track_path(self, path: List[str]) -> None:
        """
        Track the execution path through the graph.

        :param path: An array of configuration keys representing the sequence of nodes executed during graph traversal.
        """
        track_data = {**self.__get_track_data(), "path": path}
        self._ld_client.track(
            "$ld:ai:graph:path",
            self._context,
            track_data,
            1,
        )

    def track_redirect(self, source_key: str, redirected_target: str) -> None:
        """
        Track when a node redirects to a different target than originally specified.

        :param source_key: The configuration key of the source node.
        :param redirected_target: The configuration key of the target node that was redirected to.
        """
        track_data = {
            **self.__get_track_data(),
            "sourceKey": source_key,
            "redirectedTarget": redirected_target,
        }
        self._ld_client.track(
            "$ld:ai:graph:redirect",
            self._context,
            track_data,
            1,
        )

    def track_handoff_success(self, source_key: str, target_key: str) -> None:
        """
        Track successful handoffs between nodes.

        :param source_key: The configuration key of the source node.
        :param target_key: The configuration key of the target node.
        """
        track_data = {
            **self.__get_track_data(),
            "sourceKey": source_key,
            "targetKey": target_key,
        }
        self._ld_client.track(
            "$ld:ai:graph:handoff_success",
            self._context,
            track_data,
            1,
        )

    def track_handoff_failure(self, source_key: str, target_key: str) -> None:
        """
        Track failed handoffs between nodes.

        :param source_key: The configuration key of the source node.
        :param target_key: The configuration key of the target node.
        """
        track_data = {
            **self.__get_track_data(),
            "sourceKey": source_key,
            "targetKey": target_key,
        }
        self._ld_client.track(
            "$ld:ai:graph:handoff_failure",
            self._context,
            track_data,
            1,
        )
