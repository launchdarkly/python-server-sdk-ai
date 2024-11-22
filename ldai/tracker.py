import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union

from ldclient import Context, LDClient


@dataclass
class TokenMetrics:
    """
    Metrics for token usage in AI operations.

    :param total: Total number of tokens used.
    :param input: Number of input tokens.
    :param output: Number of output tokens.
    """

    total: int
    input: int
    output: int  # type: ignore


@dataclass
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

    :param total_tokens: Total number of tokens used.
    :param prompt_tokens: Number of tokens in the prompt.
    :param completion_tokens: Number of tokens in the completion.
    """

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

    def to_metrics(self):
        """
        Convert token usage to metrics format.

        :return: Dictionary containing token metrics.
        """
        return {
            'total': self['total_tokens'],
            'input': self['prompt_tokens'],
            'output': self['completion_tokens'],
        }


@dataclass
class LDOpenAIUsage:
    """
    LaunchDarkly-specific OpenAI usage tracking.

    :param total_tokens: Total number of tokens used.
    :param prompt_tokens: Number of tokens in the prompt.
    :param completion_tokens: Number of tokens in the completion.
    """

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


@dataclass
class OpenAITokenUsage:
    """
    Tracks OpenAI-specific token usage.
    """

    def __init__(self, data: LDOpenAIUsage):
        """
        Initialize OpenAI token usage tracking.

        :param data: OpenAI usage data.
        """
        self.total_tokens = data.total_tokens
        self.prompt_tokens = data.prompt_tokens
        self.completion_tokens = data.completion_tokens

    def to_metrics(self) -> TokenMetrics:
        """
        Convert OpenAI token usage to metrics format.

        :return: TokenMetrics object containing usage data.
        """
        return TokenMetrics(
            total=self.total_tokens,
            input=self.prompt_tokens,
            output=self.completion_tokens,
        )


@dataclass
class BedrockTokenUsage:
    """
    Tracks AWS Bedrock-specific token usage.
    """

    def __init__(self, data: dict):
        """
        Initialize Bedrock token usage tracking.

        :param data: Dictionary containing Bedrock usage data.
        """
        self.totalTokens = data.get('totalTokens', 0)
        self.inputTokens = data.get('inputTokens', 0)
        self.outputTokens = data.get('outputTokens', 0)

    def to_metrics(self) -> TokenMetrics:
        """
        Convert Bedrock token usage to metrics format.

        :return: TokenMetrics object containing usage data.
        """
        return TokenMetrics(
            total=self.totalTokens,
            input=self.inputTokens,
            output=self.outputTokens,
        )


class LDAIConfigTracker:
    """
    Tracks configuration and usage metrics for LaunchDarkly AI operations.
    """

    def __init__(
        self, ld_client: LDClient, version_key: str, config_key: str, context: Context
    ):
        """
        Initialize an AI configuration tracker.

        :param ld_client: LaunchDarkly client instance.
        :param version_key: Version key for tracking.
        :param config_key: Configuration key for tracking.
        :param context: Context for evaluation.
        """
        self.ld_client = ld_client
        self.version_key = version_key
        self.config_key = config_key
        self.context = context

    def __get_track_data(self):
        """
        Get tracking data for events.

        :return: Dictionary containing version and config keys.
        """
        return {
            'versionKey': self.version_key,
            'configKey': self.config_key,
        }

    def track_duration(self, duration: int) -> None:
        """
        Manually track the duration of an AI operation.

        :param duration: Duration in milliseconds.
        """
        self.ld_client.track(
            '$ld:ai:duration:total', self.context, self.__get_track_data(), duration
        )

    def track_duration_of(self, func):
        """
        Automatically track the duration of an AI operation.

        :param func: Function to track.
        :return: Result of the tracked function.
        """
        start_time = time.time()
        result = func()
        end_time = time.time()
        duration = int((end_time - start_time) * 1000)  # duration in milliseconds
        self.track_duration(duration)
        return result

    def track_feedback(self, feedback: Dict[str, FeedbackKind]) -> None:
        """
        Track user feedback for an AI operation.

        :param feedback: Dictionary containing feedback kind.
        """
        if feedback['kind'] == FeedbackKind.Positive:
            self.ld_client.track(
                '$ld:ai:feedback:user:positive',
                self.context,
                self.__get_track_data(),
                1,
            )
        elif feedback['kind'] == FeedbackKind.Negative:
            self.ld_client.track(
                '$ld:ai:feedback:user:negative',
                self.context,
                self.__get_track_data(),
                1,
            )

    def track_success(self) -> None:
        """
        Track a successful AI generation.
        """
        self.ld_client.track(
            '$ld:ai:generation', self.context, self.__get_track_data(), 1
        )

    def track_openai_metrics(self, func):
        """
        Track OpenAI-specific operations.

        :param func: Function to track.
        :return: Result of the tracked function.
        """
        result = self.track_duration_of(func)
        if result.usage:
            self.track_tokens(OpenAITokenUsage(result.usage))
        return result

    def track_bedrock_converse_metrics(self, res: dict) -> dict:
        """
        Track AWS Bedrock conversation operations.

        :param res: Response dictionary from Bedrock.
        :return: The original response dictionary.
        """
        status_code = res.get('$metadata', {}).get('httpStatusCode', 0)
        if status_code == 200:
            self.track_success()
        elif status_code >= 400:
            # Potentially add error tracking in the future.
            pass
        if res.get('metrics', {}).get('latencyMs'):
            self.track_duration(res['metrics']['latencyMs'])
        if res.get('usage'):
            self.track_tokens(BedrockTokenUsage(res['usage']))
        return res

    def track_tokens(self, tokens: Union[TokenUsage, BedrockTokenUsage]) -> None:
        """
        Track token usage metrics.

        :param tokens: Token usage data from either custom, OpenAI, or Bedrock sources.
        """
        token_metrics = tokens.to_metrics()
        if token_metrics.total > 0:
            self.ld_client.track(
                '$ld:ai:tokens:total',
                self.context,
                self.__get_track_data(),
                token_metrics.total,
            )
        if token_metrics.input > 0:
            self.ld_client.track(
                '$ld:ai:tokens:input',
                self.context,
                self.__get_track_data(),
                token_metrics.input,
            )
        if token_metrics.output > 0:
            self.ld_client.track(
                '$ld:ai:tokens:output',
                self.context,
                self.__get_track_data(),
                token_metrics.output,
            )
