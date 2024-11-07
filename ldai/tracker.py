import time
from typing import Dict, Union
from ldclient import Context, LDClient
from ldai.types import BedrockTokenUsage, FeedbackKind, OpenAITokenUsage, TokenUsage

class LDAIConfigTracker:
    def __init__(self, ld_client: LDClient, version_key: str, config_key: str, context: Context):
        self.ld_client = ld_client
        self.version_key = version_key
        self.config_key = config_key
        self.context = context

    def get_track_data(self):
        return {
            'versionKey': self.version_key,
            'configKey': self.config_key,
        }
    
    def track_duration(self, duration: int) -> None:
        self.ld_client.track('$ld:ai:duration:total', self.context, self.get_track_data(), duration)

    def track_duration_of(self, func):
        start_time = time.time()
        result = func()
        end_time = time.time()
        duration = int((end_time - start_time) * 1000)  # duration in milliseconds
        self.track_duration(duration)
        return result

    def track_feedback(self, feedback: Dict[str, FeedbackKind]) -> None:
        if feedback['kind'] == FeedbackKind.Positive:
            self.ld_client.track('$ld:ai:feedback:user:positive', self.context, self.get_track_data(), 1)
        elif feedback['kind'] == FeedbackKind.Negative:
            self.ld_client.track('$ld:ai:feedback:user:negative', self.context, self.get_track_data(), 1)

    def track_success(self) -> None:
        self.ld_client.track('$ld:ai:generation', self.context, self.get_track_data(), 1)

    def track_openai(self, func):
        result = self.track_duration_of(func)
        if result.usage:
            self.track_tokens(OpenAITokenUsage(result.usage))
        return result

    def track_bedrock_converse(self, res: dict) -> dict:    
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
        token_metrics = tokens.to_metrics()
        if token_metrics['total'] > 0:
            self.ld_client.track('$ld:ai:tokens:total', self.context, self.get_track_data(), token_metrics.total)
        if token_metrics['input'] > 0:
            self.ld_client.track('$ld:ai:tokens:input', self.context, self.get_track_data(), token_metrics.input)
        if token_metrics['output'] > 0:
            self.ld_client.track('$ld:ai:tokens:output', self.context, self.get_track_data(), token_metrics.output)