import time
from typing import Dict, Union
from ldclient import Context, LDClient
from ldai.types import BedrockTokenUsage, FeedbackKind, OpenAITokenUsage, TokenUsage, UnderscoreTokenUsage

class LDAIConfigTracker:
    def __init__(self, ld_client: LDClient, variation_id: str, config_key: str, context: Context):
        self.ld_client = ld_client
        self.variation_id = variation_id
        self.config_key = config_key
        self.context = context

    def get_track_data(self):
        return {
            'variationId': self.variation_id,
            'configKey': self.config_key,
        }
    
    def track_duration(self, duration: int) -> None:
        self.ld_client.track('$ld:ai:duration:total', self.context, self.get_track_data(), duration)

    def track_duration_of(self, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = int((end_time - start_time) * 1000)  # duration in milliseconds
        self.track_duration(duration)
        return result

    def track_error(self, error: int) -> None:
        self.ld_client.track('$ld:ai:error', self.context, self.get_track_data(), error)

    def track_feedback(self, feedback: Dict[str, FeedbackKind]) -> None:
        if feedback['kind'] == FeedbackKind.Positive:
            self.ld_client.track('$ld:ai:feedback:user:positive', self.context, self.get_track_data(), 1)
        elif feedback['kind'] == FeedbackKind.Negative:
            self.ld_client.track('$ld:ai:feedback:user:negative', self.context, self.get_track_data(), 1)

    def track_generation(self, generation: int) -> None:
        self.ld_client.track('$ld:ai:generation', self.context, self.get_track_data(), generation)

    def track_openai(self, func, *args, **kwargs):
        result = self.track_duration_of(func, *args, **kwargs)
        if result.usage:
            self.track_tokens(OpenAITokenUsage(result.usage))
        return result

    def track_tokens(self, tokens: Union[TokenUsage, UnderscoreTokenUsage, BedrockTokenUsage]) -> None:
        token_metrics = tokens.to_metrics()
        if token_metrics['total'] > 0:
            self.ld_client.track('$ld:ai:tokens:total', self.context, self.get_track_data(), token_metrics['total'])
        if token_metrics['input'] > 0:
            self.ld_client.track('$ld:ai:tokens:input', self.context, self.get_track_data(), token_metrics['input'])
        if token_metrics['output'] > 0:
            self.ld_client.track('$ld:ai:tokens:output', self.context, self.get_track_data(), token_metrics['output'])