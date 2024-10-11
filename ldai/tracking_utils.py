from typing import Union
from ldai.types import BedrockTokenUsage, TokenMetrics, TokenUsage, UnderscoreTokenUsage

def usage_to_token_metrics(usage: Union[TokenUsage, UnderscoreTokenUsage, BedrockTokenUsage]) -> TokenMetrics:
    def get_attr(obj, attr, default=0):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    if 'inputTokens' in usage and 'outputTokens' in usage:
        # Bedrock usage
        return {
            'total': get_attr(usage, 'totalTokens'),
            'input': get_attr(usage, 'inputTokens'),
            'output': get_attr(usage, 'outputTokens'),
        }

    # OpenAI usage (both camelCase and snake_case)
    return {
        'total': get_attr(usage, 'total_tokens', get_attr(usage, 'totalTokens', 0)),
        'input': get_attr(usage, 'prompt_tokens', get_attr(usage, 'promptTokens', 0)),
        'output': get_attr(usage, 'completion_tokens', get_attr(usage, 'completionTokens', 0)),
    }