from typing import Union
from ldai.types import BedrockTokenUsage, TokenMetrics, OpenAITokenUsage, UnderscoreTokenUsage

def usage_to_token_metrics(usage: Union[OpenAITokenUsage, UnderscoreTokenUsage, BedrockTokenUsage]) -> TokenMetrics:
    return usage.to_metrics()