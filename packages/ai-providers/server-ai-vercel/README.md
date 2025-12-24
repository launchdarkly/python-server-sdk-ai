# LaunchDarkly AI SDK Vercel Provider

[![PyPI](https://img.shields.io/pypi/v/launchdarkly-server-sdk-ai-vercel-dev.svg?style=flat-square)](https://pypi.org/project/launchdarkly-server-sdk-ai-vercel-dev/)

This package provides a multi-provider integration for the LaunchDarkly AI SDK, similar to the Vercel AI SDK in JavaScript. It uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood to support 100+ LLM providers.

## Installation

```bash
pip install launchdarkly-server-sdk-ai-vercel-dev
```

## Supported Providers

This provider supports all LiteLLM-compatible providers, including:

- OpenAI
- Anthropic
- Google (Gemini)
- Cohere
- Mistral
- Azure OpenAI
- AWS Bedrock
- And many more...

## Quick Start

```python
import asyncio
from ldai import AIClient
from ldai_vercel import VercelProvider

async def main():
    # Initialize the AI client
    ai_client = AIClient(ld_client)
    
    # Get AI config
    ai_config = ai_client.config(
        "my-ai-config-key",
        context,
        default_value
    )
    
    # Create a Vercel provider from the config
    provider = await VercelProvider.create(ai_config)
    
    # Invoke the model
    response = await provider.invoke_model(ai_config.messages)
    print(response.message.content)

asyncio.run(main())
```

## Features

- Multi-provider support through LiteLLM
- Automatic token usage tracking
- Support for structured output (JSON schema)
- Parameter mapping between LaunchDarkly and LiteLLM formats
- Static utility methods for custom integrations

## API Reference

### VercelProvider

#### Constructor

```python
VercelProvider(model_name: str, parameters: VercelModelParameters, logger: Optional[Any] = None)
```

#### Static Methods

- `create(ai_config: AIConfigKind, logger: Optional[Any] = None) -> VercelProvider` - Factory method to create a provider from an AI config
- `get_ai_metrics_from_response(response: Any) -> LDAIMetrics` - Extract metrics from a LiteLLM response
- `map_provider(ld_provider_name: str) -> str` - Map LD provider names to LiteLLM format
- `map_parameters(parameters: Dict) -> VercelModelParameters` - Map LD parameters to LiteLLM format

#### Instance Methods

- `invoke_model(messages: List[LDMessage]) -> ChatResponse` - Invoke the model with messages
- `invoke_structured_model(messages: List[LDMessage], response_structure: Dict[str, Any]) -> StructuredResponse` - Invoke the model with structured output

## Environment Variables

Make sure to set the appropriate API key environment variables for your chosen provider:

- `OPENAI_API_KEY` - For OpenAI
- `ANTHROPIC_API_KEY` - For Anthropic
- `GOOGLE_API_KEY` - For Google/Gemini
- `COHERE_API_KEY` - For Cohere
- `MISTRAL_API_KEY` - For Mistral

## License

Apache-2.0

