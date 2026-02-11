# LaunchDarkly AI SDK OpenAI Provider

[![PyPI](https://img.shields.io/pypi/v/launchdarkly-server-sdk-ai-openai-dev.svg?style=flat-square)](https://pypi.org/project/launchdarkly-server-sdk-ai-openai-dev/)

> [!CAUTION]
> This package is in pre-release and not subject to backwards compatibility
> guarantees. The API may change based on feedback.
>
> Pin to a specific minor version to avoid breaking changes.

This package provides an OpenAI integration for the LaunchDarkly AI SDK.

## Installation

```bash
pip install launchdarkly-server-sdk-ai-openai-dev
```

## Quick Start

```python
import asyncio
from ldai import AIClient
from ldai_openai import OpenAIProvider

async def main():
    # Initialize the AI client
    ai_client = AIClient(ld_client)
    
    # Get AI config
    ai_config = ai_client.config(
        "my-ai-config-key",
        context,
        default_value
    )
    
    # Create an OpenAI provider from the config
    provider = await OpenAIProvider.create(ai_config)
    
    # Invoke the model
    response = await provider.invoke_model(ai_config.messages)
    print(response.message.content)

asyncio.run(main())
```

## Features

- Full integration with OpenAI's chat completions API
- Automatic token usage tracking
- Support for structured output (JSON schema)
- Static utility methods for custom integrations

## API Reference

### OpenAIProvider

#### Constructor

```python
OpenAIProvider(client: OpenAI, model_name: str, parameters: Dict[str, Any], logger: Optional[Any] = None)
```

#### Static Methods

- `create(ai_config: AIConfigKind, logger: Optional[Any] = None) -> OpenAIProvider` - Factory method to create a provider from an AI config
- `get_ai_metrics_from_response(response: Any) -> LDAIMetrics` - Extract metrics from an OpenAI response

#### Instance Methods

- `invoke_model(messages: List[LDMessage]) -> ChatResponse` - Invoke the model with messages
- `invoke_structured_model(messages: List[LDMessage], response_structure: Dict[str, Any]) -> StructuredResponse` - Invoke the model with structured output
- `get_client() -> OpenAI` - Get the underlying OpenAI client

## License

Apache-2.0

