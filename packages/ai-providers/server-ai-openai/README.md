# LaunchDarkly AI SDK OpenAI Provider

[![Actions Status](https://github.com/launchdarkly/python-server-sdk-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/launchdarkly/python-server-sdk-ai/actions/workflows/ci.yml)

[![PyPI](https://img.shields.io/pypi/v/launchdarkly-server-sdk-ai-openai.svg?maxAge=2592000)](https://pypi.org/project/launchdarkly-server-sdk-ai-openai/)
[![PyPI](https://img.shields.io/pypi/pyversions/launchdarkly-server-sdk-ai-openai.svg)](https://pypi.org/project/launchdarkly-server-sdk-ai-openai/)

> [!CAUTION]
> This package is in pre-release and not subject to backwards compatibility
> guarantees. The API may change based on feedback.
>
> Pin to a specific minor version and review the [changelog](CHANGELOG.md) before upgrading.

This package provides an OpenAI integration for the LaunchDarkly AI SDK.

## Installation

```bash
pip install launchdarkly-server-sdk-ai-openai
```

## Quick Start

```python
import asyncio
from ldclient import LDClient, Config, Context
from ldai import init
from ldai.models import AICompletionConfigDefault, ModelConfig, ProviderConfig

# Initialize LaunchDarkly client
ld_client = LDClient(Config("your-sdk-key"))
ai_client = init(ld_client)

context = Context.builder("user-123").build()

async def main():
    # Create a ManagedModel backed by the OpenAI provider
    model = await ai_client.create_model(
        "ai-config-key",
        context,
        AICompletionConfigDefault(
            enabled=True,
            model=ModelConfig("gpt-4"),
            provider=ProviderConfig("openai"),
        ),
    )

    if model:
        result = await model.run("Hello, how are you?")
        print(result.content)

asyncio.run(main())
```

## Usage

### Using `create_model` (recommended)

The recommended entry point is `LDAIClient.create_model`, which evaluates a
LaunchDarkly AI config flag, selects the OpenAI runner automatically, and
returns a `ManagedModel` that wraps the runner:

```python
model = await ai_client.create_model("ai-config-key", context)

if model:
    result = await model.run("What is feature flagging?")
    print(result.content)
```

### Using the runner directly

If you need to construct a runner manually (e.g. for testing), you can use
`OpenAIRunnerFactory` from the `ldai_openai` package:

```python
from ldai_openai import OpenAIRunnerFactory

factory = OpenAIRunnerFactory()  # uses OPENAI_API_KEY from environment
runner = factory.create_model(ai_config)

result = await runner.run("Hello!")
print(result.content)
```

### Structured Output

Pass a JSON schema dict as `output_type` to request structured output:

```python
response_structure = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}

result = await runner.run(messages, output_type=response_structure)
print(result.parsed)  # {"sentiment": "positive", "confidence": 0.95}
```

### Tracking Metrics

`ManagedModel.run()` automatically tracks metrics via the associated
`LDAIConfigTracker`. For manual tracking, use the tracker directly:

```python
model = await ai_client.create_model("ai-config-key", context)

if model:
    result = await model.run("Explain feature flags.")
    # Metrics are tracked automatically; access them via result.metrics
    print(result.metrics.usage)
```

### Static Utility Methods

The `ldai_openai` helper module provides several utility functions:

#### Converting Messages

```python
from ldai.models import LDMessage
from ldai_openai import convert_messages_to_openai

messages = [
    LDMessage(role="system", content="You are helpful."),
    LDMessage(role="user", content="Hello!"),
]

openai_messages = convert_messages_to_openai(messages)
```

#### Extracting Metrics

```python
from ldai_openai import get_ai_metrics_from_response

# After getting a response from OpenAI
metrics = get_ai_metrics_from_response(response)
print(f"Success: {metrics.success}")
print(f"Tokens used: {metrics.usage.total if metrics.usage else 'N/A'}")
```

## Documentation

For full documentation, please refer to the [LaunchDarkly AI SDK documentation](https://docs.launchdarkly.com/sdk/ai/python).

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) in the repository root.

## License

Apache-2.0
