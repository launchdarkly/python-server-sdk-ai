# LaunchDarkly AI SDK Amazon Bedrock Provider

[![Actions Status](https://github.com/launchdarkly/python-server-sdk-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/launchdarkly/python-server-sdk-ai/actions/workflows/ci.yml)

[![PyPI](https://img.shields.io/pypi/v/launchdarkly-server-sdk-ai-bedrock.svg?maxAge=2592000)](https://pypi.org/project/launchdarkly-server-sdk-ai-bedrock/)
[![PyPI](https://img.shields.io/pypi/pyversions/launchdarkly-server-sdk-ai-bedrock.svg)](https://pypi.org/project/launchdarkly-server-sdk-ai-bedrock/)

> [!CAUTION]
> This package is in pre-release and not subject to backwards compatibility
> guarantees. The API may change based on feedback.
>
> Pin to a specific minor version and review the [changelog](CHANGELOG.md) before upgrading.

This package provides an Amazon Bedrock integration for the LaunchDarkly AI SDK.
Model completions use the Bedrock [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
via `boto3`. Single-turn agents are backed by the
[Strands Agents SDK](https://github.com/strands-agents/sdk-python), which is an
optional dependency.

## Installation

Install just the model runner:

```bash
pip install launchdarkly-server-sdk-ai-bedrock
```

Add the agent runner by installing the `agents` extra:

```bash
pip install "launchdarkly-server-sdk-ai-bedrock[agents]"
```

AWS credentials are read from the standard `boto3` credential chain
(environment variables, shared config, instance profile, etc.).

## Quick Start

```python
import asyncio
from ldclient import LDClient, Config, Context
from ldai import LDAIClient
from ldai.models import AICompletionConfigDefault, ModelConfig, ProviderConfig

# Initialize LaunchDarkly client
ld_client = LDClient(Config("your-sdk-key"))
ai_client = LDAIClient(ld_client)

context = Context.builder("user-123").build()

async def main():
    # Create a ManagedModel backed by the Bedrock provider
    model = await ai_client.create_model(
        "ai-config-key",
        context,
        AICompletionConfigDefault(
            enabled=True,
            model=ModelConfig("anthropic.claude-3-5-sonnet-20240620-v1:0"),
            provider=ProviderConfig("bedrock"),
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
LaunchDarkly AI config flag, selects the Bedrock runner automatically, and
returns a `ManagedModel` that wraps the runner:

```python
model = await ai_client.create_model("ai-config-key", context)

if model:
    result = await model.run("What is feature flagging?")
    print(result.content)
```

### Using the runner directly

If you need to construct a runner manually (e.g. for testing), you can use
`BedrockRunnerFactory` from the `ldai_bedrock` package:

```python
from ldai_bedrock import BedrockRunnerFactory

# Uses the default boto3 credential chain. Pass ``region_name=`` to pin a region
# or ``client=`` to supply a pre-built bedrock-runtime client.
factory = BedrockRunnerFactory()
runner = factory.create_model(ai_config)

result = await runner.run("Hello!")
print(result.content)
```

### Structured Output

Pass a JSON schema dict as `output_type` to request structured output. The
Bedrock Converse API does not have a first-class JSON-schema mode, so the
schema is included as a system instruction and the response text is parsed as
JSON:

```python
response_structure = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}

result = await runner.run("How do customers feel about flags?", output_type=response_structure)
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
    print(result.metrics.tokens)
```

### Async note

`boto3` is synchronous. The Bedrock model runner therefore dispatches each
`converse` call to a worker thread (`asyncio.to_thread`) so the event loop is
not blocked while AWS responds.

## Static Utility Methods

The `ldai_bedrock` helper module provides several utility functions:

### Converting Messages

```python
from ldai.models import LDMessage
from ldai_bedrock import convert_messages_to_bedrock

messages = [
    LDMessage(role="system", content="You are helpful."),
    LDMessage(role="user", content="Hello!"),
]

request_fragment = convert_messages_to_bedrock(messages)
# {"messages": [{"role": "user", "content": [{"text": "Hello!"}]}],
#  "system":   [{"text": "You are helpful."}]}
```

### Extracting Metrics

```python
from ldai_bedrock import get_ai_metrics_from_response

# After getting a response from bedrock-runtime.converse(...)
metrics = get_ai_metrics_from_response(response)
print(f"Success: {metrics.success}")
print(f"Tokens used: {metrics.tokens.total if metrics.tokens else 'N/A'}")
```

## Documentation

For full documentation, please refer to the [LaunchDarkly AI SDK documentation](https://docs.launchdarkly.com/sdk/ai/python).

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) in the repository root.

## License

Apache-2.0
