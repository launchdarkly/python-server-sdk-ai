# LaunchDarkly Server-Side AI SDK for Python

This package contains the LaunchDarkly Server-Side AI SDK for Python (`launchdarkly-server-sdk-ai`).

## Installation

```bash
pip install launchdarkly-server-sdk-ai
```

## Quick Start

```python
from ldclient import LDClient, Config, Context
from ldai import LDAIClient, AICompletionConfigDefault, ModelConfig

# Initialize LaunchDarkly client
ld_client = LDClient(Config("your-sdk-key"))

# Create AI client
ai_client = LDAIClient(ld_client)

# Get AI configuration
context = Context.create("user-123")
config = ai_client.completion_config(
    "my-ai-config",
    context,
    AICompletionConfigDefault(
        enabled=True,
        model=ModelConfig("gpt-4")
    )
)

# Use the configuration with your AI provider
if config.enabled:
    # Your AI implementation here
    pass
```

## Documentation

For full documentation, please refer to the [LaunchDarkly AI SDK documentation](https://docs.launchdarkly.com/sdk/ai/python).

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) in the repository root.

## License

Apache-2.0
