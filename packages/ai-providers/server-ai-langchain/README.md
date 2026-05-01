# LaunchDarkly AI SDK - LangChain Provider

[![Actions Status](https://github.com/launchdarkly/python-server-sdk-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/launchdarkly/python-server-sdk-ai/actions/workflows/ci.yml)

[![PyPI](https://img.shields.io/pypi/v/launchdarkly-server-sdk-ai-langchain.svg?maxAge=2592000)](https://pypi.org/project/launchdarkly-server-sdk-ai-langchain/)
[![PyPI](https://img.shields.io/pypi/pyversions/launchdarkly-server-sdk-ai-langchain.svg)](https://pypi.org/project/launchdarkly-server-sdk-ai-langchain/)

> [!CAUTION]
> This package is in pre-release and not subject to backwards compatibility
> guarantees. The API may change based on feedback.
>
> Pin to a specific minor version and review the [changelog](CHANGELOG.md) before upgrading.

This package provides LangChain integration for the LaunchDarkly Server-Side AI SDK, allowing you to use LangChain models and chains with LaunchDarkly's tracking and configuration capabilities.

## Installation

```bash
pip install launchdarkly-server-sdk-ai-langchain
```

You'll also need to install the LangChain provider packages for the models you want to use:

```bash
# For OpenAI
pip install langchain-openai

# For Anthropic
pip install langchain-anthropic

# For Google
pip install langchain-google-genai
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
    # Create a ManagedModel backed by the LangChain provider
    model = await ai_client.create_model(
        "ai-config-key",
        context,
        AICompletionConfigDefault(
            enabled=True,
            model=ModelConfig("gpt-4"),
            provider=ProviderConfig("langchain"),
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
LaunchDarkly AI config flag, selects the LangChain runner automatically, and
returns a `ManagedModel` that wraps the runner:

```python
model = await ai_client.create_model("ai-config-key", context)

if model:
    result = await model.run("What is feature flagging?")
    print(result.content)
```

### Using the runner directly

If you need to construct a runner manually (e.g. for testing), you can use
`LangChainRunnerFactory` from the `ldai_langchain` package:

```python
from langchain_openai import ChatOpenAI
from ldai_langchain import LangChainRunnerFactory

llm = ChatOpenAI(model="gpt-4", temperature=0.7)
runner = LangChainModelRunner(llm)

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

The `ldai_langchain` helper module provides several utility functions:

#### Converting Messages

```python
from ldai.models import LDMessage
from ldai_langchain.langchain_helper import convert_messages_to_langchain

messages = [
    LDMessage(role="system", content="You are helpful."),
    LDMessage(role="user", content="Hello!"),
]

langchain_messages = convert_messages_to_langchain(messages)
```

#### Extracting Metrics

```python
from ldai_langchain.langchain_helper import get_ai_metrics_from_response

# After getting a response from LangChain
metrics = get_ai_metrics_from_response(ai_message)
print(f"Success: {metrics.success}")
print(f"Tokens used: {metrics.usage.total if metrics.usage else 'N/A'}")
```

#### Provider Name Mapping

```python
from ldai_langchain.langchain_helper import map_provider_name

# Map LaunchDarkly provider names to LangChain provider names
langchain_provider = map_provider_name("gemini")  # Returns "google-genai"
```

## API Reference

### LangChainModelRunner

`LangChainModelRunner` implements the `Runner` protocol for LangChain chat models.

#### Constructor

```python
LangChainModelRunner(llm: BaseChatModel)
```

#### Methods

- `run(input, output_type=None) -> RunnerResult` — Run the model with a string prompt or list of `LDMessage` objects. Pass `output_type` (JSON schema dict) for structured output.
- `get_llm() -> BaseChatModel` — Return the underlying LangChain model.

### LangChainAgentRunner

`LangChainAgentRunner` implements the `Runner` protocol for compiled LangChain agent graphs.

#### Constructor

```python
LangChainAgentRunner(agent: Any)
```

#### Methods

- `run(input, output_type=None) -> RunnerResult` — Run the agent with the given input. Returns `RunnerResult` with `content`, `metrics` (including `tool_calls`), and `raw`.
- `get_agent() -> Any` — Return the underlying compiled agent graph.

## Documentation

For full documentation, please refer to the [LaunchDarkly AI SDK documentation](https://docs.launchdarkly.com/sdk/ai/python).

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) in the repository root.

## License

Apache-2.0
