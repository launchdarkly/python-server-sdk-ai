# LaunchDarkly AI SDK - LangChain Provider

[![PyPI](https://img.shields.io/pypi/v/launchdarkly-server-sdk-ai-langchain.svg)](https://pypi.org/project/launchdarkly-server-sdk-ai-langchain/)

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
from ldai_langchain import LangChainProvider

# Initialize LaunchDarkly client
ld_client = LDClient(Config("your-sdk-key"))
ai_client = init(ld_client)

# Get AI configuration
context = Context.builder("user-123").build()
config = ai_client.config("ai-config-key", context, {})

async def main():
    # Create a LangChain provider from the AI configuration
    provider = await LangChainProvider.create(config)

    # Use the provider to invoke the model
    from ldai.models import LDMessage
    messages = [
        LDMessage(role="system", content="You are a helpful assistant."),
        LDMessage(role="user", content="Hello, how are you?"),
    ]
    
    response = await provider.invoke_model(messages)
    print(response.message.content)

asyncio.run(main())
```

## Usage

### Using LangChainProvider with the Create Factory

The simplest way to use the LangChain provider is with the static `create` factory method, which automatically creates the appropriate LangChain model based on your LaunchDarkly AI configuration:

```python
from ldai_langchain import LangChainProvider

# Create provider from AI configuration
provider = await LangChainProvider.create(ai_config)

# Invoke the model
response = await provider.invoke_model(messages)
```

### Using an Existing LangChain Model

If you already have a LangChain model configured, you can use it directly:

```python
from langchain_openai import ChatOpenAI
from ldai_langchain import LangChainProvider

# Create your own LangChain model
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Wrap it with LangChainProvider
provider = LangChainProvider(llm)

# Use with LaunchDarkly tracking
response = await provider.invoke_model(messages)
```

### Structured Output

The provider supports structured output using LangChain's `with_structured_output`:

```python
response_structure = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}

result = await provider.invoke_structured_model(messages, response_structure)
print(result.data)  # {"sentiment": "positive", "confidence": 0.95}
```

### Tracking Metrics

Use the provider with LaunchDarkly's tracking capabilities:

```python
# Get the AI config with tracker
config = ai_client.config("ai-config-key", context, {})

# Create provider
provider = await LangChainProvider.create(config)

# Track metrics automatically
async def invoke():
    return await provider.invoke_model(messages)

response = await config.tracker.track_metrics_of(
    invoke,
    lambda r: r.metrics
)
```

### Static Utility Methods

The `LangChainProvider` class provides several utility methods:

#### Converting Messages

```python
from ldai.models import LDMessage
from ldai_langchain import LangChainProvider

messages = [
    LDMessage(role="system", content="You are helpful."),
    LDMessage(role="user", content="Hello!"),
]

# Convert to LangChain messages
langchain_messages = LangChainProvider.convert_messages_to_langchain(messages)
```

#### Extracting Metrics

```python
from ldai_langchain import LangChainProvider

# After getting a response from LangChain
metrics = LangChainProvider.get_ai_metrics_from_response(ai_message)
print(f"Success: {metrics.success}")
print(f"Tokens used: {metrics.usage.total if metrics.usage else 'N/A'}")
```

#### Provider Name Mapping

```python
# Map LaunchDarkly provider names to LangChain provider names
langchain_provider = LangChainProvider.map_provider("gemini")  # Returns "google-genai"
```

## API Reference

### LangChainProvider

#### Constructor

```python
LangChainProvider(llm: BaseChatModel, logger: Optional[Any] = None)
```

#### Static Methods

- `create(ai_config: AIConfigKind, logger: Optional[Any] = None) -> LangChainProvider` - Factory method to create a provider from AI configuration
- `convert_messages_to_langchain(messages: List[LDMessage]) -> List[BaseMessage]` - Convert LaunchDarkly messages to LangChain messages
- `get_ai_metrics_from_response(response: AIMessage) -> LDAIMetrics` - Extract metrics from a LangChain response
- `map_provider(ld_provider_name: str) -> str` - Map LaunchDarkly provider names to LangChain names
- `create_langchain_model(ai_config: AIConfigKind) -> BaseChatModel` - Create a LangChain model from AI configuration

#### Instance Methods

- `invoke_model(messages: List[LDMessage]) -> ChatResponse` - Invoke the model with messages
- `invoke_structured_model(messages: List[LDMessage], response_structure: Dict[str, Any]) -> StructuredResponse` - Invoke with structured output
- `get_chat_model() -> BaseChatModel` - Get the underlying LangChain model

## Documentation

For full documentation, please refer to the [LaunchDarkly AI SDK documentation](https://docs.launchdarkly.com/sdk/ai/python).

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) in the repository root.

## License

Apache-2.0
