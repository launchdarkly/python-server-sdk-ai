# LaunchDarkly AI SDK - LangChain Provider

This package provides LangChain provider support for the LaunchDarkly AI SDK.

## Installation

```bash
pip install launchdarkly-server-sdk-ai-langchain
```

This will automatically install the core SDK (`launchdarkly-server-sdk-ai`) and LangChain dependencies.

## Usage

```python
from ldclient import init, Context
from ldai import init_ai

# Initialize clients
ld_client = init('your-sdk-key')
ai_client = init_ai(ld_client)

# Create a chat - will automatically use LangChain provider
context = Context.create('user-key')
chat = await ai_client.create_chat('chat-config', context, {
    'enabled': True,
    'provider': {'name': 'openai'},
    'model': {'name': 'gpt-4'}
})

if chat:
    response = await chat.invoke('Hello!')
    print(response.message.content)
```

## Supported LangChain Providers

This provider supports any LangChain-compatible model, including:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic (Claude)
- Google (Gemini)
- And many more through LangChain integrations

## Requirements

- Python 3.9+
- launchdarkly-server-sdk-ai >= 0.10.1
- langchain >= 0.3.0
- langchain-core >= 0.3.0

## Documentation

For full documentation, visit: https://docs.launchdarkly.com/sdk/ai/python

## License

Apache-2.0

