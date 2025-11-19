# LaunchDarkly Server-side AI SDK for Python - Monorepo

This repository contains the LaunchDarkly AI SDK for Python and its provider packages.

## Packages

### Core SDK
**Package:** [`launchdarkly-server-sdk-ai`](./packages/core/)  
**PyPI:** https://pypi.org/project/launchdarkly-server-sdk-ai/

The core LaunchDarkly AI SDK providing:
- AI configuration management
- Tracking and metrics
- Provider abstraction layer
- Chat management

```bash
pip install launchdarkly-server-sdk-ai
```

### LangChain Provider
**Package:** [`launchdarkly-server-sdk-ai-langchain`](./packages/langchain/)  
**PyPI:** https://pypi.org/project/launchdarkly-server-sdk-ai-langchain/

LangChain provider supporting multiple AI providers through LangChain's unified interface.

```bash
pip install launchdarkly-server-sdk-ai-langchain
```

## Installation

### Basic Installation
```bash
# Install core SDK
pip install launchdarkly-server-sdk-ai

# Install with LangChain provider
pip install launchdarkly-server-sdk-ai-langchain
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/launchdarkly/python-server-sdk-ai.git
cd python-server-sdk-ai

# Install core package
cd packages/core
poetry install

# Install langchain package (in separate terminal/session)
cd packages/langchain
poetry install
```

## Usage

```python
from ldclient import init, Context
from ldai import init_ai

# Initialize
ld_client = init('your-sdk-key')
ai_client = init_ai(ld_client)

# Create a chat (automatically uses installed providers)
context = Context.create('user-key')
chat = await ai_client.create_chat('chat-config', context)

if chat:
    response = await chat.invoke('Hello!')
    print(response.message.content)
```

## Documentation

- [SDK Reference Guide](https://docs.launchdarkly.com/sdk/ai/python)
- [API Documentation](https://launchdarkly-python-sdk-ai.readthedocs.io/)
- [Core Package README](./packages/core/README.md)
- [LangChain Provider README](./packages/langchain/README.md)

## Repository Structure

```
python-server-sdk-ai/
├── packages/
│   ├── core/                    # Core SDK
│   │   ├── ldai/               # Main SDK code
│   │   ├── pyproject.toml
│   │   └── README.md
│   └── langchain/               # LangChain provider
│       ├── ldai/
│       │   └── providers/
│       │       └── langchain/
│       ├── pyproject.toml
│       └── README.md
├── .github/
│   └── workflows/               # CI/CD workflows
├── release-please-config.json   # Multi-package release config
└── .release-please-manifest.json # Version tracking
```

## Publishing

Each package is published independently to PyPI:
- Core: `launchdarkly-server-sdk-ai`
- LangChain: `launchdarkly-server-sdk-ai-langchain`

Releases are managed automatically via Release Please when changes are merged to `main`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing

```bash
# Test core package
cd packages/core
poetry run pytest

# Test langchain package
cd packages/langchain
poetry run pytest
```

## License

Apache-2.0. See [LICENSE.txt](LICENSE.txt)
