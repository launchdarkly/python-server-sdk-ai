# Changelog

## [0.3.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-openai-0.2.1...launchdarkly-server-sdk-ai-openai-0.3.0) (2026-04-02)


### ⚠ BREAKING CHANGES

* Bump minimum LangChain version to 1.0.0
* Extract shared utilities to openai_helper
* Extract shared utilities to langchain_helper
* Restructure provider factory and support additional create methods ([#102](https://github.com/launchdarkly/python-server-sdk-ai/issues/102))

### Features

* Add LangGraphAgentGraphRunner ([56ce0fd](https://github.com/launchdarkly/python-server-sdk-ai/commit/56ce0fd63b4301b58f33c17c55c4ecd47e9f8559))
* Add LDAIClient.create_agent() returning ManagedAgent ([53fd95e](https://github.com/launchdarkly/python-server-sdk-ai/commit/53fd95e2cfb66f4c53c6844cc41170077e6eee8c))
* Add ManagedAgentGraph support ([#111](https://github.com/launchdarkly/python-server-sdk-ai/issues/111)) ([56ce0fd](https://github.com/launchdarkly/python-server-sdk-ai/commit/56ce0fd63b4301b58f33c17c55c4ecd47e9f8559))
* Add ModelRunner ABC with invoke_model() and ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Add OpenAIAgentGraphRunner ([56ce0fd](https://github.com/launchdarkly/python-server-sdk-ai/commit/56ce0fd63b4301b58f33c17c55c4ecd47e9f8559))
* Add OpenAIAgentRunner with agentic tool-calling loop ([53fd95e](https://github.com/launchdarkly/python-server-sdk-ai/commit/53fd95e2cfb66f4c53c6844cc41170077e6eee8c))
* add optimization package stub ([872e81e](https://github.com/launchdarkly/python-server-sdk-ai/commit/872e81e29854ec03c434a32a287e9c94feb0b449))
* Adds optimization package stub ([58b7731](https://github.com/launchdarkly/python-server-sdk-ai/commit/58b7731aa4f0efbd42ff0b93760eb357cdfe219f))
* Bump minimum LangChain version to 1.0.0 ([dc592c5](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc592c5a2e2bf3bf679af14a9aa63e81678a69ab))
* Deprecated Chat object in favor of ManagedModel ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Deprecated create_chat(), use create_model() on the LDAIClient ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Drop support for python 3.9 ([#114](https://github.com/launchdarkly/python-server-sdk-ai/issues/114)) ([dc592c5](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc592c5a2e2bf3bf679af14a9aa63e81678a69ab))
* Extract shared utilities to langchain_helper ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Extract shared utilities to openai_helper ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Introduce ManagedAgent and AgentRunner implementations ([#110](https://github.com/launchdarkly/python-server-sdk-ai/issues/110)) ([53fd95e](https://github.com/launchdarkly/python-server-sdk-ai/commit/53fd95e2cfb66f4c53c6844cc41170077e6eee8c))
* Introduce ManagedModel and ModelRunner (PR-3) ([#104](https://github.com/launchdarkly/python-server-sdk-ai/issues/104)) ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Restructure provider factory and support additional create methods ([#102](https://github.com/launchdarkly/python-server-sdk-ai/issues/102)) ([e6e4907](https://github.com/launchdarkly/python-server-sdk-ai/commit/e6e49076b3b89b9bf4996d3ba8e4b4c9fb7b2078))

## [0.2.1](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-openai-0.2.0...launchdarkly-server-sdk-ai-openai-0.2.1) (2026-03-16)


### Bug Fixes

* Update comments for setting default ([#99](https://github.com/launchdarkly/python-server-sdk-ai/issues/99)) ([a14761d](https://github.com/launchdarkly/python-server-sdk-ai/commit/a14761d0f832e2fd07323f1f5322c76a1d7d7bf6))

## [0.2.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-openai-0.1.1...launchdarkly-server-sdk-ai-openai-0.2.0) (2026-03-09)


### ⚠ BREAKING CHANGES

* Rename default_value args to default

### Bug Fixes

* Make default optional with a disabled config default ([#97](https://github.com/launchdarkly/python-server-sdk-ai/issues/97)) ([39e09c6](https://github.com/launchdarkly/python-server-sdk-ai/commit/39e09c616bcb36af56983094039ee72a97bd1a19))
* Rename default_value args to default ([39e09c6](https://github.com/launchdarkly/python-server-sdk-ai/commit/39e09c616bcb36af56983094039ee72a97bd1a19))

## [0.1.1](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-openai-0.1.0...launchdarkly-server-sdk-ai-openai-0.1.1) (2026-02-23)


### Bug Fixes

* Update pre-release usage guidance ([#90](https://github.com/launchdarkly/python-server-sdk-ai/issues/90)) ([4f986c4](https://github.com/launchdarkly/python-server-sdk-ai/commit/4f986c4b4f74f001e5487892509129bdc9aa091c))

## 0.1.0 (2026-01-02)


### ⚠ BREAKING CHANGES

* Bump requirement of launchdarkly-server-sdk-ai to 0.12.0

### Features

* Add OpenAI provider package ([#78](https://github.com/launchdarkly/python-server-sdk-ai/issues/78)) ([ec2272e](https://github.com/launchdarkly/python-server-sdk-ai/commit/ec2272ef91203343f112e7262510117fc69207bd))


### Bug Fixes

* Bump requirement of launchdarkly-server-sdk-ai to 0.12.0 ([ec2272e](https://github.com/launchdarkly/python-server-sdk-ai/commit/ec2272ef91203343f112e7262510117fc69207bd))
