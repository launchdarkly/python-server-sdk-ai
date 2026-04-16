# Changelog

All notable changes to the LaunchDarkly Python AI OpenAI provider package will be documented in this file. This project adheres to [Semantic Versioning](http://semver.org).

## [0.4.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-openai-0.3.0...launchdarkly-server-sdk-ai-openai-0.4.0) (2026-04-16)


### ⚠ BREAKING CHANGES

* Move graph_key to AIConfigTracker instantiation ([#134](https://github.com/launchdarkly/python-server-sdk-ai/issues/134))

### Features

* Move graph_key to AIConfigTracker instantiation ([#134](https://github.com/launchdarkly/python-server-sdk-ai/issues/134)) ([20fff24](https://github.com/launchdarkly/python-server-sdk-ai/commit/20fff24fcd02aa101d7f9a6c21dc6a25e7916a1c))

## [0.3.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-openai-0.2.1...launchdarkly-server-sdk-ai-openai-0.3.0) (2026-04-02)


### ⚠ BREAKING CHANGES

* Restructure provider factory and support additional create methods ([#102](https://github.com/launchdarkly/python-server-sdk-ai/issues/102))
* Extract shared utilities to openai_helper

### Features

* Add OpenAIAgentRunner with agentic tool-calling loop ([53fd95e](https://github.com/launchdarkly/python-server-sdk-ai/commit/53fd95e2cfb66f4c53c6844cc41170077e6eee8c))
* Add OpenAIAgentGraphRunner ([56ce0fd](https://github.com/launchdarkly/python-server-sdk-ai/commit/56ce0fd63b4301b58f33c17c55c4ecd47e9f8559))
* Extract shared utilities to openai_helper ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Add get_ai_usage_from_response to openai_helper ([4fab18f](https://github.com/launchdarkly/python-server-sdk-ai/commit/4fab18fa62375b6c97cb12a89225805c81ca4ee8))
* Drop support for python 3.9 ([#114](https://github.com/launchdarkly/python-server-sdk-ai/issues/114)) ([dc592c5](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc592c5a2e2bf3bf679af14a9aa63e81678a69ab))

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
