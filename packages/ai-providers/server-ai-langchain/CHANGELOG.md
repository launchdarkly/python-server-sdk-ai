# Changelog

All notable changes to the LaunchDarkly Python AI LangChain provider package will be documented in this file. This project adheres to [Semantic Versioning](http://semver.org).

## [0.4.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-langchain-0.3.2...launchdarkly-server-sdk-ai-langchain-0.4.0) (2026-04-02)


### ⚠ BREAKING CHANGES

* Bump minimum LangChain version to 1.0.0
* Restructure provider factory and support additional create methods ([#102](https://github.com/launchdarkly/python-server-sdk-ai/issues/102))
* Extract shared utilities to langchain_helper

### Features

* Add LangGraphAgentGraphRunner ([56ce0fd](https://github.com/launchdarkly/python-server-sdk-ai/commit/56ce0fd63b4301b58f33c17c55c4ecd47e9f8559))
* Extract shared utilities to langchain_helper ([453c71c](https://github.com/launchdarkly/python-server-sdk-ai/commit/453c71c84adcc6b8a3e316a98907dcb511bc9d41))
* Add get_ai_usage_from_response to langchain_helper ([4fab18f](https://github.com/launchdarkly/python-server-sdk-ai/commit/4fab18fa62375b6c97cb12a89225805c81ca4ee8))
* Add get_tool_calls_from_response and sum_token_usage_from_messages to langchain_helper ([4fab18f](https://github.com/launchdarkly/python-server-sdk-ai/commit/4fab18fa62375b6c97cb12a89225805c81ca4ee8))
* Drop support for python 3.9 ([#114](https://github.com/launchdarkly/python-server-sdk-ai/issues/114)) ([dc592c5](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc592c5a2e2bf3bf679af14a9aa63e81678a69ab))


### Bug Fixes

* Use time.perf_counter_ns() for sub-millisecond precision in duration calculations ([4fab18f](https://github.com/launchdarkly/python-server-sdk-ai/commit/4fab18fa62375b6c97cb12a89225805c81ca4ee8))

## [0.3.2](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-langchain-0.3.1...launchdarkly-server-sdk-ai-langchain-0.3.2) (2026-03-16)


### Bug Fixes

* Improve metric token collection for Judge evaluations when using LangChain ([f951dac](https://github.com/launchdarkly/python-server-sdk-ai/commit/f951daccd569a0b5ff598f571d105a7d244939d1))
* Improved raw response when performing Judge evaluations using LangChain ([f951dac](https://github.com/launchdarkly/python-server-sdk-ai/commit/f951daccd569a0b5ff598f571d105a7d244939d1))
* Update comments for setting default ([#99](https://github.com/launchdarkly/python-server-sdk-ai/issues/99)) ([a14761d](https://github.com/launchdarkly/python-server-sdk-ai/commit/a14761d0f832e2fd07323f1f5322c76a1d7d7bf6))

## [0.3.1](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-langchain-0.3.0...launchdarkly-server-sdk-ai-langchain-0.3.1) (2026-02-23)


### Bug Fixes

* Update pre-release usage guidance ([#90](https://github.com/launchdarkly/python-server-sdk-ai/issues/90)) ([4f986c4](https://github.com/launchdarkly/python-server-sdk-ai/commit/4f986c4b4f74f001e5487892509129bdc9aa091c))

## [0.3.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-langchain-0.2.0...launchdarkly-server-sdk-ai-langchain-0.3.0) (2026-01-02)


### ⚠ BREAKING CHANGES

* Use client provided logger and remove optional parameter ([#81](https://github.com/launchdarkly/python-server-sdk-ai/issues/81))

### Bug Fixes

* Use client provided logger and remove optional parameter ([#81](https://github.com/launchdarkly/python-server-sdk-ai/issues/81)) ([fd4cb37](https://github.com/launchdarkly/python-server-sdk-ai/commit/fd4cb37313fefacdf109148116aa3df49cfe0d3c))

## [0.2.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/launchdarkly-server-sdk-ai-langchain-0.1.0...launchdarkly-server-sdk-ai-langchain-0.2.0) (2025-12-19)


### Features

* Add langchain provider package ([#70](https://github.com/launchdarkly/python-server-sdk-ai/issues/70)) ([70c794f](https://github.com/launchdarkly/python-server-sdk-ai/commit/70c794fe7b2e06e775a471fb6dcf1f4c03789409))
