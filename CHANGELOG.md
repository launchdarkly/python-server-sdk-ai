# Changelog

All notable changes to the LaunchDarkly Python AI package will be documented in this file. This project adheres to [Semantic Versioning](http://semver.org).

## [0.9.1](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.9.0...0.9.1) (2025-06-26)


### Bug Fixes

* Fix Bedrock response parsing ([#50](https://github.com/launchdarkly/python-server-sdk-ai/issues/50)) ([df90bc2](https://github.com/launchdarkly/python-server-sdk-ai/commit/df90bc24c98b5a57bf944225774989b689b65d93))

## [0.9.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.8.1...0.9.0) (2025-06-26)


### ⚠ BREAKING CHANGES

* Drop support for Python 3.8 (eol 2024-10-07) ([#48](https://github.com/launchdarkly/python-server-sdk-ai/issues/48))

### Features

* Drop support for Python 3.8 (eol 2024-10-07) ([#48](https://github.com/launchdarkly/python-server-sdk-ai/issues/48)) ([8f94da6](https://github.com/launchdarkly/python-server-sdk-ai/commit/8f94da65d9019e17fbabe1d1f5ce1168ace5957e))

## [0.8.1](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.8.0...0.8.1) (2025-04-03)


### Documentation

* AI config --&gt; AI Config name change ([a123248](https://github.com/launchdarkly/python-server-sdk-ai/commit/a1232483bbd8a4a2212fb16e7e98ab6f6bdc8459))
* AI config --&gt; AI Config name change ([#42](https://github.com/launchdarkly/python-server-sdk-ai/issues/42)) ([bbdf1f6](https://github.com/launchdarkly/python-server-sdk-ai/commit/bbdf1f61fab77992a1d13207c81642d056e03f02))

## [0.8.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.7.0...0.8.0) (2025-02-07)


### Features

* Add variation version to metric data ([#39](https://github.com/launchdarkly/python-server-sdk-ai/issues/39)) ([1b07d08](https://github.com/launchdarkly/python-server-sdk-ai/commit/1b07d08743c409689c5a084df8c39fce2400d2dd))

## [0.7.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.6.0...0.7.0) (2025-01-23)


### Features

* Add ability to track time to first token for LDAIConfigTracker ([#37](https://github.com/launchdarkly/python-server-sdk-ai/issues/37)) ([b4a5757](https://github.com/launchdarkly/python-server-sdk-ai/commit/b4a5757ab7a1a8149891977cdfc25bdd4f7bba09))

## [0.6.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.5.0...0.6.0) (2024-12-17)


### ⚠ BREAKING CHANGES

* Unify tracking token to use only `TokenUsage` ([#32](https://github.com/launchdarkly/python-server-sdk-ai/issues/32))
* Change version_key_to variation_key ([#29](https://github.com/launchdarkly/python-server-sdk-ai/issues/29))

### Features

* Add `LDAIConfigTracker.get_summary` method ([#31](https://github.com/launchdarkly/python-server-sdk-ai/issues/31)) ([e425b1f](https://github.com/launchdarkly/python-server-sdk-ai/commit/e425b1f9e7bf27ab195b877e62af48012eb601c1))
* Add `track_error` to mirror `track_success` ([#33](https://github.com/launchdarkly/python-server-sdk-ai/issues/33)) ([404f704](https://github.com/launchdarkly/python-server-sdk-ai/commit/404f704dd38f4fc15c718e3dc1027efbda5f36b6))


### Bug Fixes

* Unify tracking token to use only `TokenUsage` ([#32](https://github.com/launchdarkly/python-server-sdk-ai/issues/32)) ([80e1845](https://github.com/launchdarkly/python-server-sdk-ai/commit/80e18452a936356937660eabe7a186beae4d17bd))


### Code Refactoring

* Change version_key_to variation_key ([#29](https://github.com/launchdarkly/python-server-sdk-ai/issues/29)) ([fcc720a](https://github.com/launchdarkly/python-server-sdk-ai/commit/fcc720a101c97ccb92fd95509b3e7819d557dde5))

## [0.5.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.4.0...0.5.0) (2024-12-09)


### ⚠ BREAKING CHANGES

* Rename model and provider id to name ([#27](https://github.com/launchdarkly/python-server-sdk-ai/issues/27))

### Code Refactoring

* Rename model and provider id to name ([#27](https://github.com/launchdarkly/python-server-sdk-ai/issues/27)) ([65260b6](https://github.com/launchdarkly/python-server-sdk-ai/commit/65260b621acee07b38e9ebaeb4a10c1e4c9db794))

## [0.4.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.3.0...0.4.0) (2024-12-02)


### Features

* Return AIConfig and LDAITracker separately [#23](https://github.com/launchdarkly/python-server-sdk-ai/issues/23) ([96f888f](https://github.com/launchdarkly/python-server-sdk-ai/commit/96f888f50503cc2e9e2c30bf1c21f80a2773c8b5))


### Bug Fixes

* Fix context usage for message interpolation ([#24](https://github.com/launchdarkly/python-server-sdk-ai/issues/24)) ([1159aee](https://github.com/launchdarkly/python-server-sdk-ai/commit/1159aeeda7c46cf2dab93f209929dbad5d35dc80))

## [0.3.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.2.0...0.3.0) (2024-11-22)


### ⚠ BREAKING CHANGES

* Rename nest parameters under model
* Change modelId to id
* Remove max_tokens and temperature as top level model config keys
* Suffix track methods with metrics
* Rename `LDAIClient.model_config` to `LDAIClient.config`
* Rename prompt to messages

### Features

* Add custom parameter support to model config ([95015f1](https://github.com/launchdarkly/python-server-sdk-ai/commit/95015f1f29b4ddf0acc2f22b72a5c0c4241fd3f3))
* Add support for provider config ([d2a2ea7](https://github.com/launchdarkly/python-server-sdk-ai/commit/d2a2ea7a16159de5c11484114ad4a7ae6369f9c6))


### Bug Fixes

* Change modelId to id ([9564780](https://github.com/launchdarkly/python-server-sdk-ai/commit/9564780ea2b919d456431e3309b73156f8e9817d))
* Remove max_tokens and temperature as top level model config keys ([55f34fe](https://github.com/launchdarkly/python-server-sdk-ai/commit/55f34fec9410124d24318feadada9e087e7d4cb8))
* Rename `LDAIClient.model_config` to `LDAIClient.config` ([3a3e913](https://github.com/launchdarkly/python-server-sdk-ai/commit/3a3e913d9e1586278d9fe6228f79f6748cbbd605))
* Rename nest parameters under model ([a2cc966](https://github.com/launchdarkly/python-server-sdk-ai/commit/a2cc9662bdc526f0b6a3a271a4b4f46b95d0ec2f))
* Rename prompt to messages ([9a86f0a](https://github.com/launchdarkly/python-server-sdk-ai/commit/9a86f0af9322baf71d7ddddb6115d585582cfc86))
* Suffix track methods with metrics ([319f64d](https://github.com/launchdarkly/python-server-sdk-ai/commit/319f64da54815854163d663022fdffc274c2059a))

## [0.2.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.1.1...0.2.0) (2024-11-21)


### Features

* Change the typing for the AIConfig ([#16](https://github.com/launchdarkly/python-server-sdk-ai/issues/16)) ([daf9537](https://github.com/launchdarkly/python-server-sdk-ai/commit/daf95372328f1b1e4e9e27333498642136f43838))


### Bug Fixes

* Change linting tools to dev-only dependency ([#15](https://github.com/launchdarkly/python-server-sdk-ai/issues/15)) ([c752739](https://github.com/launchdarkly/python-server-sdk-ai/commit/c752739d1c34cbf7f78cc3f89c37a688671c7366))
* Verify prompt payload exists before accessing it ([#19](https://github.com/launchdarkly/python-server-sdk-ai/issues/19)) ([d9dd21f](https://github.com/launchdarkly/python-server-sdk-ai/commit/d9dd21f2189de62eac70ad9db3755e4a2cf36511))

## [0.1.1](https://github.com/launchdarkly/python-server-sdk-ai/compare/0.1.0...0.1.1) (2024-11-08)


### Bug Fixes

* Update how prompt is handled ([#12](https://github.com/launchdarkly/python-server-sdk-ai/issues/12)) ([050cc71](https://github.com/launchdarkly/python-server-sdk-ai/commit/050cc71dde52db3174153a0c9c08021580530833))

## [0.1.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/v0.1.0...0.1.0) (2024-11-08)

Initial release of the LaunchDarkly Python AI package. To learn more about the LaunchDarkly Python AI SDK, see the [Python AI SDK documentation](https://docs.launchdarkly.com/sdk/ai/python).
