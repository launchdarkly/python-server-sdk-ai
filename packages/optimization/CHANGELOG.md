# Changelog

## [0.2.1](https://github.com/launchdarkly/python-server-sdk-ai/compare/ldai_optimizer-0.2.0...ldai_optimizer-0.2.1) (2026-07-20)


### Bug Fixes

* **optimization:** wrong install path in readme ([4f73464](https://github.com/launchdarkly/python-server-sdk-ai/commit/4f7346415cb7a8321df851f74a00fc4731933249))

## [0.2.0](https://github.com/launchdarkly/python-server-sdk-ai/compare/ldai_optimizer-0.1.0...ldai_optimizer-0.2.0) (2026-07-17)


### ⚠ BREAKING CHANGES

* Bump minimum LangChain version to 1.0.0

### Features

* ability to specify variation key on config or from_options for optimization package ([c5e39eb](https://github.com/launchdarkly/python-server-sdk-ai/commit/c5e39eba8f36d97c2d8c907544fd61988b1cb0c1))
* ability to specify variation key on config or from_options for optimization package ([#162](https://github.com/launchdarkly/python-server-sdk-ai/issues/162)) ([f0c4612](https://github.com/launchdarkly/python-server-sdk-ai/commit/f0c46124e06bff4cc7b80bf109661331db0f5c56))
* add auto-commit option ([4cb8859](https://github.com/launchdarkly/python-server-sdk-ai/commit/4cb8859d2b485bb34bee758b1db1c95a7951778d))
* add optimization for duration ([5d76276](https://github.com/launchdarkly/python-server-sdk-ai/commit/5d762764fa6d8dbcce637fe6ef79cfbf35932aac))
* Add optimization package stub ([#109](https://github.com/launchdarkly/python-server-sdk-ai/issues/109)) ([ebd5166](https://github.com/launchdarkly/python-server-sdk-ai/commit/ebd5166d86c2d58e4c2fcc0b3fcc983eb49574e6))
* add shared dataclass for calls so they can be handled by same handler ([31c8385](https://github.com/launchdarkly/python-server-sdk-ai/commit/31c838587aa239604d37a1c2acef3808c27f4ed3))
* add token limit handling ([e8c6692](https://github.com/launchdarkly/python-server-sdk-ai/commit/e8c6692a334cee28ba02aff68b3c30ed5d4965a0))
* adds ability to optimize for cost ([94de596](https://github.com/launchdarkly/python-server-sdk-ai/commit/94de596d4391d6ba9f5dd36a87300b0eaafa5e4f))
* adds ability to optimize for cost ([#172](https://github.com/launchdarkly/python-server-sdk-ai/issues/172)) ([3b4baa3](https://github.com/launchdarkly/python-server-sdk-ai/commit/3b4baa326af85c8d0f0ae95ac5dfc8068f6f3ec0))
* adds ability to use inverted judges ([0ed243e](https://github.com/launchdarkly/python-server-sdk-ai/commit/0ed243e3845a0520ce82e7e4330f7d88130d5e00))
* adds ability to use inverted judges ([#168](https://github.com/launchdarkly/python-server-sdk-ai/issues/168)) ([23baeb4](https://github.com/launchdarkly/python-server-sdk-ai/commit/23baeb4b2c5ae27c7962afdcf5b764d0c4379513))
* Adds optimization package stub ([58b7731](https://github.com/launchdarkly/python-server-sdk-ai/commit/58b7731aa4f0efbd42ff0b93760eb357cdfe219f))
* Adds optimization package stub ([cc85a05](https://github.com/launchdarkly/python-server-sdk-ai/commit/cc85a05f0e81acdc33437b51238d67ddf8a92b80))
* adds reporting for cost and latency optimization failures ([365fa94](https://github.com/launchdarkly/python-server-sdk-ai/commit/365fa94cb0b5ba62097c038e5cf70b7265c68e14))
* adds reporting for cost and latency optimization failures ([#180](https://github.com/launchdarkly/python-server-sdk-ai/issues/180)) ([d267832](https://github.com/launchdarkly/python-server-sdk-ai/commit/d267832116d005ec94e795bdfd447cfc1d21d843))
* all logs -&gt; debug ([2fd55e2](https://github.com/launchdarkly/python-server-sdk-ai/commit/2fd55e2b02e4f7b2a9b772655409d5eaadba3593))
* Bump minimum LangChain version to 1.0.0 ([dc592c5](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc592c5a2e2bf3bf679af14a9aa63e81678a69ab))
* Drop support for python 3.9 ([#114](https://github.com/launchdarkly/python-server-sdk-ai/issues/114)) ([dc592c5](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc592c5a2e2bf3bf679af14a9aa63e81678a69ab))
* dx improvements for optimization package ([7074cfa](https://github.com/launchdarkly/python-server-sdk-ai/commit/7074cfa666723a2abf84eb901a94407e18751fc7))
* ground truth optimization path ([44c8c59](https://github.com/launchdarkly/python-server-sdk-ai/commit/44c8c59434d9f9f38251e636ec091cb7f91abaff))
* implement ability to use completions or agents for judge calls ([ea596a7](https://github.com/launchdarkly/python-server-sdk-ai/commit/ea596a7ec84733b9198b475cfa81b743b61feb9b))
* implement additional config optimization api fields ([#196](https://github.com/launchdarkly/python-server-sdk-ai/issues/196)) ([4ed30d8](https://github.com/launchdarkly/python-server-sdk-ai/commit/4ed30d81de5685e0789601e8f6d4e561b5692f9e))
* implement latency & token tracking for optimizations ([288336e](https://github.com/launchdarkly/python-server-sdk-ai/commit/288336ef9a2c7db5fd0cf27c44418bb9a0023441))
* Implement optimization code paths and functionality for initial release ([#140](https://github.com/launchdarkly/python-server-sdk-ai/issues/140)) ([5204c47](https://github.com/launchdarkly/python-server-sdk-ai/commit/5204c47a07404be60115e3167dc9fc3716e0e5c7))
* implementation of agent optimization + tests ([1712e4f](https://github.com/launchdarkly/python-server-sdk-ai/commit/1712e4f687a0d1925a16758fe969b252ed70399c))
* implements LD API client, optimize_from_config path ([2fecd54](https://github.com/launchdarkly/python-server-sdk-ai/commit/2fecd54c6e567fdf52dee91b8d67953ba0499809))
* implements optimize method in SDK, code moved ([9859d08](https://github.com/launchdarkly/python-server-sdk-ai/commit/9859d087f53955e38fb085c5a33a8643bcee7acb))
* partially implement optimize_from_config ([d3e1f96](https://github.com/launchdarkly/python-server-sdk-ai/commit/d3e1f966ede487ed2315c80e5878571be8137526))
* prevent overfitting via prompt changes and post-processing ([8f9f1e2](https://github.com/launchdarkly/python-server-sdk-ai/commit/8f9f1e2fd486002f3f5479fc026bee2e23d2282f))
* use judge_passed for all calcs ([a8f14de](https://github.com/launchdarkly/python-server-sdk-ai/commit/a8f14de3b7601f22ffb6af57e79a2730ce2bc5c7))


### Bug Fixes

* address cursor feedback ([f2f0894](https://github.com/launchdarkly/python-server-sdk-ai/commit/f2f0894625d763a25006d95bea9ea45780e623a4))
* adjust iteration logic so validation doesn't consume them ([3042984](https://github.com/launchdarkly/python-server-sdk-ai/commit/304298411d482ba074c461d4b3493ce714340024))
* better handling of params and custom params for optimization ([572a2aa](https://github.com/launchdarkly/python-server-sdk-ai/commit/572a2aa35a39e3eb904005e32e28214855f5883a))
* better handling of params and custom params for optimization ([#163](https://github.com/launchdarkly/python-server-sdk-ai/issues/163)) ([92f51fa](https://github.com/launchdarkly/python-server-sdk-ai/commit/92f51faf16cc588ef4cee9c2dd1d251f30b9867a))
* consistency with other makefiles ([b9a5601](https://github.com/launchdarkly/python-server-sdk-ai/commit/b9a560110b9ef1746b1b1cff2b50ea8b90297acd))
* don't only evaluate final input in GT results ([53f455f](https://github.com/launchdarkly/python-server-sdk-ai/commit/53f455f3fb1073dcc1caeead4ccc7b51153eabb4))
* don't only evaluate final input in GT results ([9bedf9e](https://github.com/launchdarkly/python-server-sdk-ai/commit/9bedf9ef91067f3122b904cf5cc78e87b8929151))
* ensure cost data is persisted ([4eb0bb0](https://github.com/launchdarkly/python-server-sdk-ai/commit/4eb0bb0c634e40539d9ae4b7f125eb6729c28c74))
* fix for single-iteration optimizations ([96eadb7](https://github.com/launchdarkly/python-server-sdk-ai/commit/96eadb70c6395d5451c9c6bb30da96223e109c39))
* fix for single-iteration optimizations ([#209](https://github.com/launchdarkly/python-server-sdk-ai/issues/209)) ([ba5aa63](https://github.com/launchdarkly/python-server-sdk-ai/commit/ba5aa63b99048c75da3e82acb8ac3655c91153da))
* lint ([aee6aa7](https://github.com/launchdarkly/python-server-sdk-ai/commit/aee6aa744468e0be20d2874dfe5701a3583a8050))
* lint + missed variable rename ([f8e5509](https://github.com/launchdarkly/python-server-sdk-ai/commit/f8e55092fa9b06cd35b90fcd2979a2c84b042e7a))
* lints + structured output tool rename ([8481690](https://github.com/launchdarkly/python-server-sdk-ai/commit/8481690903e931b67c5ae5ca5fbadd10a6b4102a))
* move failure forward ([3c65a6e](https://github.com/launchdarkly/python-server-sdk-ai/commit/3c65a6e4651e910e3547cc7114f785793114e911))
* only strip known provider prefixes ([66bc1f0](https://github.com/launchdarkly/python-server-sdk-ai/commit/66bc1f0db79f434614359b4e6724bd73e7d41216))
* **optimization:** address Bugbot comments on PR [#140](https://github.com/launchdarkly/python-server-sdk-ai/issues/140) ([9e57d86](https://github.com/launchdarkly/python-server-sdk-ai/commit/9e57d8625fd930eddbaaedfbc5f4cb8ee9ed5a85))
* **optimization:** fix CI test failures and apply Bugbot-reported fixes ([ffad0cb](https://github.com/launchdarkly/python-server-sdk-ai/commit/ffad0cb89a092d95b74356569b5de3f2a3dd8970))
* **optimization:** match model config by key as well as id for pricing lookup ([a8415be](https://github.com/launchdarkly/python-server-sdk-ai/commit/a8415be11c2cd03ecd1466d07aafa8f52031c107))
* **optimization:** remove dead persistence counter code ([c76cb16](https://github.com/launchdarkly/python-server-sdk-ai/commit/c76cb165f542f00eb848347e1df857e9703c113d))
* **optimization:** retry LLM calls on transient provider errors (429/503/529) ([a131e4b](https://github.com/launchdarkly/python-server-sdk-ai/commit/a131e4baadce5ce189c0c24e1632c76a72427f85))
* **optimization:** use run-chosen context for config judges; guard non-dict JSON ([64e12f2](https://github.com/launchdarkly/python-server-sdk-ai/commit/64e12f2b44bc6106a1f9325d3ae6dbd83c7c2980))
* pull model configs if available in options path ([4fc1ecf](https://github.com/launchdarkly/python-server-sdk-ai/commit/4fc1ecfad761ca2dbcefa4e407bddfc6adf7bb98))
* remove unnecessary token path ([dc82818](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc82818640878cdce7c657e27f6d9189f374d000))
* sort imports ([c032aaf](https://github.com/launchdarkly/python-server-sdk-ai/commit/c032aafd7ebd9384819498aab80fcf56f50eea65))
* success path + add test, cursor feedback ([8f3468f](https://github.com/launchdarkly/python-server-sdk-ai/commit/8f3468f9ea278ab910c40d03514c98f9f6c1289f))

## 0.1.0 (2026-04-02)


### Features

* Add optimization package stub ([#109](https://github.com/launchdarkly/python-server-sdk-ai/issues/109)) ([ebd5166](https://github.com/launchdarkly/python-server-sdk-ai/commit/ebd5166d86c2d58e4c2fcc0b3fcc983eb49574e6)) ([58b7731](https://github.com/launchdarkly/python-server-sdk-ai/commit/58b7731aa4f0efbd42ff0b93760eb357cdfe219f)) ([cc85a05](https://github.com/launchdarkly/python-server-sdk-ai/commit/cc85a05f0e81acdc33437b51238d67ddf8a92b80))
* Drop support for python 3.9 ([#114](https://github.com/launchdarkly/python-server-sdk-ai/issues/114)) ([dc592c5](https://github.com/launchdarkly/python-server-sdk-ai/commit/dc592c5a2e2bf3bf679af14a9aa63e81678a69ab))


### Bug Fixes

* consistency with other makefiles ([b9a5601](https://github.com/launchdarkly/python-server-sdk-ai/commit/b9a560110b9ef1746b1b1cff2b50ea8b90297acd))

## 0.0.0 (2026-03-24)

### Features

* Initial package scaffolding for optimization helpers.
