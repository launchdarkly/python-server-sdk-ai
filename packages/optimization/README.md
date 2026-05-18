# LaunchDarkly AI SDK — optimization

[![PyPI](https://img.shields.io/pypi/v/ldai_optimizer.svg?style=flat-square)](https://pypi.org/project/ldai_optimizer/)

> [!CAUTION]
> This package is in pre-release and not subject to backwards compatibility
> guarantees. The API may change based on feedback.
>
> Pin to a specific minor version and review the [changelog](CHANGELOG.md) before upgrading.

This package provides helpers for running iterative AI prompt optimization workflows from within LaunchDarkly SDK-based applications. It drives the optimization loop — generating candidate variations, evaluating them with judges, and optionally committing winners back to LaunchDarkly — while delegating all LLM calls to your own handler functions.

## Requirements

- Python `>=3.9`
- A configured [LaunchDarkly server-side SDK](https://docs.launchdarkly.com/sdk/server-side/python) client
- The [LaunchDarkly AI package](https://pypi.org/project/launchdarkly-server-sdk-ai/) (`launchdarkly-server-sdk-ai>=0.16.0`) — pulled in automatically as a dependency
- **`LAUNCHDARKLY_API_KEY` environment variable** — required only when using `auto_commit=True` or `optimize_from_config`. Not needed for basic `optimize_from_options` runs without auto-commit.

> [!NOTE]
> **`LAUNCHDARKLY_API_KEY` is used exclusively for discrete LaunchDarkly REST API calls** (fetching configs, publishing results). It is never included in any LLM prompt and is never forwarded to your handler callbacks. All API calls made by this package are isolated; they have no access to your runtime environment beyond the key you explicitly provide via the environment variable.

## Installation

```bash
pip install ldai_optimizer
```

## Quick Start

### Basic optimization (`optimize_from_options`)

No `LAUNCHDARKLY_API_KEY` required unless `auto_commit=True`.

```python
import ldclient
from ldai import LDAIClient
from ldai_optimizer import (
    OptimizationClient,
    OptimizationJudge,
    OptimizationOptions,
    OptimizationResponse,
    LLMCallConfig,
    LLMCallContext,
)

ldclient.set_config(ldclient.Config("sdk-your-sdk-key"))
ld = LDAIClient(ldclient.get())
client = OptimizationClient(ld)

def handle_llm_call(
    run_id: str,
    config: LLMCallConfig,
    context: LLMCallContext,
    is_evaluation: bool,
) -> OptimizationResponse:
    # config.model, config.instructions, config.key are available
    # context.user_input, context.current_variables are available
    response = your_llm_client.chat(
        model=config.model.name if config.model else "gpt-4o",
        system=config.instructions,
        user=context.user_input or "",
    )
    return OptimizationResponse(completion=response.text)

result = await client.optimize_from_options(
    OptimizationOptions(
        agent_key="my-agent",
        handle_agent_call=handle_llm_call,
        judge_model="gpt-4o-mini",
        judges={
            "quality": OptimizationJudge(
                threshold=1.0,
                acceptance_statement="The response is accurate and concise.",
            )
        },
        model_choices=["gpt-4o", "gpt-4o-mini"],
        variable_choices=[{"user_id": "user-123"}],
        user_input_choices=["What is my account balance?"],
    )
)
```

### Ground truth optimization

```python
from ldai_optimizer import GroundTruthOptimizationOptions, GroundTruthSample

result = await client.optimize_from_options(
    GroundTruthOptimizationOptions(
        agent_key="my-agent",
        handle_agent_call=handle_llm_call,
        judge_model="gpt-4o-mini",
        judges={
            "accuracy": OptimizationJudge(
                threshold=1.0,
                acceptance_statement="The response matches the expected answer.",
            )
        },
        model_choices=["gpt-4o", "gpt-4o-mini"],
        ground_truth_responses=[
            GroundTruthSample(
                user_input="What is 2+2?",
                ground_truth_response="4",
            )
        ],
    )
)
```

### Config-driven optimization (`optimize_from_config`)

Requires `LAUNCHDARKLY_API_KEY`.

```python
from ldai_optimizer import OptimizationFromConfigOptions

result = await client.optimize_from_config(
    OptimizationFromConfigOptions(
        config_key="my-optimization-config",
        project_key="my-project",
        handle_agent_call=handle_llm_call,
        auto_commit=True,
    )
)
```

## License

Apache-2.0
