"""Example 3 — Multi-model config: register two providers and select at runtime.

This pattern mirrors the alias-based multi-model configuration available
through the JiuwenSwarm Configuration panel (/model <alias> switching).
At the SDK level the equivalent is: register two Workflows under distinct
workflow_id values, then choose which WorkflowAgent to invoke at runtime.

extra_body on ModelRequestConfig handles provider-specific parameters that
fall outside the standard OpenAI-compatible surface.  It is the OpenAI SDK's
supported escape hatch: the entries are merged into the JSON request body.

Required environment variables:
    OPENAI_API_KEY    — OpenAI API secret key
    DEEPSEEK_API_KEY  — DeepSeek API secret key
    OPENAI_API_BASE   — (optional) defaults to https://api.openai.com/v1
    DEEPSEEK_API_BASE — (optional) defaults to https://api.deepseek.com
"""
import asyncio
import os
from typing import Literal

from openjiuwen.core.application.workflow_agent import WorkflowAgent
from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.runner.runner import Runner

from src.workflow_builder import build_agent, build_workflow

# ── Provider registry ─────────────────────────────────────────────────────
# Map short alias names to config-factory callables.  Adding a third
# provider requires only a new entry here — no other code changes.

ProviderAlias = Literal["openai", "deepseek"]


def _make_openai_configs() -> tuple[ModelClientConfig, ModelRequestConfig]:
    """OpenAI connection + sampling configs."""
    client = ModelClientConfig(
        client_provider="OpenAI",
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        timeout=60,
        max_retries=3,
        verify_ssl=True,
    )
    request = ModelRequestConfig(
        model="gpt-4o",
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        stream=False,
    )
    return client, request


def _make_deepseek_configs() -> tuple[ModelClientConfig, ModelRequestConfig]:
    """DeepSeek connection + sampling configs.

    extra_body passes provider-specific parameters that sit outside the
    standard interface.  The OpenAI SDK merges them verbatim into the request
    body so the Workflow code never needs to know about them.  (Unknown
    top-level fields such as extra_fields would be forwarded as illegal SDK
    keyword arguments and fail.)
    """
    client = ModelClientConfig(
        client_provider="DeepSeek",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        timeout=60,
        max_retries=3,
        verify_ssl=True,
    )
    request = ModelRequestConfig(
        model="deepseek-chat",
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        stream=False,
        # extra_body forwards non-standard provider parameters without
        # polluting the shared ModelRequestConfig interface.
        extra_body={"frequency_penalty": 0},
    )
    return client, request


# Factory map: alias → config-factory function
_PROVIDER_FACTORIES = {
    "openai": _make_openai_configs,
    "deepseek": _make_deepseek_configs,
}

# Default provider alias — equivalent to is_default: true in the
# JiuwenSwarm YAML configuration.
_DEFAULT_PROVIDER: ProviderAlias = "openai"


def get_agent(alias: ProviderAlias = _DEFAULT_PROVIDER) -> WorkflowAgent:
    """Build and return a WorkflowAgent for the named provider alias.

    This is the SDK-level analogue of the /model <alias> runtime switch
    available in the JiuwenSwarm terminal interface.
    """
    if alias not in _PROVIDER_FACTORIES:
        raise ValueError(f"Unknown provider alias '{alias}'. "
                         f"Available: {list(_PROVIDER_FACTORIES)}")
    client_config, request_config = _PROVIDER_FACTORIES[alias]()
    flow = build_workflow(
        client_config,
        request_config,
        workflow_id=f"summarise_{alias}_multi",
    )
    return build_agent(
        agent_id=f"summarise_agent_{alias}_multi",
        description=f"Summarisation agent — {alias} provider (multi-model config)",
        flow=flow,
    )


async def run_multi_model() -> None:
    """Demonstrate runtime provider selection via alias."""
    query = (
        "The dual-config architecture enforces a hard separation between "
        "provider topology and inference behaviour at the type level, "
        "eliminating accidental coupling between the two concerns."
    )

    # Run with the default provider (openai).
    default_agent = get_agent(_DEFAULT_PROVIDER)
    try:
        result = await Runner.run_agent(default_agent, {"query": query})
        output = result.get("output")
        print(f"[{_DEFAULT_PROVIDER} — default] "
              f"{output.result if hasattr(output, 'result') else output}")
    except Exception as exc:
        print(f"[multi-model default] ERROR: {exc}")

    # Switch to the alternative provider (deepseek) — zero workflow changes.
    alt_alias: ProviderAlias = "deepseek"
    alt_agent = get_agent(alt_alias)
    try:
        result = await Runner.run_agent(alt_agent, {"query": query})
        output = result.get("output")
        print(f"[{alt_alias} — alternative] "
              f"{output.result if hasattr(output, 'result') else output}")
    except Exception as exc:
        print(f"[multi-model alt] ERROR: {exc}")


if __name__ == "__main__":
    asyncio.run(run_multi_model())