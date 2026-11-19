"""Example 1 — Baseline WorkflowAgent backed by OpenAI.

This file demonstrates the full construction sequence:
  1. ModelClientConfig  — connection topology (provider, key, endpoint)
  2. ModelRequestConfig — inference sampling (model, temperature, tokens)
  3. LLMCompConfig      — binds both layers plus the prompt template
  4. Workflow graph     — Start → LLMComponent → End
  5. WorkflowAgent      — wraps the graph into an agent
  6. Runner.run_agent   — executes the agent

Only steps 1 and 2 change when you switch providers (see provider_swap.py).

Required environment variables:
    OPENAI_API_KEY   — OpenAI API secret key
    OPENAI_API_BASE  — (optional) custom endpoint; defaults to
                       https://api.openai.com/v1
"""
import asyncio
import os

from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.runner.runner import Runner

from src.workflow_builder import build_agent, build_workflow


def build_openai_client_config() -> ModelClientConfig:
    """ModelClientConfig — connection topology.

    Owns:
      client_provider  — which SDK client to instantiate
      api_key          — credential (read from environment; never hardcoded)
      api_base         — endpoint root (omit the /chat/completions suffix;
                         the framework appends it automatically)
      timeout          — per-request timeout in seconds
      max_retries      — automatic retry count on transient network errors
      verify_ssl       — SSL certificate verification

    None of these fields are visible to the LLM sampling layer.
    """
    return ModelClientConfig(
        client_provider="OpenAI",
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        timeout=60,
        max_retries=3,
        verify_ssl=True,
    )


def build_openai_request_config() -> ModelRequestConfig:
    """ModelRequestConfig — inference / sampling parameters.

    Owns:
      model        — exact model identifier for this provider
      temperature  — randomness of sampling
      top_p        — nucleus-sampling cutoff
      max_tokens   — upper token budget for the completion
      stream       — whether to stream the response token-by-token

    No connection details appear here.
    """
    return ModelRequestConfig(
        model="gpt-4o",
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        stream=False,
    )


async def run_baseline() -> None:
    """Build and run the OpenAI-backed summarisation agent."""
    # ── Step 1-2: construct the two config objects ─────────────────────
    client_config = build_openai_client_config()
    request_config = build_openai_request_config()

    # ── Step 3-4: wire configs into a workflow graph ───────────────────
    # The graph topology and prompt template are defined inside
    # build_workflow and are completely independent of the configs.
    flow = build_workflow(
        client_config,
        request_config,
        workflow_id="summarise_openai_v1",
    )

    # ── Step 5: create the agent ───────────────────────────────────────
    agent = build_agent(
        agent_id="summarise_agent_openai",
        description="Summarisation agent backed by OpenAI gpt-4o",
        flow=flow,
    )

    # ── Step 6: run via the class-level Runner facade ──────────────────
    try:
        result = await Runner.run_agent(
            agent,
            {
                "query": (
                    "Agent Core enforces a hard separation between connection config "
                    "(ModelClientConfig) and inference config (ModelRequestConfig), "
                    "so switching providers requires no changes to the workflow graph."
                )
            },
        )
        output = result.get("output")
        print(f"[OpenAI gpt-4o] {output.result if hasattr(output, 'result') else output}")
    except Exception as exc:  # surface provider errors clearly
        print(f"[OpenAI baseline] ERROR: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(run_baseline())