"""Example 2 — Provider swap: replace only ModelClientConfig and ModelRequestConfig.

Every other object — the Workflow graph, the LLMComponent, the prompt
template, the output schema, and the WorkflowAgent — is imported verbatim
from workflow_builder.py and used without a single modification.

Compare with baseline_agent.py: the only differences are the two
build_*_config functions.  The workflow, agent, and runner invocation
are character-for-character identical.

Required environment variables:
    DEEPSEEK_API_KEY   — DeepSeek API secret key
    DEEPSEEK_API_BASE  — (optional) custom endpoint; defaults to
                         https://api.deepseek.com
"""
import asyncio
import os

from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.runner.runner import Runner

from src.workflow_builder import build_agent, build_workflow


def build_deepseek_client_config() -> ModelClientConfig:
    """Swap Layer 1: connection topology now points at DeepSeek.

    Only client_provider, api_key, and api_base change versus the
    OpenAI config.  timeout, max_retries, and verify_ssl are identical.
    """
    return ModelClientConfig(
        client_provider="DeepSeek",                              # ← changed
        api_key=os.environ["DEEPSEEK_API_KEY"],                  # ← changed
        api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),  # ← changed
        timeout=60,
        max_retries=3,
        verify_ssl=True,
    )


def build_deepseek_request_config() -> ModelRequestConfig:
    """Swap Layer 2: inference params now target deepseek-chat.

    Only model changes.  temperature, top_p, max_tokens, and stream
    are identical to the OpenAI config.
    """
    return ModelRequestConfig(
        model="deepseek-chat",   # ← changed
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        stream=False,
    )


async def run_swap() -> None:
    """Run the same summarisation workflow — now backed by DeepSeek.

    The call to build_workflow(), build_agent(), and Runner.run_agent()
    below is copied verbatim from baseline_agent.run_baseline().
    Only the two config-building lines differ.
    """
    # ── Swapped: two new config objects ───────────────────────────────
    client_config = build_deepseek_client_config()   # ← provider swap
    request_config = build_deepseek_request_config() # ← model swap

    # ── Unchanged: identical workflow graph ───────────────────────────
    flow = build_workflow(
        client_config,
        request_config,
        workflow_id="summarise_deepseek_v1",
    )

    # ── Unchanged: identical agent construction ────────────────────────
    agent = build_agent(
        agent_id="summarise_agent_deepseek",
        description="Summarisation agent backed by DeepSeek deepseek-chat",
        flow=flow,
    )

    # ── Unchanged: identical runner invocation ────────────────────────
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
        print(f"[DeepSeek deepseek-chat] {output.result if hasattr(output, 'result') else output}")
    except Exception as exc:
        print(f"[DeepSeek swap] ERROR: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(run_swap())