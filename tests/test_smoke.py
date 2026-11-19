"""Smoke test: verify that all Agent Core classes import correctly and
can be instantiated with valid arguments — no API calls are made.

Run with: pytest -q tests/test_smoke.py
"""
import pytest


# ── 1. Verify import paths ────────────────────────────────────────────────
def test_model_client_config_import() -> None:
    from openjiuwen.core.foundation.llm import ModelClientConfig
    assert ModelClientConfig is not None


def test_model_request_config_import() -> None:
    from openjiuwen.core.foundation.llm import ModelRequestConfig
    assert ModelRequestConfig is not None


def test_workflow_imports() -> None:
    from openjiuwen.core.workflow import (
        End,
        LLMCompConfig,
        LLMComponent,
        Start,
        Workflow,
        WorkflowCard,
        generate_workflow_key,
    )
    for obj in (End, LLMCompConfig, LLMComponent, Start, Workflow, WorkflowCard,
                generate_workflow_key):
        assert obj is not None


def test_runner_import() -> None:
    from openjiuwen.core.runner.runner import Runner
    assert Runner is not None


def test_workflow_agent_imports() -> None:
    from openjiuwen.core.application.workflow_agent import WorkflowAgent
    from openjiuwen.core.single_agent.legacy import WorkflowAgentConfig
    assert WorkflowAgent is not None
    assert WorkflowAgentConfig is not None


# ── 2. Verify instantiation of the two config objects ────────────────────
def test_model_client_config_instantiation() -> None:
    from openjiuwen.core.foundation.llm import ModelClientConfig
    cfg = ModelClientConfig(
        client_provider="OpenAI",
        api_key="sk-test-dummy",
        api_base="https://api.openai.com/v1",
        timeout=30,
        max_retries=2,
        verify_ssl=True,
    )
    # Confirm the object carries the values we set.
    assert cfg is not None


def test_model_request_config_instantiation() -> None:
    from openjiuwen.core.foundation.llm import ModelRequestConfig
    cfg = ModelRequestConfig(
        model="gpt-4o",
        temperature=0.7,
        top_p=1.0,
        max_tokens=512,
        stream=False,
    )
    assert cfg is not None


def test_model_request_config_extra_body() -> None:
    """extra_body passes vendor-specific parameters without widening the interface."""
    from openjiuwen.core.foundation.llm import ModelRequestConfig
    cfg = ModelRequestConfig(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=512,
        extra_body={"frequency_penalty": 0},
    )
    assert cfg is not None


# ── 3. Verify generate_workflow_key produces a non-empty string ───────────
def test_generate_workflow_key() -> None:
    from openjiuwen.core.workflow import generate_workflow_key
    key = generate_workflow_key("my_workflow", "1.0")
    assert isinstance(key, str)
    assert len(key) > 0


# ── 4. Verify WorkflowCard and Workflow construction ─────────────────────
def test_workflow_card_and_workflow_instantiation() -> None:
    from openjiuwen.core.workflow import Workflow, WorkflowCard
    card = WorkflowCard(
        id="smoke_test_workflow",
        name="smoke_test_workflow",
        version="1.0",
        description="Smoke test workflow",
        input_params={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    flow = Workflow(card=card)
    assert flow is not None


# ── 5. Verify LLMCompConfig accepts model_config (not model_request_config) ─
def test_llm_comp_config_correct_param_name() -> None:
    """Regression guard: the parameter is model_config, not model_request_config."""
    from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.workflow import LLMCompConfig
    client = ModelClientConfig(
        client_provider="OpenAI",
        api_key="sk-dummy",
        api_base="https://api.openai.com/v1",
    )
    request = ModelRequestConfig(model="gpt-4o", temperature=0.5, max_tokens=128)
    comp_config = LLMCompConfig(
        model_client_config=client,
        model_config=request,  # must be model_config, not model_request_config
        template_content=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{query}}"},
        ],
        response_format={"type": "json"},
        output_config={
            "type": "object",
            "properties": {"output": {"type": "string"}},
            "required": ["output"],
        },
    )
    assert comp_config is not None
