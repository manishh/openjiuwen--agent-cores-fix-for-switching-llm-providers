"""Shared workflow-graph factory.

This module owns ONLY the graph topology (Start → LLM → End) and the
prompt template.  It accepts pre-built ModelClientConfig and
ModelRequestConfig objects and returns a fully wired Workflow.

The key architectural point: the graph is constructed once and is
completely decoupled from the provider-connection and inference-sampling
layers.  Swapping providers requires only replacing the two config objects
passed into `build_workflow`; the graph, prompt, and output schema stay
identical.
"""
from openjiuwen.core.application.workflow_agent import WorkflowAgent
from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.single_agent.legacy import WorkflowAgentConfig
from openjiuwen.core.workflow import (
    End,
    LLMCompConfig,
    LLMComponent,
    Start,
    Workflow,
    WorkflowCard,
    generate_workflow_key,
)


def build_llm_comp_config(
    client_config: ModelClientConfig,
    request_config: ModelRequestConfig,
) -> LLMCompConfig:
    """Bind both config layers into a single component-level config.

    ``model_client_config`` (connection layer) and ``model_config``
    (sampling layer) are the two independently replaceable type-enforced
    slots.  Everything else — the system message, the prompt template,
    and the JSON output schema — is provider-agnostic.
    """
    return LLMCompConfig(
        # ── Layer 1: connection topology ──────────────────────────────────
        model_client_config=client_config,
        # ── Layer 2: inference / sampling ────────────────────────────────
        model_config=request_config,
        # ── Provider-agnostic prompt template ────────────────────────────
        template_content=[
            {
                "role": "system",
                "content": (
                    "You are a concise summarisation assistant. "
                    "Output only the summary with no preamble."
                ),
            },
            {
                "role": "user",
                # {{query}} is resolved by the workflow engine from
                # the inputs_schema wired to this component (see below).
                "content": "Summarise the following in one sentence:\n\n{{query}}",
            },
        ],
        # Tell the framework to parse the response as structured JSON.
        response_format={"type": "json"},
        # Declare the output key that downstream nodes (including End)
        # can reference via the ${llm.output} variable expression.
        output_config={
            "type": "object",
            "description": "LLM structured output",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "One-sentence summary",
                }
            },
            "required": ["output"],
        },
    )


def build_workflow(
    client_config: ModelClientConfig,
    request_config: ModelRequestConfig,
    *,
    workflow_id: str,
) -> Workflow:
    """Construct and register a Workflow graph.

    Graph topology: Start ──► LLMComponent ──► End
    Variable interpolation in inputs_schema defines the edges:
      - ${query}        top-level workflow input
      - ${start.query}  output of the Start node named "start"
      - ${llm.output}   output field of the LLMComponent named "llm"
    """
    llm_config = build_llm_comp_config(client_config, request_config)
    llm = LLMComponent(llm_config)

    card = WorkflowCard(
        id=workflow_id,
        name=workflow_id,
        version="1.0",
        description="One-sentence summarisation workflow",
        input_params={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to summarise"}
            },
            "required": ["query"],
        },
    )
    flow = Workflow(card=card)

    start = Start()
    # End receives the rendered responseTemplate as its final output.
    end = End({"responseTemplate": "{{output}}"})

    # Register and connect each node.  inputs_schema expressions are how
    # Agent Core wires data between components — no explicit edge API needed.
    flow.set_start_comp("start", start, inputs_schema={"query": "${query}"})
    flow.add_workflow_comp("llm", llm, inputs_schema={"query": "${start.query}"})
    flow.set_end_comp("end", end, inputs_schema={"output": "${llm.output}"})
    flow.add_connection("start", "llm")
    flow.add_connection("llm", "end")

    # Register the workflow factory with the global resource manager so
    # Runner can resolve it by its versioned key at execution time.
    # add_workflows() also performs this registration internally; doing it
    # explicitly here matches the documented pattern.
    Runner.resource_mgr.add_workflow(
        WorkflowCard(id=generate_workflow_key(card.id, card.version)),
        lambda: flow,
    )

    return flow


def build_agent(
    agent_id: str,
    description: str,
    flow: Workflow,
) -> WorkflowAgent:
    """Wrap a Workflow in a WorkflowAgent and bind the two together."""
    agent_config = WorkflowAgentConfig(
        id=agent_id,
        version="1.0",
        description=description,
    )
    agent = WorkflowAgent(agent_config)
    # add_workflows registers the workflow into the agent's ability manager
    # and (internally) into Runner.resource_mgr.
    agent.add_workflows([flow])
    return agent