"""
Executes the worker agent on ONE real GitHub issue.

This version uses the OpenAI-compatible tool-calling format used by NVIDIA's
hosted API. The rest of AgentForge remains provider-independent.
"""

import json
import time

from llm_client import get_client, MODEL, estimate_cost_usd
from models import AgentState, IssueCase, ToolCallRecord

MAX_TURNS = 6

FINAL_ANSWER_INSTRUCTION = (
    "\n\nWhen you have decided, respond with ONLY a single-line JSON object "
    '{"labels": ["bug", "documentation"]} - no prose, no markdown fences. '
    "Choose zero or more labels from the exact valid label names."
)


def _render_memory(state: AgentState) -> str:
    if not state.memory:
        return "(no learned notes yet - this is an early run)"
    domain = [m.statement for m in state.memory if m.kind == "domain_fact"]
    tool_usage = [m.statement for m in state.memory if m.kind == "tool_usage"]
    parts = []
    if domain:
        parts.append(
            "Things learned about this repo's own conventions:\n"
            + "\n".join(f"- {s}" for s in domain)
        )
    if tool_usage:
        parts.append(
            "Things learned about using the tools effectively:\n"
            + "\n".join(f"- {s}" for s in tool_usage)
        )
    return "\n\n".join(parts)


def _openai_tool_schema(anthropic_style_schema: dict) -> dict:
    """Convert the project's existing tool schema to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": anthropic_style_schema["name"],
            "description": anthropic_style_schema.get("description", ""),
            "parameters": anthropic_style_schema.get(
                "input_schema",
                {"type": "object", "properties": {}},
            ),
        },
    }


def _assistant_message_for_history(message) -> dict:
    """
    Convert an SDK message object into the assistant message structure needed
    for the next OpenAI-compatible tool-result turn.
    """
    tool_calls = []
    for tc in (message.tool_calls or []):
        tool_calls.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        )

    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": tool_calls,
    }


def run_worker_agent(
    state: AgentState,
    issue: IssueCase,
    tools_registry: dict,
) -> dict:
    """Returns predicted_labels, tool_calls, cost_usd, and latency_s."""
    client = get_client()

    system_prompt = (
        state.core_instructions
        + "\n\n--- LEARNED MEMORY (apply this) ---\n"
        + _render_memory(state)
        + FINAL_ANSWER_INSTRUCTION
    )

    user_input = (
        f"Issue #{issue.number}\n"
        f"Title: {issue.title}\n\n"
        f"Body:\n{issue.body}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    tool_schemas = [
        _openai_tool_schema(entry["schema"])
        for entry in tools_registry.values()
    ]

    tool_call_records: list[ToolCallRecord] = []
    total_cost = 0.0
    start = time.time()
    predicted_labels: list[str] = []

    for _turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1024,
        )

        usage = getattr(response, "usage", None)
        if usage:
            total_cost += estimate_cost_usd(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )

        message = response.choices[0].message

        if not message.tool_calls:
            final_text = (message.content or "").strip()
            try:
                final_text = final_text.strip()
                if final_text.startswith("```"):
                    final_text = final_text.split("\n", 1)[1].rsplit(
                        "```", 1
                    )[0].strip()
                parsed = json.loads(final_text)
                predicted_labels = parsed.get("labels", [])
                if not isinstance(predicted_labels, list):
                    predicted_labels = []
            except (json.JSONDecodeError, AttributeError, TypeError):
                predicted_labels = []
            break

        # Add the assistant's tool-call message to conversation history.
        messages.append(_assistant_message_for_history(message))

        # Execute every tool requested in this turn.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            entry = tools_registry.get(name)
            was_error = False

            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
                output = "ERROR: model produced invalid JSON tool arguments"
                was_error = True
            else:
                try:
                    if entry is None:
                        output = f"ERROR: unknown tool {name}"
                        was_error = True
                    else:
                        output = entry["fn"](**arguments)
                        if (
                            isinstance(output, str)
                            and output.startswith("SEARCH ERROR")
                        ):
                            was_error = True
                except Exception as exc:
                    output = f"ERROR: {exc}"
                    was_error = True

            tool_call_records.append(
                ToolCallRecord(
                    tool_name=name,
                    tool_input=arguments,
                    tool_output=str(output),
                    was_error=was_error,
                )
            )

            # OpenAI-compatible APIs expect one tool message per tool call.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                }
            )

    return {
        "predicted_labels": predicted_labels,
        "tool_calls": tool_call_records,
        "cost_usd": round(total_cost, 6),
        "latency_s": round(time.time() - start, 3),
    }
