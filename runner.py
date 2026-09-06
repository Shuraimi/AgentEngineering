"""
Executes the worker agent on ONE real GitHub issue.

This version uses the OpenAI-compatible tool-calling format used by NVIDIA's
hosted API. The rest of AgentForge remains provider-independent.
"""

import json
import time

from llm_client import clean_model_message, clean_tool_name, get_client, MODEL, estimate_cost_usd
from memory import rank_memories
from models import AgentState, IssueCase, ToolCallRecord

MAX_TURNS = 6

FINAL_ANSWER_INSTRUCTION = (
    "\n\nWhen you have decided, respond with ONLY a single-line JSON object "
    '{"labels": ["bug", "documentation"]} - no prose, no markdown fences. '
    "Choose zero or more labels from the exact valid label names."
)


def _render_memory(state: AgentState, issue_text: str = "") -> str:
    """
    Render the LEARNED MEMORY section of the worker prompt.

    Memory is ranked per-issue (relevance first, then reinforcement strength
    and recency) and capped, so the worker receives the memories that are
    most relevant to the issue at hand instead of an ever-growing dump.
    """
    if not state.memory:
        return "(no learned notes yet - this is an early run)"

    selected = rank_memories(state.memory, issue_text=issue_text)
    domain = [m for m in selected if m.kind == "domain_fact"]
    tool_usage = [m for m in selected if m.kind == "tool_usage"]
    parts = []
    if domain:
        parts.append(
            "Things learned about this repo's own conventions "
            "(reinforced N times):\n"
            + "\n".join(f"- ({m.evidence_count}x) {m.statement}" for m in domain)
        )
    if tool_usage:
        parts.append(
            "Things learned about using the tools effectively "
            "(reinforced N times):\n"
            + "\n".join(f"- ({m.evidence_count}x) {m.statement}" for m in tool_usage)
        )
    text = "\n\n".join(parts)
    withheld = len(state.memory) - len(selected)
    if withheld > 0:
        text += (
            f"\n\n(Showing the {len(selected)} most relevant learned notes for this "
            f"issue; {withheld} older/less relevant notes withheld.)"
        )
    return text


def render_worker_prompt_for_issue(state: AgentState, issue: IssueCase) -> str:
    """Full system prompt the worker agent sees for ONE issue. Public so tests
    and experiment probes can verify that learned memory is actually injected."""
    return (
        state.core_instructions
        + "\n\n--- LEARNED MEMORY (apply this) ---\n"
        + _render_memory(state, issue_text=f"{issue.title}\n{issue.body}")
        + FINAL_ANSWER_INSTRUCTION
    )


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
                    "name": clean_tool_name(tc.function.name),
                    "arguments": tc.function.arguments,
                },
            }
        )

    return {
        "role": "assistant",
        "content": clean_model_message(message.content),
        "tool_calls": tool_calls,
    }


def _expected_params(entry: dict) -> str:
    """Human-readable list of a tool's declared parameters, for error messages
    that actually teach the model the correct shape (prevents the Reflector
    from hallucinating a wrong "fix" from a raw Python TypeError)."""
    if entry is None:
        return "(unknown tool)"
    schema = entry.get("schema") or {}
    input_schema = schema.get("input_schema") or {}
    props = list((input_schema.get("properties") or {}).keys())
    return ", ".join(props) if props else "(none - call it with {})"


def run_worker_agent(
    state: AgentState,
    issue: IssueCase,
    tools_registry: dict,
) -> dict:
    """Returns predicted_labels, tool_calls, cost_usd, and latency_s."""
    client = get_client()

    system_prompt = render_worker_prompt_for_issue(state, issue)

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
            final_text = clean_model_message(message.content).strip()
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
            # GPT-OSS can append a harmony channel marker to the function name
            # (e.g. search_similar_issues<|channel|>commentary) - strip it so
            # the call resolves instead of erroring with "unknown tool".
            name = clean_tool_name(tool_call.function.name)
            entry = tools_registry.get(name)
            was_error = False

            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
                output = "ERROR: model produced invalid JSON tool arguments"
                was_error = True
            else:
                # Confirmed live failure: the model can emit a non-object
                # argument (e.g. list_labels(1330)). json.loads accepts it,
                # **-unpacking crashes, and ToolCallRecord.tool_input: dict
                # then kills the whole run with a pydantic ValidationError.
                if not isinstance(arguments, dict):
                    raw = arguments
                    arguments = {"_raw_args": raw}
                    output = (
                        f"ERROR: tool arguments must be a JSON object of "
                        f"parameters, got {type(raw).__name__} ({raw!r}). "
                        f"Expected params for {name}: {_expected_params(entry)}"
                    )
                    was_error = True
                    tool_call_records.append(
                        ToolCallRecord(
                            tool_name=name,
                            tool_input=arguments,
                            tool_output=output,
                            was_error=was_error,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output,
                        }
                    )
                    continue
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
                    output = (
                        f"ERROR calling {name}: {exc}. "
                        f"Expected params: {_expected_params(entry)}"
                    )
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
