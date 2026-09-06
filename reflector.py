"""
The Reflector is the single meta-agent role in this design (deliberately
merged from what were two separate roles in an earlier draft - "don't build
more than you need"). It has two jobs:

  1. build_initial_state: write the agent's core instructions once, at the
     start of a learning session. This is meant to be generic and stay
     mostly stable - almost all of the improvement over time should show up
     in memory, not in prompt rewrites, because that's what actually
     demonstrates "learning" rather than "search."

  2. reflect: after each batch, look ONLY at the failing cases (score < 0.5)
     - including their tool-call traces - and propose new/revised memory
     entries. It explicitly separates domain_fact ("what's true about this
     repo") from tool_usage ("what works when calling these tools"),
     because the brief asks about both as distinct axes. instruction_patch
     is reserved for genuine structural problems (e.g. output format
     breaking repeatedly) and should stay rare - most learning belongs in
     memory, not in prompt churn.
"""

import uuid

from llm_client import call_for_json
from models import AgentState, BatchResult, MemoryEntry, Reflection

FAILURE_THRESHOLD = 0.5

INIT_PROMPT_TEMPLATE = """You are writing the initial system instructions for an AI \
agent whose job is to triage incoming GitHub issues on the repo {owner}/{repo} by \
predicting which of the repo's existing labels apply.

Valid labels on this repo:
{label_list}

Write clear, general system instructions (not specific to any one issue) covering: \
what the agent should look at (title, body), how to decide which labels apply, that \
it should use its tools (checking valid labels, searching precedent) rather than \
guessing, and that multiple labels can apply.

Return ONLY JSON: {{"core_instructions": "..."}}
"""

REFLECT_PROMPT_TEMPLATE = """You are improving a GitHub issue-triage agent by learning \
from its mistakes. Here is what it currently "knows" (its memory):

{current_memory}

Here are cases from its most recent batch where it scored poorly (expected labels vs. \
what it predicted, plus the tools it called and any tool errors):

{failing_cases}

Diagnose the root causes. Then propose:
  - memory_additions: NEW general, reusable lessons (not quotes of specific issues) \
that would help on FUTURE issues. Tag each as "domain_fact" (something true about \
this repo's own conventions) or "tool_usage" (something about using the tools \
effectively, e.g. a tool_error pattern to avoid repeating).
  - memory_revisions: if any EXISTING memory statement (by id, given above) turned \
out to be wrong or too narrow, give its id and a corrected statement.
  - instruction_patch: ONLY if the failures point to a structural/format problem \
(e.g. it keeps returning malformed output) rather than a knowledge gap. Otherwise null.

Return ONLY JSON:
{{
  "summary": "one or two sentences",
  "memory_additions": [{{"kind": "domain_fact"|"tool_usage", "statement": "..."}}],
  "memory_revisions": {{"<existing_id>": "corrected statement"}},
  "instruction_patch": null
}}
"""


def build_initial_state(owner: str, repo: str, labels: list[dict]) -> AgentState:
    label_list = "\n".join(f"- {l['name']}: {l['description']}" for l in labels)
    data = call_for_json(INIT_PROMPT_TEMPLATE.format(owner=owner, repo=repo, label_list=label_list))
    return AgentState(owner=owner, repo=repo, core_instructions=data["core_instructions"])


def _render_memory_for_prompt(state: AgentState) -> str:
    if not state.memory:
        return "(empty - this is the first batch)"
    return "\n".join(f"[{m.id}] ({m.kind}) {m.statement}" for m in state.memory)


def _render_failing_cases(batch: BatchResult) -> str:
    failing = [c for c in batch.cases if c.score < FAILURE_THRESHOLD]
    lines = []
    for c in failing:
        tool_summary = "; ".join(
            f"{tc.tool_name}({tc.tool_input}) -> {'ERROR: ' if tc.was_error else ''}{tc.tool_output[:120]}"
            for tc in c.tool_calls
        ) or "(no tool calls)"
        lines.append(
            f"- issue #{c.issue_number}: expected={c.actual_labels}, predicted={c.predicted_labels}\n"
            f"  tool calls: {tool_summary}"
        )
    return "\n".join(lines) if lines else "(no cases scored below 0.5 this batch)"


def reflect(state: AgentState, batch: BatchResult) -> Reflection:
    failing = [c for c in batch.cases if c.score < FAILURE_THRESHOLD]
    if not failing:
        return Reflection(summary="No significant failures this batch; memory left unchanged.")

    prompt = REFLECT_PROMPT_TEMPLATE.format(
        current_memory=_render_memory_for_prompt(state),
        failing_cases=_render_failing_cases(batch),
    )
    data = call_for_json(prompt)

    additions = [
        MemoryEntry(id=uuid.uuid4().hex[:8], kind=a["kind"], statement=a["statement"],
                    created_at_step=state.step + 1, last_reinforced_step=state.step + 1)
        for a in data.get("memory_additions", [])
    ]
    return Reflection(
        summary=data.get("summary", ""),
        memory_additions=additions,
        memory_revisions=data.get("memory_revisions", {}) or {},
        instruction_patch=data.get("instruction_patch"),
    )


def apply_reflection(state: AgentState, reflection: Reflection) -> AgentState:
    """Applies a Reflection to produce the NEXT state. Additive by default -
    memory grows; existing entries are only touched if explicitly revised."""
    new_memory = list(state.memory)
    revised_ids = set(reflection.memory_revisions.keys())
    for m in new_memory:
        if m.id in revised_ids:
            m.statement = reflection.memory_revisions[m.id]
            m.last_reinforced_step = state.step + 1
    new_memory.extend(reflection.memory_additions)

    new_instructions = reflection.instruction_patch or state.core_instructions

    return AgentState(
        owner=state.owner, repo=state.repo,
        core_instructions=new_instructions,
        memory=new_memory,
        step=state.step + 1,
    )
