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

import memory
from llm_client import call_for_json
from models import (
    AgentState,
    BatchResult,
    MemoryChange,
    MemoryEntry,
    Reflection,
)

FAILURE_THRESHOLD = 0.5
MAX_ADDITIONS_PER_REFLECTION = 3  # the quality gate also enforces a global cap

# The EXACT tool signatures the worker can call. Shown to the Reflector so a
# tool_usage lesson it writes matches reality - without this it has invented
# wrong arguments (e.g. "call list_labels(issue_number)"), which then taught
# the worker to make tool calls that fail.
TOOLS_DESCRIPTION = (
    "The worker's ONLY tools are:\n"
    "  - list_labels(): takes NO arguments. Returns the repo's exact valid label names.\n"
    "  - search_similar_issues(query: str): searches past closed issues for precedent."
)

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
from its mistakes. Here is what it currently "knows" (its memory, with evidence counts):
{current_memory}

Here are cases from its most recent batch where it scored poorly (expected labels vs. \
what it predicted, plus the tools it called and any tool errors):

{failing_cases}

{TOOLS_DESCRIPTION}

Diagnose the root causes. Then propose:
  - memory_additions: at most 3 NEW general, reusable lessons (not quotes of specific \
issues) that would help on FUTURE issues. Tag each as "domain_fact" (something true \
about this repo's own conventions) or "tool_usage" (something about using the tools \
effectively, e.g. a tool_error pattern to avoid repeating). Be SPECIFIC to this repo: \
name the actual labels or tool behaviors involved. Do NOT repeat a lesson that is \
already in memory, do NOT add generic advice ("read the issue carefully", "be \
thoughtful") that would apply to any repo, and do NOT propose something that \
contradicts an existing entry - if an existing entry is wrong, correct it via \
memory_revisions instead.
  - memory_revisions: if any EXISTING memory statement (by id, given above) turned \
out to be wrong or too narrow, give its id and a corrected statement. Only reference \
ids that actually appear in the memory list above.
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
    return "\n".join(
        f"[{m.id}] ({m.kind}, evidence={m.evidence_count}) {m.statement}"
        for m in state.memory
    )


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
        TOOLS_DESCRIPTION=TOOLS_DESCRIPTION,
    )
    try:
        data = call_for_json(prompt)
    except Exception:
        # The hosted model occasionally returns empty content (confirmed live:
        # 'Could not find JSON in model response' twice). Retry once, then
        # degrade to a no-op reflection rather than killing the whole session
        # and losing the step's persisted state.
        try:
            data = call_for_json(prompt)
        except Exception:
            return Reflection(
                summary=(
                    "Reflection LLM call failed twice (empty/invalid JSON); "
                    "memory left unchanged this step."
                ),
            )

    additions = []
    raw_additions = data.get("memory_additions", []) or []
    if not isinstance(raw_additions, list):
        raw_additions = []
    for a in raw_additions[:MAX_ADDITIONS_PER_REFLECTION]:
        if not isinstance(a, dict):
            continue
        statement = str(a.get("statement", "")).strip()
        if not statement:
            continue
        kind = a.get("kind")
        if kind not in ("domain_fact", "tool_usage"):
            kind = "domain_fact"  # be forgiving: a bad kind shouldn't kill the step
        additions.append(MemoryEntry(
            id=uuid.uuid4().hex[:8], kind=kind, statement=statement,
            created_at_step=state.step + 1, last_reinforced_step=state.step + 1,
        ))
    return Reflection(
        summary=data.get("summary", ""),
        memory_additions=additions,
        memory_revisions=data.get("memory_revisions", {}) or {},
        instruction_patch=data.get("instruction_patch"),
    )


def apply_reflection(state: AgentState, reflection: Reflection) -> tuple[AgentState, MemoryChange]:
    """
    Applies a Reflection to produce the NEXT state, running every proposed
    change through the deterministic quality gate in memory.py:

      - additions that duplicate an existing entry are MERGED (evidence bump,
        no new entry);
      - additions that contradict an existing entry are resolved as a
        REVISION (the newer, evidence-informed statement replaces the old);
      - generic boilerplate and overflow beyond the memory cap are REJECTED;
      - explicit memory_revisions are applied only when the id actually exists.

    Returns (next_state, MemoryChange) so callers can report exactly what was
    created / merged / revised / rejected instead of guessing from size deltas.
    """
    new_memory = [m.model_copy(deep=True) for m in state.memory]
    change = MemoryChange()
    step = state.step + 1

    # 1) Explicit LLM revisions: only apply to ids that actually exist.
    for mid, new_statement in (reflection.memory_revisions or {}).items():
        target = next((m for m in new_memory if m.id == mid), None)
        if target is None:
            change.dropped_revisions.append(mid)
            continue
        target.statement = new_statement
        target.last_reinforced_step = step
        change.revised.append(mid)

    # 2) Proposed additions through the quality gate.
    for addition in reflection.memory_additions:
        if memory.is_generic(addition.statement):
            change.rejected.append((addition.statement, "generic"))
            continue

        duplicate = next(
            (m for m in new_memory if memory.is_duplicate(m.statement, addition.statement)),
            None,
        )
        if duplicate is not None:
            duplicate.evidence_count += 1
            duplicate.last_reinforced_step = step
            change.merged.append(addition.statement)
            continue

        contradiction = next(
            (m for m in new_memory if memory.is_contradiction(m.statement, addition.statement)),
            None,
        )
        if contradiction is not None:
            contradiction.statement = addition.statement
            contradiction.last_reinforced_step = step
            change.revised.append(contradiction.id)
            continue

        if len(new_memory) >= memory.MAX_MEMORY_SIZE:
            change.rejected.append((addition.statement, "memory cap"))
            continue

        new_memory.append(addition)
        change.created.append(addition.id)

    new_instructions = reflection.instruction_patch or state.core_instructions

    return AgentState(
        owner=state.owner, repo=state.repo,
        core_instructions=new_instructions,
        memory=new_memory,
        step=step,
    ), change
