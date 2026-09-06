"""
Deterministic tests for the memory quality gate (memory.py + apply_reflection).

No LLM calls, no network, no API keys: pure logic tests that pin the behavior
the task requires - duplicate merge, contradiction detection, genericness
rejection, relevance ranking, cap enforcement, and the apply_reflection report.
"""

import pytest

import memory
from models import AgentState, MemoryEntry, Reflection
from reflector import apply_reflection


def _entry(statement, kind="domain_fact", evidence=1, step=1, id="e1"):
    return MemoryEntry(
        id=id, kind=kind, statement=statement,
        evidence_count=evidence, created_at_step=step, last_reinforced_step=step,
    )


def _state(memory_entries, step=3):
    return AgentState(
        owner="pallets", repo="flask",
        core_instructions="Triage issues.", memory=memory_entries, step=step,
    )


def _reflection(*additions, revisions=None):
    return Reflection(summary="s", memory_additions=list(additions),
                      memory_revisions=revisions or {})


# ---------------------------------------------------------------------------
# Duplicate detection / merging
# ---------------------------------------------------------------------------

def test_exact_duplicate_is_merged_not_added():
    existing = _entry("issues mentioning 'segfault' are labeled 'bug' in this repo")
    state = _state([existing])
    duplicate = MemoryEntry(id="new1", kind="domain_fact",
                            statement="issues mentioning 'segfault' are labeled 'bug' in this repo",
                            created_at_step=4, last_reinforced_step=4)

    next_state, change = apply_reflection(state, _reflection(duplicate))

    assert len(next_state.memory) == 1
    assert next_state.memory[0].evidence_count == 2          # reinforced, not duplicated
    assert change.created == []
    assert change.merged == [duplicate.statement]
    assert change.revised == []
    assert change.rejected == []


def test_paraphrase_is_merged():
    existing = _entry("auth issues are labeled 'bug' in this repo")
    paraphrase = MemoryEntry(id="new1", kind="domain_fact",
                             statement="issues about auth are labeled 'bug' in this repo",
                             created_at_step=4, last_reinforced_step=4)

    next_state, change = apply_reflection(_state([existing]), _reflection(paraphrase))

    assert len(next_state.memory) == 1
    assert next_state.memory[0].evidence_count == 2
    assert change.merged == [paraphrase.statement]


def test_same_batch_duplicates_merge_into_each_other():
    a = MemoryEntry(id="a", kind="domain_fact", statement="crashes are 'bug'",
                    created_at_step=4, last_reinforced_step=4)
    b = MemoryEntry(id="b", kind="domain_fact", statement="crashes are labeled bug",
                    created_at_step=4, last_reinforced_step=4)

    next_state, change = apply_reflection(_state([]), _reflection(a, b))

    assert len(next_state.memory) == 1
    assert next_state.memory[0].evidence_count == 2
    assert change.created == ["a"]
    assert len(change.merged) == 1


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def test_contradiction_detected_and_resolved_as_revision():
    existing = _entry("issues mentioning 'segfault' are labeled 'bug' in this repo")
    contradictory = MemoryEntry(
        id="new1", kind="domain_fact",
        statement="issues mentioning 'segfault' are NOT labeled 'bug' in this repo",
        created_at_step=4, last_reinforced_step=4,
    )

    next_state, change = apply_reflection(_state([existing]), _reflection(contradictory))

    assert len(next_state.memory) == 1                     # no growth
    assert next_state.memory[0].id == existing.id          # same identity kept
    assert next_state.memory[0].statement == contradictory.statement  # newer wins
    assert next_state.memory[0].evidence_count == 1        # corrected, not reinforced
    assert change.revised == [existing.id]


def test_same_topic_different_label_is_contradiction():
    existing = _entry("auth issues are 'bug'")
    conflicting = MemoryEntry(id="new1", kind="domain_fact",
                              statement="auth issues are 'feature'",
                              created_at_step=4, last_reinforced_step=4)

    next_state, change = apply_reflection(_state([existing]), _reflection(conflicting))

    assert len(next_state.memory) == 1
    assert next_state.memory[0].statement == "auth issues are 'feature'"
    assert change.revised == [existing.id]
    assert change.created == []


def test_compatible_overlap_is_neither_dup_nor_contradiction():
    existing = _entry("server errors are 'bug'")
    new = MemoryEntry(id="new1", kind="domain_fact",
                      statement="server errors are 'bug' and they should be investigated quickly",
                      created_at_step=4, last_reinforced_step=4)

    next_state, change = apply_reflection(_state([existing]), _reflection(new))

    # "server errors are 'bug'" is a strict subset -> duplicate/paraphrase, merged.
    assert len(next_state.memory) == 1
    assert change.merged
    assert change.revised == []


# ---------------------------------------------------------------------------
# Generic statements rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("statement", [
    "read the issue carefully and understand it before labeling",
    "be careful",
    "always check the title and body and think about what applies",
    "make sure to be thoughtful when deciding",
])
def test_generic_statements_rejected(statement):
    new = MemoryEntry(id="new1", kind="domain_fact", statement=statement,
                      created_at_step=4, last_reinforced_step=4)

    next_state, change = apply_reflection(_state([]), _reflection(new))

    assert len(next_state.memory) == 0
    assert change.rejected == [(statement, "generic")]
    assert change.created == []


def test_specific_statement_not_rejected():
    new = MemoryEntry(id="new1", kind="domain_fact",
                      statement="issues mentioning 'segfault' are labeled 'bug' in this repo",
                      created_at_step=4, last_reinforced_step=4)

    next_state, change = apply_reflection(_state([]), _reflection(new))

    assert len(next_state.memory) == 1
    assert change.created == ["new1"]
    assert change.rejected == []


# ---------------------------------------------------------------------------
# Memory cap
# ---------------------------------------------------------------------------

def test_memory_cap_rejects_overflow(monkeypatch):
    monkeypatch.setattr(memory, "MAX_MEMORY_SIZE", 3)
    existing = [_entry(f"fact number {i} about this repo", id=f"e{i}") for i in range(2)]
    overflow = [
        MemoryEntry(id="new0", kind="tool_usage",
                    statement="search queries should include words from the issue title",
                    created_at_step=4, last_reinforced_step=4),
        MemoryEntry(id="new1", kind="tool_usage",
                    statement="use the repo's real label names when proposing labels",
                    created_at_step=4, last_reinforced_step=4),
    ]

    next_state, change = apply_reflection(_state(existing), _reflection(*overflow))

    assert len(next_state.memory) == 3
    assert change.created == ["new0"]
    assert change.rejected == [(
        "use the repo's real label names when proposing labels", "memory cap",
    )]


# ---------------------------------------------------------------------------
# Revisions
# ---------------------------------------------------------------------------

def test_revision_applies_to_existing_id():
    existing = _entry("flask routing issues are 'bug'")
    state = _state([existing])

    next_state, change = apply_reflection(
        state, _reflection(revisions={"e1": "flask routing issues are 'bug' AND 'feature'"}),
    )

    assert next_state.memory[0].statement == "flask routing issues are 'bug' AND 'feature'"
    assert next_state.memory[0].last_reinforced_step == 4
    assert change.revised == ["e1"]


def test_revision_with_unknown_id_is_dropped():
    state = _state([_entry("flask routing issues are 'bug'")])

    next_state, change = apply_reflection(
        state, _reflection(revisions={"ghost-id": "some corrected statement"}),
    )

    assert next_state.memory[0].statement == "flask routing issues are 'bug'"
    assert change.dropped_revisions == ["ghost-id"]
    assert change.revised == []


# ---------------------------------------------------------------------------
# Relevance ranking for worker prompt injection
# ---------------------------------------------------------------------------

def _memories():
    return [
        _entry("issues mentioning 'werkzeug' are 'bug'", evidence=3, id="werk"),
        _entry("issues mentioning 'template' are 'documentation'", evidence=1, id="tmpl"),
        _entry("never guess label names; call list_labels first", kind="tool_usage",
               evidence=5, id="labels"),
    ]


def test_ranking_prioritizes_issue_relevance():
    ranked = memory.rank_memories(_memories(), issue_text="werkzeug crashes on startup")
    assert ranked[0].id == "werk"     # statement token 'werkzeug' in the issue


def test_ranking_falls_back_to_evidence_and_recency():
    ranked = memory.rank_memories(_memories(), issue_text="")
    assert ranked[0].id == "labels"   # evidence 5 > werk(3) > tmpl(1)
    assert ranked[-1].id == "tmpl"


def test_ranking_caps_at_top_k():
    mems = [_entry(f"distinct lesson number {i} about label conventions", id=f"m{i}",
                   evidence=1) for i in range(20)]
    ranked = memory.rank_memories(mems, issue_text="", top_k=12)
    assert len(ranked) == 12


# ---------------------------------------------------------------------------
# Runner injection (worker receives relevant learned memories)
# ---------------------------------------------------------------------------

def test_render_memory_only_includes_top_k():
    from runner import _render_memory
    mems = [_entry(f"lesson {i}: issues mentioning token{i} are 'bug'", id=f"m{i}",
                   evidence=1) for i in range(20)]
    state = _state(mems)
    text = _render_memory(state, issue_text="token3 crash")
    assert "token3" in text
    assert "withheld" in text
    assert text.count("- (1x)") <= memory.MAX_MEMORY_PER_PROMPT
    # the most relevant entry is listed first
    lines = [l for l in text.splitlines() if l.startswith("- (")]
    assert "token3" in lines[0]


def test_render_memory_empty_state():
    from runner import _render_memory
    assert "no learned notes" in _render_memory(_state([]))


def test_worker_prompt_includes_learned_memory():
    from runner import render_worker_prompt_for_issue
    from models import IssueCase
    state = _state([_entry("issues mentioning 'werkzeug' are 'bug'", evidence=3, id="werk")])
    issue = IssueCase(number=1, title="werkzeug crashes on startup", body="lots of text",
                      created_at="2020-01-01T00:00:00Z", actual_labels=["bug"])
    prompt = render_worker_prompt_for_issue(state, issue)
    assert "LEARNED MEMORY" in prompt
    assert "werkzeug" in prompt
    assert "(3x)" in prompt            # reinforcement strength shown to the worker