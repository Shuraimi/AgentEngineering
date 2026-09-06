"""
Shared data models for AgentForge (GitHub Issue Triage edition).

The key conceptual change from a "regenerate N candidate prompts, keep the
best" design: there is now exactly ONE AgentState per (owner, repo). It
persists across time steps and accumulates knowledge in `memory` - it is
never thrown away and replaced. That's what makes this "an agent that gets
better at a task over time" rather than "automated prompt search."
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, timezone


class MemoryEntry(BaseModel):
    """
    One atomic, reusable piece of learned knowledge. Two kinds:
      - "domain_fact": something true about THIS repo's own conventions
        (e.g. "issues mentioning 'CUDA out of memory' are labeled
        'performance', not 'bug', in this repo").
      - "tool_usage": something learned about using the tools effectively
        (e.g. "label names are case-sensitive and must match list_labels()
        exactly - guessing a plausible-sounding label name fails").

    Memory entries are meant to be genuralizable statements, NOT verbatim
    quotes of specific issues - the Reflector is instructed to abstract,
    not memorize individual examples.
    """
    id: str
    kind: Literal["domain_fact", "tool_usage"]
    statement: str
    evidence_count: int = 1          # how many batches have reinforced this
    created_at_step: int = 0
    last_reinforced_step: int = 0


class AgentState(BaseModel):
    """The full, persistent identity of one triage agent for one repo."""
    owner: str
    repo: str
    core_instructions: str           # relatively stable; only patched for structural fixes
    memory: list[MemoryEntry] = Field(default_factory=list)
    step: int = 0                    # how many time steps this agent has lived through


class IssueCase(BaseModel):
    """One real GitHub issue pulled via the API, used as an eval case.
    `actual_labels` is real ground truth (the maintainers' own labels) -
    hidden from the agent at prediction time, revealed only for scoring."""
    number: int
    title: str
    body: str
    created_at: str                  # ISO8601, used for temporal-leakage filtering
    actual_labels: list[str]


class ToolCallRecord(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str
    was_error: bool = False          # True if the tool call itself failed
                                      # (e.g. invalid label name) - this is
                                      # exactly the signal tool_usage memory
                                      # entries are meant to reduce over time


class CaseResult(BaseModel):
    issue_number: int
    predicted_labels: list[str]
    actual_labels: list[str]
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    score: float                     # Jaccard similarity, 0-1
    cost_usd: float
    latency_s: float


class BatchResult(BaseModel):
    """The outcome of running the CURRENT agent state on one time step's
    worth of held-out issues, before reflection updates the state."""
    step: int
    cases: list[CaseResult]
    accuracy: float                  # mean Jaccard score across the batch
    avg_cost_usd: float
    avg_latency_s: float
    tool_error_count: int
    memory_size_before: int


class Reflection(BaseModel):
    """What the Reflector proposes after seeing one BatchResult."""
    summary: str
    memory_additions: list[MemoryEntry] = Field(default_factory=list)
    memory_revisions: dict[str, str] = Field(default_factory=dict)  # id -> new statement
    instruction_patch: Optional[str] = None   # rare - only for structural/format fixes


class MemoryChange(BaseModel):
    """
    What actually HAPPENED to memory when a Reflection was applied, after the
    deterministic quality gate (memory.py) disposed of the model's proposals:

      - created:    ids of brand-new entries that entered memory
      - merged:     proposed statements absorbed into an existing duplicate
      - revised:    ids of existing entries whose statement was corrected
                    (explicit LLM revision OR contradiction resolution)
      - rejected:   (statement, reason) - generic boilerplate or cap overflow
      - dropped_revisions: revision ids the model referenced that don't exist
    """
    created: list[str] = Field(default_factory=list)
    merged: list[str] = Field(default_factory=list)
    revised: list[str] = Field(default_factory=list)
    rejected: list[tuple[str, str]] = Field(default_factory=list)
    dropped_revisions: list[str] = Field(default_factory=list)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
