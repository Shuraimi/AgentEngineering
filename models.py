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


class RoundMetrics(BaseModel):
    """
    One scored round inside a learning experiment. `round_type` says what was
    evaluated and whether the agent was ALLOWED to learn from it:

      - "baseline_eval": the held-out eval set, run with the fresh agent
        (empty memory) BEFORE any learning. This is the accuracy every
        improvement is measured against.
      - "train_cycle": a training batch the agent worked through and then
        reflected on. Learning happens here and only here.
      - "cycle_eval": the SAME held-out eval set re-run after one training
        cycle. Pure measurement - the results are never shown to the
        Reflector, so the agent can never learn from the eval set.

    `memory_size` is len(state.memory) at the START of the round, which makes
    it obvious when eval rounds use the post-training memory without feeding
    anything back.
    """
    round_type: Literal["baseline_eval", "train_cycle", "cycle_eval"]
    cycle: int = 0                      # 0 for baseline_eval, else 1-based cycle index
    issue_numbers: list[int]
    accuracy: float                     # mean Jaccard score (0-1) across the round
    tool_calls: int                     # total tool calls in the round
    tool_errors: int                    # total tool errors in the round
    average_latency_s: float            # mean seconds per issue
    estimated_cost_usd: float           # total USD for the round
    memory_size: int


class ExperimentResult(BaseModel):
    """
    The comparable outcome of ONE experiment run: a baseline evaluation of a
    fixed held-out eval set, then num_cycles of (train on a time-ordered
    batch -> re-evaluate the SAME eval set). Because every cycle accuracy is
    measured on the identical issues with the identical deterministic scorer,
    the deltas are attributable to learning and not to a different test.

    All improvements are in PERCENTAGE POINTS (pp), not relative percent:
    an agent going 42% -> 67% improved by 25pp, not by 25%.
    """
    owner: str
    repo: str
    created_at: str                     # UTC ISO8601, when the experiment ran
    num_cycles: int
    baseline_accuracy: float                            # mean Jaccard, 0-1
    cycle_accuracy: list[float]                         # one per cycle, SAME eval set
    improvement_from_baseline: list[float]              # pp vs baseline, len == num_cycles
    improvement_from_previous_cycle: list[float]        # pp vs previous measurement, len == num_cycles
    memory_added: int
    memory_merged: int
    memory_revised: int
    memory_rejected: int
    tool_calls: int                     # summed over all rounds
    tool_errors: int                    # summed over all rounds
    average_latency: float              # seconds, latency-weighted mean over all rounds
    estimated_cost: float               # USD, summed over all rounds
    train_issue_numbers: list[int]
    eval_issue_numbers: list[int]
    rounds: list[RoundMetrics] = Field(default_factory=list)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
