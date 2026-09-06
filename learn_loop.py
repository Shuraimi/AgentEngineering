"""
Ties everything together: build (or resume) an AgentState, pull a real
historical dataset once, slice it into time-ordered batches, and step
through them - run, score, reflect, persist, repeat.

This file is intentionally the simplest one in the project; all real
complexity lives in github_tools/runner/evaluator/reflector. If something
breaks, you're almost always debugging one of those four, not this glue.
"""

from typing import Callable, Optional

import github_tools
import state_store
from evaluator import score_case
from models import AgentState, BatchResult, CaseResult, IssueCase
from reflector import apply_reflection, build_initial_state, reflect
from runner import run_worker_agent

DEFAULT_LABEL_POOL_SIZE = 6  # how many of the repo's real labels to focus the demo on


def prepare_dataset(owner: str, repo: str, token: str | None,
                     num_batches: int, batch_size: int) -> tuple[list[dict], list[IssueCase]]:
    """Fetch real repo labels + a real historical, labeled, time-ordered
    dataset sized to exactly fill num_batches * batch_size."""
    all_labels = github_tools.fetch_repo_labels(owner, repo, token)
    # Focus on the most commonly-used labels so the label space stays tractable
    # for a demo - a real production version would use all of them.
    label_pool = [l["name"] for l in all_labels[:DEFAULT_LABEL_POOL_SIZE]]

    total_needed = num_batches * batch_size
    raw = github_tools.build_historical_dataset(
        owner, repo, label_pool, token=token,
        per_label=max(10, total_needed // max(1, len(label_pool)) + 4),
        total_target=total_needed,
    )
    cases = [IssueCase(**item) for item in raw[:total_needed]]
    return all_labels, cases


def run_time_step(
    state: AgentState, batch: list[IssueCase], owner: str, repo: str, token: str | None,
    labels: list[dict],
) -> BatchResult:
    case_results: list[CaseResult] = []
    tool_error_count = 0

    for issue in batch:
        tools_registry = github_tools.build_tool_registry(
            owner, repo, labels, token,
            before_date=issue.created_at, exclude_number=issue.number,
        )
        run_out = run_worker_agent(state, issue, tools_registry)
        score, _reason = score_case(issue.actual_labels, run_out["predicted_labels"])
        errors_this_case = sum(1 for tc in run_out["tool_calls"] if tc.was_error)
        tool_error_count += errors_this_case

        case_results.append(CaseResult(
            issue_number=issue.number,
            predicted_labels=run_out["predicted_labels"],
            actual_labels=issue.actual_labels,
            tool_calls=run_out["tool_calls"],
            score=score,
            cost_usd=run_out["cost_usd"],
            latency_s=run_out["latency_s"],
        ))

    n = len(case_results)
    return BatchResult(
        step=state.step + 1,
        cases=case_results,
        accuracy=sum(c.score for c in case_results) / n,
        avg_cost_usd=sum(c.cost_usd for c in case_results) / n,
        avg_latency_s=sum(c.latency_s for c in case_results) / n,
        tool_error_count=tool_error_count,
        memory_size_before=len(state.memory),
    )


def run_learning_session(
    owner: str, repo: str, token: Optional[str], num_batches: int = 4, batch_size: int = 6,
    resume: bool = True,
    on_step: Optional[Callable[[BatchResult, AgentState], None]] = None,
) -> tuple[AgentState, list[BatchResult]]:
    """
    Runs (or resumes) a full learning session: num_batches sequential time
    steps over real historical issues. on_step(batch_result, state_after)
    is called after each step - this is what app.py hooks into for the live
    UI, same pattern as before.
    """
    labels, cases = prepare_dataset(owner, repo, token, num_batches, batch_size)

    state = state_store.load_state(owner, repo) if resume else None
    if state is None:
        state = build_initial_state(owner, repo, labels)
        state_store.save_state(state)

    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)][:num_batches]

    history: list[BatchResult] = []
    for batch in batches:
        if not batch:
            continue
        result = run_time_step(state, batch, owner, repo, token, labels)
        reflection = reflect(state, result)
        state = apply_reflection(state, reflection)

        state_store.save_state(state)
        state_store.save_batch_result(owner, repo, result)
        history.append(result)

        if on_step:
            on_step(result, state)

    return state, history
