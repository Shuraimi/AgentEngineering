"""
Ties everything together: build (or resume) an AgentState, pull a real
historical dataset once, slice it into time-ordered batches, and step
through them - run, score, reflect, persist, repeat.

This file is intentionally the simplest one in the project; all real
complexity lives in github_tools/runner/evaluator/reflector. If something
breaks, you're almost always debugging one of those four, not this glue.
"""

from typing import Callable, Optional

import neatlogs
import github_tools
import state_store
from evaluator import score_case
from metrics import improvement_from_baseline, improvement_from_previous_cycle
from models import (
    AgentState,
    BatchResult,
    CaseResult,
    ExperimentResult,
    IssueCase,
    MemoryChange,
    RoundMetrics,
    utcnow_iso,
)
from reflector import apply_reflection, build_initial_state, reflect
from runner import run_worker_agent

DEFAULT_LABEL_POOL_SIZE = 6  # how many of the repo's real labels to focus the demo on


def prepare_dataset(owner: str, repo: str, token: str | None,
                    num_batches: int, batch_size: int,
                    eval_size: int = 0) -> tuple[list[dict], list[IssueCase]]:
    """Fetch real repo labels + a real historical, labeled, time-ordered
    dataset sized to exactly fill num_batches * batch_size training issues
    plus eval_size held-out eval issues.

    The search returns issues oldest-first, so the LAST eval_size issues are
    the NEWEST - evaluating them repeatedly across cycles measures
    generalization to unseen issues, not memorization. eval_size defaults to
    0 so the normal learning session's behavior is unchanged."""
    all_labels = github_tools.fetch_repo_labels(owner, repo, token)
    # Focus on the most commonly-used labels so the label space stays tractable
    # for a demo - a real production version would use all of them.
    label_pool = [l["name"] for l in all_labels[:DEFAULT_LABEL_POOL_SIZE]]

    total_needed = num_batches * batch_size + eval_size
    raw = github_tools.build_historical_dataset(
        owner, repo, label_pool, token=token,
        # The +6 slack exists because build_historical_dataset dedupes
        # candidates ACROSS labels (flask's old issues carry several pool
        # labels), so a fetch sized to exactly total_needed/len(pool) per
        # label comes up short: a 36-issue request got 35 unique issues at
        # +4 slack. +6 yields ~40-51 unique for the same request shape while
        # leaving every smaller run unchanged (max(10, ...) still dominates).
        per_label=max(10, total_needed // max(1, len(label_pool)) + 6),
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


@neatlogs.span(kind="WORKFLOW", name="issue-triage-session", capture_input=False, capture_output=False)
def run_learning_session(
    owner: str, repo: str, token: Optional[str], num_batches: int = 4, batch_size: int = 6,
    resume: bool = True,
    on_step: Optional[Callable[[BatchResult, AgentState, MemoryChange], None]] = None,
) -> tuple[AgentState, list[BatchResult]]:
    """
    Runs (or resumes) a full learning session: num_batches sequential time
    steps over real historical issues. on_step(batch_result, state_after,
    memory_change) is called after each step - this is what app.py hooks into
    for the live UI, same pattern as before.
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
        state, change = apply_reflection(state, reflection)

        state_store.save_state(state)
        state_store.save_batch_result(owner, repo, result)
        history.append(result)

        if on_step:
            on_step(result, state, change)

    return state, history


def _round_metrics(round_type: str, cycle: int, result: BatchResult) -> RoundMetrics:
    """Convert one runtime BatchResult into its comparable RoundMetrics form."""
    return RoundMetrics(
        round_type=round_type,
        cycle=cycle,
        issue_numbers=[c.issue_number for c in result.cases],
        accuracy=result.accuracy,
        tool_calls=sum(len(c.tool_calls) for c in result.cases),
        tool_errors=result.tool_error_count,
        average_latency_s=result.avg_latency_s,
        estimated_cost_usd=round(result.avg_cost_usd * len(result.cases), 6),
        memory_size=result.memory_size_before,
    )


@neatlogs.span(kind="WORKFLOW", name="issue-triage-session", capture_input=False, capture_output=False)
def run_learning_experiment(
    owner: str, repo: str, token: Optional[str],
    train_batch_size: int = 6, num_cycles: int = 2, eval_size: int = 6,
    on_round: Optional[Callable[[RoundMetrics, AgentState, MemoryChange], None]] = None,
) -> tuple[AgentState, ExperimentResult]:
    """
    Runs the MEASURED learning protocol - the one that can actually prove
    "the agent gets better at the task":

      1. baseline_eval: score the held-out eval set with the fresh agent
         (empty memory). This is the baseline every improvement is measured
         against.
      2. per cycle: train on one time-ordered batch, reflect ONLY on that
         batch's failures, apply the reflection (memory grows here), then
         re-score the SAME held-out eval set.

    The eval set is fixed and newest-first (prepare_dataset is time-ordered
    oldest-first), so every cycle accuracy is measured on identical issues we
    never trained on. Eval runs are pure measurement: their results are never
    passed to reflect(), so the agent can never learn from the eval set.

    on_round(round_metrics, state_after, memory_change) is called after each
    round with an empty MemoryChange for eval/baseline rounds (no learning
    happened). The final ExperimentResult is persisted alongside the agent
    state to data/agents/<owner>_<repo>/experiment.json.
    """
    labels, cases = prepare_dataset(
        owner, repo, token, num_cycles, train_batch_size, eval_size=eval_size,
    )

    train_end = num_cycles * train_batch_size
    train_cases = cases[:train_end]
    eval_cases = cases[train_end:train_end + eval_size]

    if len(train_cases) < train_end:
        raise ValueError(
            f"Need {train_end} training issues ({num_cycles} cycles x "
            f"{train_batch_size}/cycle) but only {len(train_cases)} were fetched. "
            "Increase the dataset size or lower --cycles/--train-batch-size."
        )
    if len(eval_cases) < eval_size:
        raise ValueError(
            f"Need {eval_size} held-out eval issues but only {len(eval_cases)} "
            f"remain after reserving {train_end} for training. Increase the "
            "dataset size or lower --eval-size."
        )

    # An experiment always starts from a fresh agent: a memory-free baseline
    # is what the improvements are measured against.
    state = build_initial_state(owner, repo, labels)
    state_store.save_state(state)

    rounds: list[RoundMetrics] = []
    memory_added = memory_merged = memory_revised = memory_rejected = 0

    # Baseline: score the held-out eval set BEFORE any learning.
    baseline_result = run_time_step(state, eval_cases, owner, repo, token, labels)
    baseline = _round_metrics("baseline_eval", 0, baseline_result)
    rounds.append(baseline)
    if on_round:
        on_round(baseline, state, MemoryChange())

    cycle_accuracies: list[float] = []
    train_issue_numbers: list[int] = []

    for cycle in range(1, num_cycles + 1):
        train_batch = train_cases[(cycle - 1) * train_batch_size: cycle * train_batch_size]
        train_issue_numbers.extend(issue.number for issue in train_batch)

        # --- learn from THIS train batch ---
        train_result = run_time_step(state, train_batch, owner, repo, token, labels)
        train_metrics = _round_metrics("train_cycle", cycle, train_result)
        rounds.append(train_metrics)

        reflection = reflect(state, train_result)
        state, change = apply_reflection(state, reflection)
        memory_added += len(change.created)
        memory_merged += len(change.merged)
        memory_revised += len(change.revised)
        memory_rejected += len(change.rejected)
        state_store.save_state(state)

        if on_round:
            on_round(train_metrics, state, change)

        # --- measure on the SAME held-out eval set (never reflected on) ---
        eval_result = run_time_step(state, eval_cases, owner, repo, token, labels)
        eval_metrics = _round_metrics("cycle_eval", cycle, eval_result)
        rounds.append(eval_metrics)
        cycle_accuracies.append(eval_metrics.accuracy)

        if on_round:
            on_round(eval_metrics, state, MemoryChange())

    total_cases = sum(len(r.issue_numbers) for r in rounds)
    total_latency = sum(r.average_latency_s * len(r.issue_numbers) for r in rounds)

    experiment = ExperimentResult(
        owner=owner,
        repo=repo,
        created_at=utcnow_iso(),
        num_cycles=num_cycles,
        baseline_accuracy=round(baseline.accuracy, 4),
        cycle_accuracy=[round(a, 4) for a in cycle_accuracies],
        improvement_from_baseline=improvement_from_baseline(cycle_accuracies, baseline.accuracy),
        improvement_from_previous_cycle=improvement_from_previous_cycle(cycle_accuracies, baseline.accuracy),
        memory_added=memory_added,
        memory_merged=memory_merged,
        memory_revised=memory_revised,
        memory_rejected=memory_rejected,
        tool_calls=sum(r.tool_calls for r in rounds),
        tool_errors=sum(r.tool_errors for r in rounds),
        average_latency=round(total_latency / total_cases, 4) if total_cases else 0.0,
        estimated_cost=round(sum(r.estimated_cost_usd for r in rounds), 6),
        train_issue_numbers=train_issue_numbers,
        eval_issue_numbers=[issue.number for issue in eval_cases],
        rounds=rounds,
    )
    state_store.save_experiment_result(owner, repo, experiment)

    return state, experiment
