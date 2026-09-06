"""
No-UI end-to-end test of a full learning session against a REAL repo.

Usage:

    python run_cli.py pallets flask --batches 4 --batch-size 6

    python run_cli.py <owner> <repo> --reset  # wipe and start fresh

    # The MEASURED protocol: baseline + cycles over a FIXED held-out eval set.
    python run_cli.py pallets flask --experiment --reset

Run this before touching the Streamlit UI - if the accuracy/cost/latency
curve across steps prints correctly here, the UI (which calls the exact
same functions) will work too.
"""

import argparse
import os
import sys

import neatlogs
from dotenv import load_dotenv

load_dotenv()

# Windows consoles default to cp1252, which cannot encode characters the
# LLM's memory statements can contain (confirmed live: U+2011 crashed every
# print of a learned memory). Route console output through UTF-8 instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

neatlogs.init(workflow_name="issue-triage-cli")

import state_store

from learn_loop import run_learning_experiment, run_learning_session
from models import ExperimentResult


def print_experiment_report(experiment: ExperimentResult) -> None:
    e = experiment
    total_issue_runs = sum(len(r.issue_numbers) for r in e.rounds) or 1

    print("\n" + "=" * 60)
    print("LEARNING EXPERIMENT REPORT")
    print("=" * 60)
    print(
        f"Repo: {e.owner}/{e.repo} | {e.num_cycles} cycles x "
        f"{len(e.train_issue_numbers) // max(1, e.num_cycles)} train issues/cycle "
        f"| {len(e.eval_issue_numbers)} held-out eval issues"
    )
    print(
        f"Baseline accuracy (fresh agent, empty memory): "
        f"{e.baseline_accuracy:.1%}"
    )

    train_acc = {r.cycle: r.accuracy for r in e.rounds if r.round_type == "train_cycle"}
    print()
    print("Cycle | Train acc  | Eval acc (SAME set) | vs baseline | vs previous")
    for i, acc in enumerate(e.cycle_accuracy, start=1):
        print(
            f"  {i}   | {train_acc.get(i, 0):.1%}      | "
            f"{acc:.1%}               | "
            f"{e.improvement_from_baseline[i - 1]:+.1f} pp   | "
            f"{e.improvement_from_previous_cycle[i - 1]:+.1f} pp"
        )

    print()
    print("Memory changes (train cycles only):")
    print(
        f"  {e.memory_added} created | {e.memory_merged} merged | "
        f"{e.memory_revised} revised | {e.memory_rejected} rejected"
    )
    print("Tool usage across all rounds:")
    print(
        f"  {e.tool_calls} calls (avg {e.tool_calls / total_issue_runs:.2f}/issue) | "
        f"{e.tool_errors} errors (avg {e.tool_errors / total_issue_runs:.2f}/issue)"
    )
    print(
        f"  Avg latency: {e.average_latency:.2f}s | "
        f"Estimated cost: ${e.estimated_cost:.4f}"
    )
    print()
    print("Improvements are in PERCENTAGE POINTS (pp), not %.")
    print(f"Results persisted at data/agents/{e.owner}_{e.repo}/experiment.json")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("owner")
    parser.add_argument("repo")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="wipe existing agent state first",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help=(
            "run the measured learning protocol: baseline eval of a fixed "
            "held-out set, then --cycles training cycles that re-evaluate "
            "the SAME held-out set"
        ),
    )
    parser.add_argument("--cycles", type=int, default=2,
                        help="training cycles for --experiment (default: 2)")
    parser.add_argument("--train-batch-size", type=int, default=6,
                        help="train issues per cycle for --experiment (default: 6)")
    parser.add_argument("--eval-size", type=int, default=6,
                        help="held-out eval issues for --experiment (default: 6)")

    args = parser.parse_args()

    if args.experiment:
        for name, value in (("--cycles", args.cycles),
                            ("--train-batch-size", args.train_batch_size),
                            ("--eval-size", args.eval_size)):
            if value < 1:
                parser.error(f"{name} must be >= 1, got {value}")
        if not args.reset and state_store.load_state(args.owner, args.repo) is not None:
            print(
                f"An agent state already exists for {args.owner}/{args.repo}, but an "
                "experiment must start from a clean agent (empty memory baseline). "
                "Re-run with --reset."
            )
            sys.exit(1)

    if args.reset:
        state_store.reset_agent(args.owner, args.repo)
        print(f"Reset agent state for {args.owner}/{args.repo}")

    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        print(
            "WARNING: no GITHUB_TOKEN set - unauthenticated rate limits are tight (60/hr). "
            "Set GITHUB_TOKEN in .env for a smoother run.\n"
        )

    def on_step(result, state, change):
        print(f"\n=== Step {result.step} ===")

        tool_calls = sum(len(c.tool_calls) for c in result.cases)

        print(
            f"Accuracy: {result.accuracy:.0%} | "
            f"Avg cost: ${result.avg_cost_usd:.4f} | "
            f"Avg latency: {result.avg_latency_s:.2f}s | "
            f"Tool calls: {tool_calls} | "
            f"Tool errors: {result.tool_error_count}"
        )

        print(
            f"Memory size: {result.memory_size_before} "
            f"-> {len(state.memory)}"
        )

        if change.created:
            print(f"Memories created ({len(change.created)}):")
            by_id = {m.id: m for m in state.memory}
            for mid in change.created:
                m = by_id.get(mid)
                if m:
                    print(f"  [{m.kind}] {m.statement}")
        if change.merged:
            print(f"Memories merged (duplicates reinforced, {len(change.merged)}):")
            for stmt in change.merged:
                print(f"  ~ {stmt}")
        if change.revised:
            print(f"Memories revised ({len(change.revised)}):")
            by_id = {m.id: m for m in state.memory}
            for mid in change.revised:
                m = by_id.get(mid)
                if m:
                    print(f"  [revised {mid}] {m.statement}")
        if change.rejected:
            print(f"Memories rejected ({len(change.rejected)}):")
            for stmt, reason in change.rejected:
                print(f"  x {stmt}  ({reason})")
        if change.dropped_revisions:
            print(f"Revisions dropped (unknown ids): {change.dropped_revisions}")

    def on_round(metrics, state, change):
        label = {
            "baseline_eval": "baseline eval  ",
            "train_cycle": f"train cycle {metrics.cycle}",
            "cycle_eval": f"eval cycle {metrics.cycle} (SAME set)",
        }[metrics.round_type]
        print(
            f"{label} | {len(metrics.issue_numbers)} issues | "
            f"acc {metrics.accuracy:.0%} | mem {metrics.memory_size} | "
            f"tools {metrics.tool_calls} ({metrics.tool_errors} err) | "
            f"{metrics.average_latency_s:.2f}s | ${metrics.estimated_cost_usd:.4f}"
        )
        if change and (change.created or change.merged or change.revised):
            print(
                f"  memory: +{len(change.created)} created, "
                f"{len(change.merged)} merged, {len(change.revised)} revised"
            )

    if args.experiment:
        try:
            final_state, experiment = run_learning_experiment(
                args.owner,
                args.repo,
                token,
                train_batch_size=args.train_batch_size,
                num_cycles=args.cycles,
                eval_size=args.eval_size,
                on_round=on_round,
            )
        finally:
            neatlogs.flush()
            neatlogs.shutdown()
        print_experiment_report(experiment)
        return

    try:
        final_state, history = run_learning_session(
            args.owner,
            args.repo,
            token,
            num_batches=args.batches,
            batch_size=args.batch_size,
            on_step=on_step,
        )
    finally:
        neatlogs.flush()
        neatlogs.shutdown()

    print("\n" + "=" * 50)

    print(f"Session complete: {len(history)} steps run.")

    if len(history) >= 2:
        print(
            f"Accuracy: {history[0].accuracy:.0%} "
            f"(step 1) -> "
            f"{history[-1].accuracy:.0%} "
            f"(step {history[-1].step})"
        )

    print(f"Final memory size: {len(final_state.memory)} entries")

    print(
        f"State persisted at "
        f"data/agents/{args.owner}_{args.repo}/"
    )


if __name__ == "__main__":
    main()