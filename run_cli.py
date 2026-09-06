"""
No-UI end-to-end test of a full learning session against a REAL repo.

Usage:

    python run_cli.py pallets flask --batches 4 --batch-size 6

    python run_cli.py <owner> <repo> --reset  # wipe and start fresh

Run this before touching the Streamlit UI - if the accuracy/cost/latency
curve across steps prints correctly here, the UI (which calls the exact
same functions) will work too.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Windows consoles default to cp1252, which cannot encode characters the
# LLM's memory statements can contain (confirmed live: U+2011 crashed every
# print of a learned memory). Route console output through UTF-8 instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import state_store

from learn_loop import run_learning_session


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

    args = parser.parse_args()

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

    final_state, history = run_learning_session(
        args.owner,
        args.repo,
        token,
        num_batches=args.batches,
        batch_size=args.batch_size,
        on_step=on_step,
    )

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