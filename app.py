"""
AgentForge Streamlit Dashboard — READ-ONLY.

A thin visualization layer over the two persisted artifact files:
  data/agents/<owner>_<repo>/experiment.json
  data/agents/<owner>_<repo>/state.json

This file does NOT run the backend, does NOT modify any backend module, and
does NOT hardcode any metrics. Everything rendered on screen is read from the
JSON files at runtime. If either file is missing or malformed the dashboard
shows a graceful empty state instead of crashing.

Accuracy improvements are always shown in PERCENTAGE POINTS (pp). A move from
30.6% to 41.7% is displayed as "+11.1 pp", never "+11.1%" (unless the value is
a genuine relative change). Flat results are shown honestly with the explicit
`No measurable improvement on the fixed held-out evaluation set.` message.
"""

import atexit
import json
import os
from datetime import datetime

import neatlogs
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Data access (pure Python + stdlib so tests can import without streamlit-run)
# ---------------------------------------------------------------------------

neatlogs.init(workflow_name="issue-triage-ui")


_neatlogs_shutdown_done = False


def _shutdown_neatlogs() -> None:
    """Flush and stop the Neatlogs exporter once at Streamlit process exit.

    Best-effort and never raises: at interpreter exit stdout may already be
    closed (e.g. a pytest run that imports this module), and Neatlogs logs its
    final "shutdown complete" line to stdout via a StreamHandler. Raising the
    logger level during teardown filters that final info message so a closed
    stdout can never surface a logging error or crash an atexit hook.
    """
    global _neatlogs_shutdown_done
    if _neatlogs_shutdown_done:
        return
    _neatlogs_shutdown_done = True
    try:
        import logging
        root = logging.getLogger("neatlogs")
        prev_level = root.level
        try:
            if prev_level > logging.WARNING:
                # Already above INFO threshold; leave it alone.
                root.level = prev_level
            else:
                root.setLevel(logging.WARNING)
            neatlogs.flush()
            neatlogs.shutdown()
        finally:
            root.setLevel(prev_level)
    except Exception:
        pass


atexit.register(_shutdown_neatlogs)

DEFAULT_OWNER = "pallets"
DEFAULT_REPO = "flask"

HONEST_RESULT_MESSAGE = (
    "No measurable improvement on the fixed held-out evaluation set."
)


def data_dir(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> str:
    """Absolute path to the persisted artifacts for a given owner/repo."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "agents")
    return os.path.join(base, f"{owner}_{repo}")


def experiment_path(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> str:
    return os.path.join(data_dir(owner, repo), "experiment.json")


def state_path(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> str:
    return os.path.join(data_dir(owner, repo), "state.json")


def _load_json(path: str) -> dict | None:
    """Return parsed JSON dict, or None if file missing/malformed/empty."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) and data else None
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def load_experiment(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> dict | None:
    """Read + parse experiment.json. Returns None on missing/malformed file."""
    return _load_json(experiment_path(owner, repo))


def load_state(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> dict | None:
    """Read + parse state.json. Returns None on missing/malformed file."""
    return _load_json(state_path(owner, repo))


# ---------------------------------------------------------------------------
# Formatting helpers (pp = percentage points)
# ---------------------------------------------------------------------------


def pct(accuracy: float | None) -> str:
    """Format a 0-1 accuracy as a percentage string."""
    if accuracy is None:
        return "—"
    return f"{accuracy * 100:.1f}%"


def pp(delta_pp: float | None) -> str:
    """
    Format an improvement given in PERCENTAGE POINTS.

    A value of 11.1 means the accuracy rose by 11.1 percentage points → renders
    as "+11.1 pp". A value of 0 (flat) renders as "0 pp". A negative value is
    shown as its own sign, e.g. "-2.8 pp".
    """
    if delta_pp is None:
        return "—"
    if delta_pp == 0:
        return "0 pp"
    return f"{delta_pp:+.1f} pp"


def improvement_from_baseline_pp(experiment: dict) -> float:
    """
    Return the improvement of the LAST cycle_eval over the baseline, in
    percentage points. Falls back to 0.0 when no data is available.
    """
    baseline = experiment.get("baseline_accuracy")
    cycles = experiment.get("cycle_accuracy") or []
    if baseline is None or not cycles:
        return 0.0
    # Already stored as pp floats; the last element is the most recent eval.
    stored = experiment.get("improvement_from_baseline") or []
    if stored:
        return float(stored[-1])
    return (float(cycles[-1]) - float(baseline)) * 100.0


def is_flat(experiment: dict) -> bool:
    """True when there is no measurable improvement on the held-out eval set."""
    return improvement_from_baseline_pp(experiment) <= 0


def current_eval_accuracy(experiment: dict) -> float | None:
    """Accuracy of the last cycle_eval round, else the last stored eval acc."""
    rounds = experiment.get("rounds") or []
    eval_rounds = [r for r in rounds if r.get("round_type") == "cycle_eval"]
    if eval_rounds:
        return float(eval_rounds[-1].get("accuracy"))
    cycles = experiment.get("cycle_accuracy") or []
    if cycles:
        return float(cycles[-1])
    return None


# ---------------------------------------------------------------------------
# Pure data-extraction helpers (used by both the UI and the tests)
# ---------------------------------------------------------------------------


def memory_counts(experiment: dict) -> dict:
    """memory size + added/merged/revised/rejected counts from experiment."""
    return {
        "added": experiment.get("memory_added", 0),
        "merged": experiment.get("memory_merged", 0),
        "revised": experiment.get("memory_revised", 0),
        "rejected": experiment.get("memory_rejected", 0),
    }


def rounds_frame(experiment: dict) -> pd.DataFrame:
    """Flatten `rounds` into a tidy DataFrame for the EXPERIMENT ROUNDS table."""
    rounds = experiment.get("rounds") or []
    rows = []
    for r in rounds:
        rows.append(
            {
                "Round": (
                    "Baseline" if r.get("round_type") == "baseline_eval"
                    else f"Train Cycle {r.get('cycle')}"
                    if r.get("round_type") == "train_cycle"
                    else f"Eval Cycle {r.get('cycle')}"
                ),
                "Type": r.get("round_type", ""),
                "Issue count": len(r.get("issue_numbers") or []),
                "Accuracy": pct(r.get("accuracy")),
                "Memory size": r.get("memory_size", 0),
                "Tool calls": r.get("tool_calls", 0),
                "Tool errors": r.get("tool_errors", 0),
                "Latency": f"{r.get('average_latency_s', 0.0):.1f}s",
                "Cost": f"${r.get('estimated_cost_usd', 0.0):.4f}",
            }
        )
    return pd.DataFrame(rows)


def train_eval_frame(experiment: dict) -> pd.DataFrame:
    """
    Per-cycle training vs evaluation accuracy (train_cycle vs cycle_eval).

    Returns a DataFrame with columns: Cycle, Train Accuracy, Eval Accuracy.
    """
    cycles: dict[int, dict] = {}
    for r in experiment.get("rounds") or []:
        cyc = r.get("cycle")
        acc = r.get("accuracy")
        if cyc is None or acc is None:
            continue
        bucket = cycles.setdefault(cyc, {"train": None, "eval": None})
        if r.get("round_type") == "train_cycle":
            bucket["train"] = acc
        elif r.get("round_type") == "cycle_eval":
            bucket["eval"] = acc
    return pd.DataFrame(
        [
            {"Cycle": cyc, "Train Accuracy": pct(b["train"]), "Eval Accuracy": pct(b["eval"])}
            for cyc, b in sorted(cycles.items())
        ]
    )


def group_memories(state: dict) -> dict[str, list]:
    """Split state memory into domain_fact and tool_usage lists."""
    grouped = {"domain_fact": [], "tool_usage": []}
    for m in state.get("memory") or []:
        kind = m.get("kind")
        if kind in grouped:
            grouped[kind].append(m)
        # unknown kinds are ignored rather than crashing
    return grouped


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_empty_state() -> None:
    st.balloons()
    st.header("AGENTFORGE")
    st.subheader("Self-Improving GitHub Issue Triage Agent")
    st.caption(f"Repository: {DEFAULT_OWNER}/{DEFAULT_REPO}")
    st.divider()
    st.info(
        "No experiment data yet - run the learning experiment first. "
        "Once `experiment.json` and `state.json` are written under "
        "`data/agents/pallets_flask/`, press the **Refresh** button to load them."
    )


def render_header(owner: str, repo: str) -> None:
    st.set_page_config(page_title="AgentForge — Dashboard", layout="wide")
    st.title("AGENTFORGE")
    st.subheader("Self-Improving GitHub Issue Triage Agent")
    st.caption(f"Repository: {owner}/{repo}")


def render_summary(experiment: dict) -> None:
    st.header("EXPERIMENT SUMMARY")
    b = st.columns(6)
    baseline = experiment.get("baseline_accuracy")
    current = current_eval_accuracy(experiment)
    delta = improvement_from_baseline_pp(experiment)
    eval_issue_numbers = experiment.get("eval_issue_numbers") or []
    memory = memory_counts(experiment)
    b[0].metric("Baseline Accuracy", pct(baseline))
    b[1].metric("Current Evaluation Accuracy", pct(current))
    b[2].metric("Improvement from Baseline", pp(delta))
    b[3].metric("Number of Learning Cycles", int(experiment.get("num_cycles", 0)))
    b[4].metric("Evaluation Set Size", len(eval_issue_numbers))
    b[5].metric("Memory Size", memory["added"])


def render_learning_curve(experiment: dict) -> None:
    st.header("LEARNING CURVE")
    baseline = experiment.get("baseline_accuracy")
    cycle_accuracy = experiment.get("cycle_accuracy") or []
    st.subheader("Evaluation Accuracy Over Time")
    if baseline is None and not cycle_accuracy:
        st.caption("No accuracy data available.")
        return

    # Cycle evaluation points only — plotted exactly as stored, no smoothing.
    points = pd.DataFrame(
        {
            "Cycle": [f"Cycle {i}" for i in range(1, len(cycle_accuracy) + 1)],
            "Accuracy": [float(a) for a in cycle_accuracy],
        }
    )

    try:
        import altair as alt

        chart = (
            alt.Chart(points)
            .mark_line(point=True)
            .encode(
                x=alt.X("Cycle:N", sort=None, title="Learning Cycle"),
                y=alt.Y("Accuracy:Q", scale=alt.Scale(zero=False), title="Evaluation Accuracy"),
                tooltip=["Cycle", "Accuracy"],
            )
        )
        if baseline is not None:
            # Dashed horizontal reference line — visually distinct baseline.
            baseline_df = pd.DataFrame({"Accuracy": [float(baseline)]})
            baseline_rule = (
                alt.Chart(baseline_df)
                .mark_rule(strokeDash=[6, 4], color="#FF4B4B", size=1.5)
                .encode(
                    y="Accuracy:Q",
                    tooltip=alt.Tooltip("Accuracy:Q", title="Baseline Accuracy"),
                )
            )
            st.altair_chart(
                (baseline_rule + chart).properties(height=380),
                width="stretch",
            )
            st.caption(
                f"Red dashed line = baseline ({pct(baseline)}), followed by "
                f"Cycle 1..{len(cycle_accuracy)} evaluation accuracy on the fixed "
                "held-out set. Plotted honestly — no smoothing."
            )
        else:
            st.altair_chart(chart.properties(height=380), width="stretch")
            st.caption(
                "Cycle 1..N evaluation accuracy on the fixed held-out set. "
                "Plotted honestly — no smoothing."
            )
    except ImportError:
        # Altair unavailable — fall back to a plain line chart (still honest).
        data = {"Point": [], "Accuracy": []}
        if baseline is not None:
            data["Point"].append("Baseline")
            data["Accuracy"].append(float(baseline))
        for i, acc in enumerate(cycle_accuracy, start=1):
            data["Point"].append(f"Cycle {i}")
            data["Accuracy"].append(float(acc))
        df = pd.DataFrame(data)
        st.line_chart(df.set_index("Point")["Accuracy"])
        st.caption(
            "Baseline followed by Cycle 1..N evaluation accuracy on the fixed "
            "held-out set. Plotted honestly — no smoothing."
        )


def render_train_vs_eval(experiment: dict) -> None:
    st.header("TRAIN VS EVALUATION")
    frame = train_eval_frame(experiment)
    if frame.empty:
        st.caption("No per-cycle train/eval rounds available.")
        return
    st.dataframe(frame, width="stretch")


def render_memory_quality(experiment: dict, state: dict | None) -> None:
    st.header("MEMORY QUALITY")
    counts = memory_counts(experiment)
    c = st.columns(4)
    c[0].metric("Memory Size", counts["added"])
    c[1].metric("Created", counts["added"])
    c[2].metric("Merged", counts["merged"])
    c[3].metric("Rejected", counts["rejected"])
    if counts["revised"]:
        st.caption(f"Revised: {counts['revised']}")

    if not state or not state.get("memory"):
        st.caption("No memory entries yet.")
        return

    grouped = group_memories(state)
    st.subheader("DOMAIN FACTS")
    for m in grouped["domain_fact"]:
        _memory_card(m)
    st.subheader("TOOL USAGE")
    for m in grouped["tool_usage"]:
        _memory_card(m)


def _memory_card(m: dict) -> None:
    st.markdown(
        f"- **{m.get('statement', '')}** "
        f"_(evidence ×{m.get('evidence_count', 0)}, "
        f"created@step {m.get('created_at_step', 0)}, "
        f"last reinforced@step {m.get('last_reinforced_step', 0)})_"
    )


def render_tool_performance(experiment: dict) -> None:
    st.header("TOOL PERFORMANCE")
    calls = int(experiment.get("tool_calls", 0))
    errors = int(experiment.get("tool_errors", 0))
    error_rate = (errors / calls * 100.0) if calls else 0.0
    eval_count = len(experiment.get("eval_issue_numbers") or [])
    calls_per_issue = (calls / eval_count) if eval_count else 0.0
    c = st.columns(5)
    c[0].metric("Total Tool Calls", calls)
    c[1].metric("Tool Errors", errors)
    c[2].metric("Tool Error Rate", f"{error_rate:.2f}%")
    c[3].metric("Avg Calls / Issue", f"{calls_per_issue:.2f}")
    c[4].metric("Average Latency", f"{float(experiment.get('average_latency', 0.0)):.1f}s")
    st.metric("Estimated Cost", f"${float(experiment.get('estimated_cost', 0.0)):.4f}")


def render_rounds(experiment: dict) -> None:
    st.header("EXPERIMENT ROUNDS")
    frame = rounds_frame(experiment)
    if frame.empty:
        st.caption("No rounds recorded yet.")
    else:
        st.dataframe(frame, width="stretch", hide_index=True)


def render_system_flow() -> None:
    st.header("SYSTEM FLOW")
    flow = [
        "GitHub Issues",
        "Worker Agent",
        "GitHub Tools",
        "Prediction",
        "Deterministic Evaluation",
        "Reflection",
        "Memory Quality Gate",
        "Persistent Memory",
        "Next Learning Cycle ↗",
    ]
    st.markdown("→ ".join(flow))


def render_honest_result(experiment: dict) -> None:
    st.header("HONEST RESULT")
    if is_flat(experiment):
        st.warning(HONEST_RESULT_MESSAGE)
    else:
        delta = improvement_from_baseline_pp(experiment)
        st.success(f"Measured improvement: {pp(delta)} vs baseline.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    owner = DEFAULT_OWNER
    repo = DEFAULT_REPO

    # Refresh re-reads the persisted files from disk each time it is pressed.
    if st.button("↻ Refresh", help="Re-read experiment.json and state.json from disk"):
        st.cache_data.clear()

    experiment = load_experiment(owner, repo)
    state = load_state(owner, repo)

    if experiment is None:
        # Missing/malformed → graceful empty state, never crash.
        render_empty_state()
        return

    with neatlogs.trace("issue-triage-session", kind="WORKFLOW"):
        render_header(experiment.get("owner", owner), experiment.get("repo", repo))
        render_summary(experiment)
        st.divider()
        render_learning_curve(experiment)
        st.divider()
        render_train_vs_eval(experiment)
        st.divider()
        render_memory_quality(experiment, state)
        st.divider()
        render_tool_performance(experiment)
        st.divider()
        render_rounds(experiment)
        st.divider()
        render_system_flow()
        st.divider()
        render_honest_result(experiment)

        st.caption(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            "Read-only dashboard — data from experiment.json / state.json."
        )


if __name__ == "__main__":
    main()
