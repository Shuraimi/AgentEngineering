"""
AgentForge Streamlit UI - GitHub Issue Triage edition.

Same design principle as before: call the individual pieces of the loop
(run_time_step, reflect, apply_reflection) one step at a time from inside
this script, rather than one black-box function, so Streamlit renders each
step LIVE as it happens - the accuracy climbing, the memory panel growing -
instead of showing a spinner and dumping everything at the end.
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import state_store
from github_tools import fetch_repo_labels
from learn_loop import DEFAULT_LABEL_POOL_SIZE, prepare_dataset, run_time_step
from reflector import apply_reflection, build_initial_state, reflect

st.set_page_config(page_title="AgentForge — GitHub Triage", layout="wide")
st.title("🔨 AgentForge — an agent that learns to triage YOUR repo's issues over time")
st.caption(
    "Point it at a real GitHub repo. It replays real historical closed issues in "
    "time order, predicts labels using the repo's own tools, checks itself against "
    "what the maintainers actually labeled them, and grows a persistent memory of "
    "what it's learned — both about the repo's conventions and about using its tools well."
)

with st.sidebar:
    st.header("Session configuration")
    owner = st.text_input("Repo owner", value="pallets")
    repo = st.text_input("Repo name", value="flask")
    num_batches = st.slider("Time steps (batches)", 2, 8, 4)
    batch_size = st.slider("Issues per step", 3, 12, 6)
    token_from_env = os.environ.get("GITHUB_TOKEN")
    st.caption(
        "✅ GITHUB_TOKEN found in environment" if token_from_env
        else "⚠️ No GITHUB_TOKEN set — unauthenticated GitHub API limits are tight (60/hr)."
    )
    st.divider()
    if st.button("🗑️ Reset this agent's memory"):
        state_store.reset_agent(owner, repo)
        st.success(f"Reset agent for {owner}/{repo}. It will start fresh next run.")

existing_state = state_store.load_state(owner, repo)
if existing_state:
    st.info(f"Resuming existing agent for **{owner}/{repo}** — "
            f"already at step {existing_state.step} with {len(existing_state.memory)} learned memory entries.")
else:
    st.info(f"No existing agent for **{owner}/{repo}** yet — this run starts one from scratch.")

run_clicked = st.button("🚀 Run learning session", type="primary")

if run_clicked:
    token = token_from_env
    with st.spinner("Fetching real labels + historical labeled issues from GitHub..."):
        try:
            labels, cases = prepare_dataset(owner, repo, token, num_batches, batch_size)
        except Exception as e:
            st.error(f"Couldn't fetch data from GitHub: {e}")
            st.stop()

    if len(cases) < batch_size:
        st.warning(f"Only found {len(cases)} labeled historical issues for the top "
                   f"{DEFAULT_LABEL_POOL_SIZE} labels — try a more active repo or a smaller batch size.")
        st.stop()

    state = existing_state or build_initial_state(owner, repo, labels)
    if existing_state is None:
        state_store.save_state(state)

    with st.expander("Repo labels this agent is choosing between"):
        st.write(", ".join(f"`{l['name']}`" for l in labels[:DEFAULT_LABEL_POOL_SIZE]))
    with st.expander("Core instructions (stable — should rarely change)"):
        st.code(state.core_instructions)

    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)][:num_batches]
    history = []
    progress_area = st.container()

    for batch in batches:
        if not batch:
            continue
        with progress_area:
            st.markdown(f"### Step {state.step + 1}")
            with st.spinner(f"Running on {len(batch)} historical issues..."):
                result = run_time_step(state, batch, owner, repo, token, labels)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{result.accuracy:.0%}")
            c2.metric("Avg cost / issue", f"${result.avg_cost_usd:.4f}")
            c3.metric("Avg latency / issue", f"{result.avg_latency_s:.2f}s")
            c4.metric("Tool errors", result.tool_error_count)

            with st.spinner("Reflecting on this step's mistakes..."):
                reflection = reflect(state, result)
                state = apply_reflection(state, reflection)

            state_store.save_state(state)
            state_store.save_batch_result(owner, repo, result)
            history.append(result)

            if reflection.memory_additions or reflection.memory_revisions:
                with st.expander(f"🧠 What it learned this step", expanded=True):
                    st.write(reflection.summary)
                    for m in reflection.memory_additions:
                        st.write(f"➕ **[{m.kind}]** {m.statement}")
                    for mid, stmt in reflection.memory_revisions.items():
                        st.write(f"✏️ **[revised {mid}]** {stmt}")

    st.divider()
    st.header("📊 Session report")
    if len(history) >= 2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{history[-1].accuracy:.0%}",
                   delta=f"{(history[-1].accuracy - history[0].accuracy):+.0%} vs step 1")
        c2.metric("Avg cost / issue", f"${history[-1].avg_cost_usd:.4f}",
                   delta=f"{(history[-1].avg_cost_usd - history[0].avg_cost_usd):+.4f} vs step 1",
                   delta_color="inverse")
        c3.metric("Tool errors", history[-1].tool_error_count,
                   delta=f"{(history[-1].tool_error_count - history[0].tool_error_count):+d} vs step 1",
                   delta_color="inverse")

    chart_df = pd.DataFrame({
        "step": [r.step for r in history],
        "accuracy": [r.accuracy for r in history],
        "avg_cost_usd": [r.avg_cost_usd for r in history],
        "memory_size": [r.memory_size_before for r in history[1:]] + [len(state.memory)] if history else [],
    }).set_index("step")
    st.subheader("Accuracy over time")
    st.line_chart(chart_df[["accuracy"]])
    st.subheader("Cost per issue over time")
    st.line_chart(chart_df[["avg_cost_usd"]])
    st.subheader("Memory size over time (the growth the judges want to see)")
    st.bar_chart(chart_df[["memory_size"]])

    st.subheader("Full current memory")
    if state.memory:
        st.dataframe(pd.DataFrame([m.model_dump() for m in state.memory]))
    else:
        st.caption("No memory accumulated yet.")

    with st.expander("Full case-by-case results (final step)"):
        st.dataframe(pd.DataFrame([c.model_dump() for c in history[-1].cases]))
