# AgentForge — GitHub Issue Triage Agent That Learns Over Time

**Syndicate by Maximor, Track 1: Automated Agent Engineering**

AgentForge is a single, persistent agent that triages a real GitHub repo's
issues — predicting which of the repo's own labels apply — and genuinely
gets better at it over time. It does this by replaying real historical
closed-and-labeled issues in time order across sequential "steps," checking
its predictions against what the maintainers actually labeled them, and
self-reflecting after each step to grow a persistent memory: both **domain
facts** about this specific repo's conventions, and **tool-usage lessons**
about using its GitHub tools effectively.

## Problem & target users

Teams that maintain busy repos re-explain the same triage conventions to
every new maintainer/bot rule by hand. AgentForge is an agent that starts
knowing nothing about a repo's specific conventions, and — using only the
repo's real historical data, accessed through the real GitHub API — builds
up that institutional knowledge itself, with the accumulation visible and
auditable (it's a plain memory file, not a black box).

## What makes this "learning," not just "prompting"

- **One agent, not disposable candidates.** There's a single `AgentState`
  per repo (`models.py`). It's loaded, used, updated, and saved — the same
  identity persists across every run, exactly like resuming a person's
  memory rather than hiring a new one each time.
- **Memory grows, it isn't replaced.** Each step's `Reflection` *adds to*
  memory (`reflector.apply_reflection`); existing entries are only touched
  when explicitly revised. Growth is quality-controlled by a deterministic
  gate (`memory.py`), not just left to the LLM: duplicate proposals are
  merged into the existing entry (evidence reinforced), contradictory
  proposals are detected and resolved as revisions, generic boilerplate is
  rejected before it enters memory, and only the most relevant entries (per
  issue, capped) are injected into the worker's prompt. The Streamlit UI
  charts this growth directly.
- **Two distinct kinds of learning**, because the brief asks for both:
  `domain_fact` entries (what's true about *this repo*) and `tool_usage`
  entries (what works when calling *these tools*) — see `reflector.py`.
- **Real third-party tool use**, not a toy function. `github_tools.py` hits
  the live GitHub REST/Search API. The agent must call `list_labels` to see
  real valid label names (guessing a plausible-but-wrong one is a tracked
  tool error) and `search_similar_issues` to find real precedent.

## Evaluation methodology (the honest version)

There's no live stream of brand-new issues to score against during a
30-hour build, so this project uses **replay of real historical data** as a
stand-in for "time passing" — a legitimate, common technique for evaluating
sequential/continual-learning systems:

1. Pull the repo's real labels, and a real, time-ordered sample of closed
   issues that already carry a label from the top N most-used labels
   (`github_tools.build_historical_dataset`, using the Search API — verified
   necessary because most *recently*-closed issues turn out to be unlabeled
   in practice, see comments in that file).
2. Slice that sample into sequential batches by real creation date.
3. At each step, the agent predicts labels for that batch's issues using
   **only** its current memory + tools — it cannot see the real labels.
4. Score = Jaccard similarity between predicted and real label sets
   (`evaluator.py`) — deterministic, free, no extra LLM call needed.
5. **No temporal leakage**: `search_similar_issues` filters out any issue
   created on/after the issue currently being judged, and excludes the
   issue's own number — verified with a dedicated test (see git history /
   AO_LOG.md), not just assumed.
6. After scoring, the Reflector sees only the *failing* cases (score < 0.5)
   plus their tool-call traces and proposes new/revised memory.

This means the accuracy-over-time chart in the UI is a real measurement
against real ground truth the model never saw, not a curated demo number.

## Setup

```bash
git clone <this repo>
cd agentforge
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set LLM_API_KEY, and GITHUB_TOKEN (strongly recommended - see .env.example)
```

## Running it

```bash
# No-UI end-to-end test first (always do this before the UI):
python run_cli.py pallets flask --batches 4 --batch-size 6

# Then the dashboard:
streamlit run app.py
```

Pick a repo with: a small, clean-ish label taxonomy (5-10 meaningful
labels), and at least ~30-40 historically labeled closed issues. Very large
repos (e.g. `microsoft/vscode`) tend to have messy, high-cardinality label
sets that make for a noisier demo; a mid-size, well-maintained repo works
better. You can also point it at your own repo.

To re-run a demo from a clean slate: `python run_cli.py <owner> <repo> --reset`,
or use the "Reset this agent's memory" button in the sidebar.

## Cost / speed notes

- Evaluation is deterministic (no LLM-judge call) — the only paid calls per
  issue are the worker agent's own turns (capped at `MAX_TURNS = 6` in
  `runner.py`) plus one Reflector call per step (not per issue).
- The `tool_error_count` metric tracked per step is a direct, honest
  cost/speed signal: an agent that has learned the real label names and
  effective search queries makes fewer wasted tool round-trips than one
  that hasn't yet — this is expected to visibly drop across steps.

## Built with Agent Orchestrator (AO)

Built end-to-end by delegating tasks to AO, driving Claude Code, in isolated
git worktrees. See `AO_LOG.md` for the task-by-task log.

## Project structure

```
models.py         # shared Pydantic models - AgentState is the one persistent identity
github_tools.py   # the real third-party integration: GitHub REST + Search API
llm_client.py     # shared OpenAI-compatible client + robust JSON parsing
runner.py         # executes the worker agent (tool-use loop) on one real issue
evaluator.py      # deterministic Jaccard scoring against real ground truth
reflector.py      # builds initial instructions; self-reflects to grow/revise memory
state_store.py    # persists AgentState + batch history as JSON, resumable
learn_loop.py     # the main loop: fetch real data once, step through time
app.py            # Streamlit dashboard
run_cli.py        # no-UI end-to-end test harness
```


## LLM provider

AgentForge uses an OpenAI-compatible API so the runtime model can be changed
without changing the Builder/Reflector/Runner architecture.

The default configuration targets NVIDIA's hosted API:

- Base URL: https://integrate.api.nvidia.com/v1
- Model: openai/gpt-oss-20b

Copy `.env.example` to `.env`, set `LLM_API_KEY`, and run:

    python test_llm.py

Then run the normal AgentForge smoke test:

    python run_cli.py pallets flask --batches 4 --batch-size 6
