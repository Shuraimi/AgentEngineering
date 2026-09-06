"""
Persistence, kept deliberately simple (flat JSON files, no database - same
reasoning as before: zero schema risk in a short build window).

Layout: data/agents/<owner>_<repo>/state.json          <- current AgentState
        data/agents/<owner>_<repo>/history/step_N.json <- each BatchResult

Loading `state.json` is literally "resume this agent where it left off" -
which is the honest, load-bearing proof that memory persists across
sessions, not just within one run of the loop.
"""

import json
import os

from models import AgentState, BatchResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "agents")


def _agent_dir(owner: str, repo: str) -> str:
    path = os.path.join(DATA_DIR, f"{owner}_{repo}")
    os.makedirs(os.path.join(path, "history"), exist_ok=True)
    return path


def load_state(owner: str, repo: str) -> AgentState | None:
    path = os.path.join(_agent_dir(owner, repo), "state.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return AgentState(**json.load(f))


def save_state(state: AgentState) -> None:
    path = os.path.join(_agent_dir(state.owner, state.repo), "state.json")
    with open(path, "w") as f:
        f.write(state.model_dump_json(indent=2))


def save_batch_result(owner: str, repo: str, result: BatchResult) -> None:
    path = os.path.join(_agent_dir(owner, repo), "history", f"step_{result.step}.json")
    with open(path, "w") as f:
        f.write(result.model_dump_json(indent=2))


def load_history(owner: str, repo: str) -> list[BatchResult]:
    hist_dir = os.path.join(_agent_dir(owner, repo), "history")
    results = []
    for fname in sorted(os.listdir(hist_dir)):
        if fname.startswith("step_") and fname.endswith(".json"):
            with open(os.path.join(hist_dir, fname)) as f:
                results.append(BatchResult(**json.load(f)))
    return sorted(results, key=lambda r: r.step)


def reset_agent(owner: str, repo: str) -> None:
    """Wipes state + history for a repo, so a demo can be re-run from scratch."""
    import shutil
    path = _agent_dir(owner, repo)
    if os.path.exists(path):
        shutil.rmtree(path)
