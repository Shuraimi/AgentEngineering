"""
Regression test for a confirmed live bug: LLM-written memory statements can
contain characters that Windows' default cp1252 encoding cannot encode
(e.g. U+2011 NON-BREAKING HYPHEN crashed save_state mid-session with
UnicodeEncodeError). State JSON must always be persisted as UTF-8.
"""

import json

import state_store
from models import AgentState, MemoryEntry


def test_state_round_trips_unicode_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "DATA_DIR", str(tmp_path))

    statement = "Issues mentioning 'AWOL\u2011style' are bug (non\u2011breaking hyphens)."
    state = AgentState(
        owner="pallets", repo="flask",
        core_instructions="Triage.",
        step=1,
        memory=[MemoryEntry(
            id="u1", kind="domain_fact", statement=statement,
            evidence_count=1, created_at_step=1, last_reinforced_step=1,
        )],
    )

    state_store.save_state(state)

    # The file must exist and be valid UTF-8 JSON.
    path = tmp_path / "pallets_flask" / "state.json"
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["memory"][0]["statement"] == statement

    loaded = state_store.load_state("pallets", "flask")
    assert loaded is not None
    assert loaded.memory[0].statement == statement
    assert loaded.step == 1


def test_batch_result_round_trips_unicode(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "DATA_DIR", str(tmp_path))

    from models import BatchResult, CaseResult

    result = BatchResult(
        step=1,
        cases=[CaseResult(
            issue_number=42, predicted_labels=["bug"], actual_labels=["bug"],
            score=1.0, cost_usd=0.0, latency_s=1.0,
        )],
        accuracy=1.0, avg_cost_usd=0.0, avg_latency_s=1.0,
        tool_error_count=0, memory_size_before=0,
    )
    # Put the unicode character somewhere the JSON serialization will carry it.
    result.cases[0].tool_calls = []
    state_store.save_batch_result("pallets", "flask", result)

    history = state_store.load_history("pallets", "flask")
    assert len(history) == 1
    assert history[0].step == 1