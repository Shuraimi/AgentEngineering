"""
Hermetic tests for the Phase 4 "learning metrics" protocol (run_learning_experiment).

No network, no LLM: every learn_loop dependency is replaced with a deterministic
fake so the protocol invariants are tested exactly:

  1. the baseline is recorded BEFORE any learning, with empty memory;
  2. every cycle re-evaluates the SAME held-out eval set;
  3. the Reflector only ever sees TRAIN issues - eval results can never be
     learned from (memory grows only between train cycles);
  4. improvements are reported in PERCENTAGE POINTS, not percent;
  5. the full result persists to experiment.json and round-trips.

Fake scoring: eval issues (numbers 200-299) score 0.4 + 0.1 per memory entry,
so accuracy climbs 0.4 -> 0.5 -> 0.6 as the agent learns; train issues
(numbers 100-199) score a flat 0.5 and emit one tool error each.
"""

import json

import pytest

import learn_loop
import state_store
from models import (
    AgentState,
    BatchResult,
    CaseResult,
    IssueCase,
    MemoryChange,
    MemoryEntry,
    Reflection,
    ToolCallRecord,
)


def _make_cases(start: int, n: int) -> list[IssueCase]:
    return [
        IssueCase(
            number=num, title=f"issue {num}", body="body",
            created_at="2024-01-01T00:00:00Z", actual_labels=["bug"],
        )
        for num in range(start, start + n)
    ]


def _fake_prepare_dataset(owner, repo, token, num_batches, batch_size, eval_size=0):
    labels = [{"name": "bug", "description": "a bug"}]
    train = _make_cases(101, num_batches * batch_size)
    holdout = _make_cases(201, eval_size)
    return labels, train + holdout


def _fake_run_time_step(state, batch, owner, repo, token, labels):
    case_results = []
    tool_errors = 0
    for issue in batch:
        if 200 <= issue.number < 300:
            # eval issues: accuracy improves as memory grows (learning works)
            score = min(1.0, 0.4 + 0.1 * len(state.memory))
        else:
            # train issues: flat, and each produces one tool error
            score = 0.5
            tool_errors += 1
        case_results.append(CaseResult(
            issue_number=issue.number,
            predicted_labels=["bug"],
            actual_labels=issue.actual_labels,
            tool_calls=[
                ToolCallRecord(tool_name="list_labels", tool_input={}, tool_output="[bug]"),
                ToolCallRecord(tool_name="search_similar_issues", tool_input={"query": "x"}, tool_output="none"),
            ],
            score=score,
            cost_usd=0.001,
            latency_s=1.0,
        ))
    return BatchResult(
        step=state.step + 1,
        cases=case_results,
        accuracy=sum(c.score for c in case_results) / len(case_results),
        avg_cost_usd=0.001,
        avg_latency_s=1.0,
        tool_error_count=tool_errors,
        memory_size_before=len(state.memory),
    )


def _make_reflect(rec):
    """Reflector fake that records what it saw and proposes one new memory entry."""
    def reflect(state, batch_result):
        rec["calls"] += 1
        rec["numbers"].extend(c.issue_number for c in batch_result.cases)
        mid = f"m{len(state.memory) + 1}"
        return Reflection(
            summary="learned one lesson",
            memory_additions=[MemoryEntry(
                id=mid, kind="domain_fact", statement=f"lesson {len(state.memory) + 1}",
                created_at_step=state.step + 1, last_reinforced_step=state.step + 1,
            )],
        )
    return reflect


def _fake_apply_reflection(state, reflection):
    new_memory = [m.model_copy(deep=True) for m in state.memory]
    created = []
    for addition in reflection.memory_additions:
        new_memory.append(addition.model_copy(deep=True))
        created.append(addition.id)
    next_state = AgentState(
        owner=state.owner, repo=state.repo,
        core_instructions=state.core_instructions,
        memory=new_memory, step=state.step + 1,
    )
    return next_state, MemoryChange(created=created)


def _fake_build_initial_state(owner, repo, labels):
    return AgentState(owner=owner, repo=repo, core_instructions="Triage.")


def _wire_fakes(monkeypatch, tmp_path, rec):
    monkeypatch.setattr(learn_loop, "prepare_dataset", _fake_prepare_dataset)
    monkeypatch.setattr(learn_loop, "run_time_step", _fake_run_time_step)
    monkeypatch.setattr(learn_loop, "reflect", _make_reflect(rec))
    monkeypatch.setattr(learn_loop, "apply_reflection", _fake_apply_reflection)
    monkeypatch.setattr(learn_loop, "build_initial_state", _fake_build_initial_state)
    monkeypatch.setattr(state_store, "DATA_DIR", str(tmp_path))
    rec["calls"] = 0
    rec["numbers"] = []


def test_baseline_and_improvements_in_percentage_points(monkeypatch, tmp_path):
    rec = {}
    _wire_fakes(monkeypatch, tmp_path, rec)

    state, exp = learn_loop.run_learning_experiment("pallets", "flask", None)

    assert exp.num_cycles == 2
    assert exp.baseline_accuracy == pytest.approx(0.4)         # empty memory
    assert exp.cycle_accuracy == [pytest.approx(0.5), pytest.approx(0.6)]
    # pp over baseline: 50% - 40% = +10 pp; 60% - 40% = +20 pp
    assert exp.improvement_from_baseline == [10.0, 20.0]
    # pp vs previous measurement: cycle 1 vs baseline, cycle 2 vs cycle 1
    assert exp.improvement_from_previous_cycle == [10.0, 10.0]

    # learning happened: one reflection per cycle advanced both step and memory
    assert state.step == 2
    assert len(state.memory) == 2


def test_same_eval_set_used_every_cycle(monkeypatch, tmp_path):
    rec = {}
    _wire_fakes(monkeypatch, tmp_path, rec)
    _, exp = learn_loop.run_learning_experiment("pallets", "flask", None)

    eval_rounds = [r for r in exp.rounds if r.round_type == "cycle_eval"]
    assert len(eval_rounds) == 2
    expected = list(range(201, 207))
    assert all(r.issue_numbers == expected for r in eval_rounds)
    assert exp.eval_issue_numbers == expected
    assert exp.train_issue_numbers == list(range(101, 113))


def test_agent_never_learns_from_eval_set(monkeypatch, tmp_path):
    rec = {}
    _wire_fakes(monkeypatch, tmp_path, rec)
    _, exp = learn_loop.run_learning_experiment("pallets", "flask", None)

    # Reflector ran exactly once per cycle - never for the baseline or eval rounds.
    assert rec["calls"] == 2
    # ...and only ever saw TRAIN issues.
    assert rec["numbers"]
    assert all(100 <= n < 200 for n in rec["numbers"])

    # Memory only grows between train cycles: baseline(0), train1(0), eval1(1),
    # train2(1), eval2(2) - the eval rounds run on the post-training memory but
    # add nothing themselves.
    assert [r.memory_size for r in exp.rounds] == [0, 0, 1, 1, 2]


def test_short_train_slice_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "DATA_DIR", str(tmp_path))

    def short_train(owner, repo, token, num_batches, batch_size, eval_size=0):
        return [{"name": "bug", "description": "a bug"}], _make_cases(101, 4) + _make_cases(201, eval_size)

    monkeypatch.setattr(learn_loop, "prepare_dataset", short_train)
    with pytest.raises(ValueError, match="training issues"):
        learn_loop.run_learning_experiment("pallets", "flask", None)


def test_short_eval_slice_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "DATA_DIR", str(tmp_path))

    def short_eval(owner, repo, token, num_batches, batch_size, eval_size=0):
        return [{"name": "bug", "description": "a bug"}], _make_cases(101, 12) + _make_cases(201, 3)

    monkeypatch.setattr(learn_loop, "prepare_dataset", short_eval)
    with pytest.raises(ValueError, match="held-out eval"):
        learn_loop.run_learning_experiment("pallets", "flask", None)


def test_experiment_persists_and_round_trips(monkeypatch, tmp_path):
    rec = {}
    _wire_fakes(monkeypatch, tmp_path, rec)
    _, exp = learn_loop.run_learning_experiment("pallets", "flask", None)

    path = tmp_path / "pallets_flask" / "experiment.json"
    assert path.exists()

    loaded = state_store.load_experiment_result("pallets", "flask")
    assert loaded is not None
    assert loaded.baseline_accuracy == pytest.approx(0.4)
    assert loaded.cycle_accuracy == [pytest.approx(0.5), pytest.approx(0.6)]
    assert loaded.improvement_from_baseline == [10.0, 20.0]
    assert loaded.improvement_from_previous_cycle == [10.0, 10.0]
    assert [r.round_type for r in loaded.rounds] == [
        "baseline_eval", "train_cycle", "cycle_eval",
        "train_cycle", "cycle_eval",
    ]
    assert loaded.train_issue_numbers == list(range(101, 113))
    assert loaded.eval_issue_numbers == list(range(201, 207))

    # aggregated usage/cost/latency: 5 rounds x 6 issues x 2 tool calls = 60;
    # train rounds only produce the 12 tool errors; $0.006 x 5 rounds; 1.0s each.
    assert loaded.tool_calls == 60
    assert loaded.tool_errors == 12
    assert loaded.average_latency == pytest.approx(1.0)
    assert loaded.estimated_cost == pytest.approx(0.03)
    assert loaded.memory_added == 2
    assert loaded.memory_merged == 0

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["eval_issue_numbers"] == list(range(201, 207))
    assert payload["rounds"][0]["round_type"] == "baseline_eval"


def test_eval_size_passed_to_prepare_dataset(monkeypatch, tmp_path):
    rec = {"calls": 0, "numbers": []}
    captured = {}

    def spy_prepare(owner, repo, token, num_batches, batch_size, eval_size=0):
        captured.update(num_batches=num_batches, batch_size=batch_size, eval_size=eval_size)
        return ([{"name": "bug", "description": "a bug"}],
                _make_cases(101, num_batches * batch_size) + _make_cases(201, eval_size))

    monkeypatch.setattr(learn_loop, "prepare_dataset", spy_prepare)
    monkeypatch.setattr(learn_loop, "run_time_step", _fake_run_time_step)
    monkeypatch.setattr(learn_loop, "reflect", _make_reflect(rec))
    monkeypatch.setattr(learn_loop, "apply_reflection", _fake_apply_reflection)
    monkeypatch.setattr(learn_loop, "build_initial_state", _fake_build_initial_state)
    monkeypatch.setattr(state_store, "DATA_DIR", str(tmp_path))

    learn_loop.run_learning_experiment(
        "pallets", "flask", None, train_batch_size=4, num_cycles=3, eval_size=5,
    )
    assert captured == {"num_batches": 3, "batch_size": 4, "eval_size": 5}


def test_metrics_are_percentage_points_not_percent():
    from metrics import (
        improvement_from_baseline,
        improvement_from_previous_cycle,
        percentage_points,
    )
    assert percentage_points(0.67, 0.42) == 25.0    # "+25 pp" is correct, "+25%" is not
    assert percentage_points(0.42, 0.42) == 0.0
    assert percentage_points(0.30, 0.42) == -12.0   # regressions stay visible
    assert improvement_from_baseline([0.67, 0.75], 0.42) == [25.0, 33.0]
    assert improvement_from_previous_cycle([0.67, 0.75], 0.42) == [25.0, 8.0]