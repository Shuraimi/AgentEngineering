"""
Tests for the read-only AgentForge Streamlit dashboard.

Covers the 7 REQUIRED cases from the task spec:
  1. dashboard loads existing experiment.json
  2. dashboard loads existing state.json
  3. missing experiment file handled gracefully
  4. missing state file handled gracefully
  5. actual metrics are not hardcoded (values come from the JSON files)
  6. flat 30.6% evaluation result is displayed correctly
  7. percentage-point values displayed correctly (pp, not %)

These tests exercise the pure data/formatting layer of app.py so they run
without a streamlit server process.
"""

import json
import os
import sys
import tempfile
from unittest import mock

import pytest

# Make app.py importable from its own directory.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import app as dashboard  # noqa: E402

REAL_EXPERIMENT = os.path.join(HERE, "data", "agents", "pallets_flask", "experiment.json")
REAL_STATE = os.path.join(HERE, "data", "agents", "pallets_flask", "state.json")


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _flat_experiment() -> dict:
    """A flat fixture: baseline and every cycle eval at 30.56% → 0 pp."""
    return {
        "owner": "pallets",
        "repo": "flask",
        "num_cycles": 2,
        "baseline_accuracy": 0.3056,
        "cycle_accuracy": [0.3056, 0.3056],
        "improvement_from_baseline": [0.00, 0.00],
        "improvement_from_previous_cycle": [0.00, 0.00],
        "memory_added": 2,
        "memory_merged": 0,
        "memory_revised": 0,
        "memory_rejected": 0,
        "tool_calls": 20,
        "tool_errors": 2,
        "average_latency": 22.0,
        "estimated_cost": 0.0,
        "eval_issue_numbers": [100, 200],
        "rounds": [
            {"round_type": "baseline_eval", "cycle": 0, "accuracy": 0.3056,
             "issue_numbers": [100, 200], "tool_calls": 4, "tool_errors": 0,
             "average_latency_s": 20.0, "estimated_cost_usd": 0.0, "memory_size": 0},
            {"round_type": "train_cycle", "cycle": 1, "accuracy": 0.3333,
             "issue_numbers": [1, 2, 3], "tool_calls": 5, "tool_errors": 1,
             "average_latency_s": 21.0, "estimated_cost_usd": 0.0, "memory_size": 0},
            {"round_type": "cycle_eval", "cycle": 1, "accuracy": 0.3056,
             "issue_numbers": [100, 200], "tool_calls": 5, "tool_errors": 1,
             "average_latency_s": 21.5, "estimated_cost_usd": 0.0, "memory_size": 1},
            {"round_type": "train_cycle", "cycle": 2, "accuracy": 0.3056,
             "issue_numbers": [4, 5, 6], "tool_calls": 3, "tool_errors": 0,
             "average_latency_s": 22.0, "estimated_cost_usd": 0.0, "memory_size": 1},
            {"round_type": "cycle_eval", "cycle": 2, "accuracy": 0.3056,
             "issue_numbers": [100, 200], "tool_calls": 3, "tool_errors": 0,
             "average_latency_s": 23.0, "estimated_cost_usd": 0.0, "memory_size": 2},
        ],
    }


class TestLoadsExistingData:
    def test_loads_existing_experiment_json(self):
        exp = dashboard.load_experiment()
        assert exp is not None
        assert exp["owner"] == "pallets"
        assert exp["repo"] == "flask"
        assert isinstance(exp["num_cycles"], int)
        assert exp["num_cycles"] >= 1
        assert isinstance(exp["baseline_accuracy"], (int, float))
        assert "rounds" in exp
        assert isinstance(exp["rounds"], list) and len(exp["rounds"]) > 0

    def test_loads_existing_state_json(self):
        state = dashboard.load_state()
        assert state is not None
        assert state["owner"] == "pallets"
        assert state["repo"] == "flask"
        assert isinstance(state["memory"], list)
        # Actual memories listed, grouped by kind.
        grouped = dashboard.group_memories(state)
        all_kinds = set(m["kind"] for m in state["memory"])
        assert all_kinds <= {"domain_fact", "tool_usage"}


class TestMissingFiles:
    def test_missing_experiment_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                dashboard, "data_dir", return_value=os.path.join(tmp, "agents", "pallets_flask")
            ):
                assert dashboard.load_experiment() is None
                # Empty-state does not raise.
                with mock.patch.object(dashboard.st, "balloons", lambda: None):
                    dashboard.render_empty_state()

    def test_missing_state_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                dashboard, "data_dir", return_value=os.path.join(tmp, "agents", "pallets_flask")
            ):
                assert dashboard.load_state() is None
                # UI still builds out of the experiment even with no state file.
                exp = _flat_experiment()
                st_mock = mock.MagicMock()
                st_mock.empty.return_value  # no-op
                with mock.patch.object(dashboard, "st", st_mock):
                    dashboard.render_memory_quality(exp, None)  # state is None

    def test_malformed_experiment_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "experiment.json")
            _write_json(path, "not valid json {{{")
            with mock.patch.object(
                dashboard, "experiment_path", return_value=path
            ):
                assert dashboard.load_experiment() is None

    def test_malformed_state_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            _write_json(path, "also not json")
            with mock.patch.object(dashboard, "state_path", return_value=path):
                assert dashboard.load_state() is None


class TestNoHardcodedMetrics:
    def test_metrics_come_from_json_not_hardcoded(self):
        exp = dashboard.load_experiment()
        assert exp is not None
        # The dashboard reads these OUT OF the JSON; therefore values must
        # match the persisted file exactly, proving they're not baked in.
        eval_rounds = [r for r in exp["rounds"] if r.get("round_type") == "cycle_eval"]
        assert dashboard.current_eval_accuracy(exp) == pytest.approx(
            eval_rounds[-1]["accuracy"]
        )
        assert dashboard.improvement_from_baseline_pp(exp) == pytest.approx(
            exp["improvement_from_baseline"][-1]
        )
        # Changing a value in the JSON must change what the dashboard reports.
        exp["baseline_accuracy"] = 0.9999
        exp["improvement_from_baseline"] = [12.34, 56.78]
        assert dashboard.improvement_from_baseline_pp(exp) == pytest.approx(56.78)


class TestFlatResult:
    def test_flat_306_result_shows_0_pp_and_honest_message(self):
        exp = _flat_experiment()
        assert exp["baseline_accuracy"] == pytest.approx(0.3056)
        delta = dashboard.improvement_from_baseline_pp(exp)
        assert delta == 0.0
        assert dashboard.pp(delta) == "0 pp"
        assert dashboard.is_flat(exp) is True
        assert dashboard.HONEST_RESULT_MESSAGE == (
            "No measurable improvement on the fixed held-out evaluation set."
        )
        # current eval accuracy is 30.56% (the flat value, not something else)
        assert dashboard.current_eval_accuracy(exp) == pytest.approx(0.3056)


class TestPPFormatting:
    def test_3056_to_4167_renders_plus_111_pp_not_percent(self):
        baseline = 0.3056
        current = 0.4167
        delta = (current - baseline) * 100.0
        assert delta == pytest.approx(11.11)
        rendered = dashboard.pp(delta)
        assert rendered == "+11.1 pp"
        assert "%" not in rendered  # never "+11.1%"
        # And the accuracy itself is displayed as a percentage.
        assert dashboard.pct(current) == "41.7%"
        assert dashboard.pct(baseline) == "30.6%"

    def test_flat_renders_zero_pp(self):
        assert dashboard.pp(0.0) == "0 pp"

    def test_negative_pp_keeps_sign(self):
        assert dashboard.pp(-2.78) == "-2.8 pp"

    def test_real_delta_from_persisted_file(self):
        exp = dashboard.load_experiment()
        assert exp is not None
        rendered = dashboard.pp(dashboard.improvement_from_baseline_pp(exp))
        assert rendered.endswith("pp")
        assert "%" not in rendered


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
