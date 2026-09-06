"""
Tests for the confirmed tool-error fixes:

  1. runner.py: GPT-OSS channel markers (<|channel|>final / <|message|>) in
     the worker's FINAL ANSWER used to break json.loads -> predicted_labels=[]
     for every such issue. Fixed by routing through clean_model_message()
     (the same fix llm_client.call_for_json already had).
  2. github_tools.search_similar_issues: LLM-written queries containing
     AND/OR/NOT or unbalanced quotes made the GitHub Search API return
     HTTP 422, and the error message mislabeled it as "possibly rate-limited".
     Fixed with sanitize_search_query() + accurate error classification.

No network, no API keys: requests.get is monkeypatched.
"""

from types import SimpleNamespace

import pytest

import github_tools
import llm_client
from models import AgentState, IssueCase, MemoryEntry

# ---------------------------------------------------------------------------
# clean_model_message (shared by llm_client + runner)
# ---------------------------------------------------------------------------

def test_clean_model_message_strips_channel_markers():
    raw = ("some reasoning<|channel|>final\n"
           '{"labels": ["bug"]}<|message|>')
    assert llm_client.clean_model_message(raw) == '{"labels": ["bug"]}'


def test_clean_model_message_passthrough_when_no_markers():
    raw = '{"labels": ["bug"]}'
    assert llm_client.clean_model_message(raw) == raw


def test_clean_model_message_handles_none():
    assert llm_client.clean_model_message(None) == ""


# ---------------------------------------------------------------------------
# clean_tool_name (confirmed tool error: channel marker in function name)
# ---------------------------------------------------------------------------

def test_clean_tool_name_strips_channel_marker():
    assert llm_client.clean_tool_name("search_similar_issues<|channel|>commentary") \
        == "search_similar_issues"


def test_clean_tool_name_passthrough_when_clean():
    assert llm_client.clean_tool_name("list_labels") == "list_labels"


def test_clean_tool_name_handles_none_and_empty():
    assert llm_client.clean_tool_name(None) == ""
    assert llm_client.clean_tool_name("") == ""


# ---------------------------------------------------------------------------
# runner: final answer with channel markers parses (confirmed tool error)
# ---------------------------------------------------------------------------

def _fake_response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _state():
    return AgentState(owner="pallets", repo="flask",
                      core_instructions="Triage the issue.", step=1)


def _issue():
    return IssueCase(number=1, title="cors broken", body="cors fails",
                     created_at="2020-01-01T00:00:00Z", actual_labels=["bug"])


def test_runner_parses_channel_marked_final_answer(monkeypatch):
    import runner as runner_mod

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _fake_response(
            "thinking text<|channel|>final {\"labels\": [\"bug\"]}<|message|>"
        ))),
    )
    monkeypatch.setattr(runner_mod, "get_client", lambda: fake_client)

    out = runner_mod.run_worker_agent(_state(), _issue(), tools_registry={})
    assert out["predicted_labels"] == ["bug"]
    assert out["tool_calls"] == []


def _fake_tool_call(name, args_json, tool_call_id="call_1"):
    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments=args_json),
    )


def _fake_tool_response(first_ok=True):
    """First call returns a tool call (channel-marked name), second the final answer."""
    calls = {"n": 0}

    def create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            system_prompt = kw["messages"][0]["content"]
            assert first_ok and "cors" in system_prompt    # memory injected
            message = SimpleNamespace(
                content="",
                tool_calls=[_fake_tool_call(
                    "search_similar_issues<|channel|>commentary", '{"query": "cors preflight"}',
                )],
            )
        else:
            message = SimpleNamespace(content='{"labels": ["bug"]}', tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    return create


def test_runner_resolves_channel_marked_tool_name(monkeypatch):
    """The confirmed live error: function names like
    search_similar_issues<|channel|>commentary used to fail with
    'unknown tool'; they must resolve and execute the real tool."""
    import runner as runner_mod

    executed = {}

    def fake_list_labels():
        return "- bug: a bug"

    def fake_search(query):
        executed["query"] = query
        return "No matching prior issues found."

    registry = {
        "list_labels": {
            "schema": {"name": "list_labels", "description": "list",
                       "input_schema": {"type": "object", "properties": {}}},
            "fn": fake_list_labels,
        },
        "search_similar_issues": {
            "schema": {"name": "search_similar_issues", "description": "search",
                       "input_schema": {"type": "object",
                                        "properties": {"query": {"type": "string"}},
                                        "required": ["query"]}},
            "fn": fake_search,
        },
    }

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=_fake_tool_response(first_ok=True),
        )),
    )
    monkeypatch.setattr(runner_mod, "get_client", lambda: fake_client)

    # Give the agent a learned memory entry mentioning 'cors' so the injected
    # system prompt really contains it (the _fake_tool_response asserts this).
    state = _state()
    state.memory = [
        MemoryEntry(id="m1", kind="domain_fact",
                    statement="issues mentioning 'cors' are labeled 'bug' in this repo",
                    evidence_count=2, created_at_step=1, last_reinforced_step=1),
    ]

    out = runner_mod.run_worker_agent(state, _issue(), tools_registry=registry)

    assert out["predicted_labels"] == ["bug"]
    assert executed["query"] == "cors preflight"          # tool actually ran
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0].tool_name == "search_similar_issues"  # cleaned name
    assert not out["tool_calls"][0].was_error             # no "unknown tool" error


def test_runner_non_dict_tool_args_do_not_crash(monkeypatch):
    """Confirmed live error: the model emitted list_labels(1330) - a bare int.
    json.loads accepted it, ToolCallRecord.tool_input: dict then raised a
    pydantic ValidationError that killed the whole session. It must record a
    graceful tool error instead, with the expected params in the message."""
    import runner as runner_mod

    calls = {"n": 0}
    executed = {"called": False}

    def fake_list_labels():
        executed["called"] = True
        return "- bug: a bug"

    registry = {
        "list_labels": {
            "schema": {"name": "list_labels", "description": "list",
                       "input_schema": {"type": "object", "properties": {}}},
            "fn": fake_list_labels,
        },
    }

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            message = SimpleNamespace(
                content="",
                tool_calls=[_fake_tool_call("list_labels", "1330")],
            )
        else:
            message = SimpleNamespace(content='{"labels": ["bug"]}', tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
    )
    monkeypatch.setattr(runner_mod, "get_client", lambda: fake_client)

    out = runner_mod.run_worker_agent(_state(), _issue(), tools_registry=registry)

    assert out["predicted_labels"] == ["bug"]              # run completed, no crash
    assert not executed["called"]                          # tool never ran with an int
    assert len(out["tool_calls"]) == 1
    rec = out["tool_calls"][0]
    assert rec.was_error
    assert rec.tool_input == {"_raw_args": 1330}           # record is a valid dict
    assert "JSON object of parameters" in rec.tool_output
    assert "Expected params for list_labels: (none - call it with {})" in rec.tool_output


def test_runner_shows_relevant_memory_in_system_prompt(monkeypatch):
    """Learned memories are actually injected into the worker's prompt."""
    import runner as runner_mod

    captured = {}

    def fake_create(**kw):
        captured["system_prompt"] = kw["messages"][0]["content"]
        return _fake_response('{"labels": ["bug"]}')

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
    )
    monkeypatch.setattr(runner_mod, "get_client", lambda: fake_client)

    state = _state()
    state.memory = [
        MemoryEntry(id="m1", kind="domain_fact",
                    statement="issues mentioning 'cors' are labeled 'bug' in this repo",
                    evidence_count=2, created_at_step=1, last_reinforced_step=1),
        MemoryEntry(id="m2", kind="domain_fact",
                    statement="issues mentioning 'templates' are 'documentation'",
                    evidence_count=1, created_at_step=1, last_reinforced_step=1),
    ]

    runner_mod.run_worker_agent(state, _issue(), tools_registry={})

    assert "LEARNED MEMORY" in captured["system_prompt"]
    assert "cors" in captured["system_prompt"]              # relevant memory shown
    assert "(2x)" in captured["system_prompt"]              # reinforcement strength


# ---------------------------------------------------------------------------
# github_tools: search query sanitization (confirmed tool error)
# ---------------------------------------------------------------------------

def test_sanitize_removes_boolean_operators():
    assert github_tools.sanitize_search_query("cors AND preflight OR error NOT fixed") \
        == "cors preflight error fixed"


def test_sanitize_removes_balanced_and_unbalanced_quotes():
    assert github_tools.sanitize_search_query('cors "preflight" broken "') \
        == "cors preflight broken"


def test_sanitize_keeps_plain_phrase():
    assert github_tools.sanitize_search_query("werkzeug dev server crash") \
        == "werkzeug dev server crash"


def test_sanitize_empty_returns_empty():
    assert github_tools.sanitize_search_query("   AND OR NOT \" \" ") == ""


def test_empty_sanitized_query_returns_helpful_error():
    registry = github_tools.build_tool_registry(
        "pallets", "flask", [], None,
        before_date="2020-01-01T00:00:00Z", exclude_number=1,
    )
    out = registry["search_similar_issues"]["fn"]("AND AND")
    assert out.startswith("SEARCH ERROR")
    assert "empty search query" in out


# ---------------------------------------------------------------------------
# github_tools: accurate error classification (confirmed tool error)
# ---------------------------------------------------------------------------

class FakeResp:
    """Stand-in for requests.Response: status_code + optional JSON message."""

    def __init__(self, status_code, message=None):
        self.status_code = status_code
        self._message = message

    def json(self):
        return {"message": self._message} if self._message else {}

    def raise_for_status(self):
        pass


def _registry_with(monkeypatch, fake_resp, before_date="2020-01-01T00:00:00Z",
                   exclude_number=1, items=None):
    holder = {}

    def fake_get(url, headers, params, timeout):
        holder["params"] = params
        if items is None:
            return fake_resp
        return SimpleNamespace(status_code=200, json=lambda: {"items": items})

    monkeypatch.setattr(github_tools.requests, "get", fake_get)
    registry = github_tools.build_tool_registry(
        "pallets", "flask", [{"name": "bug", "description": ""}], "tok",
        before_date=before_date, exclude_number=exclude_number,
    )
    return registry, holder


@pytest.mark.parametrize("status,message,expected", [
    (422, "Validation Failed", "invalid search query syntax"),
    (429, "Rate Limit Exceeded", "rate limited"),
    (403, "Forbidden", "rate limited"),
    (401, "Bad credentials", "GITHUB_TOKEN"),
    (500, None, "HTTP 500"),
])
def test_search_error_messages_name_the_real_cause(monkeypatch, status, message, expected):
    registry, _ = _registry_with(monkeypatch, FakeResp(status, message))
    out = registry["search_similar_issues"]["fn"]("cors broken")
    assert out.startswith("SEARCH ERROR")
    assert expected in out


def test_search_error_no_longer_blames_rate_limit_for_bad_query(monkeypatch):
    registry, _ = _registry_with(monkeypatch, FakeResp(422, "Validation Failed"))
    out = registry["search_similar_issues"]["fn"]("cors broken")
    assert "rate-limited" not in out
    assert "invalid search query syntax" in out


def test_search_similar_builds_sanitized_query_and_filters_temporally(monkeypatch):
    items = [
        {"number": 100, "title": "old cors bug", "created_at": "2019-05-01T00:00:00Z",
         "labels": [{"name": "bug"}]},
        {"number": 7, "title": "future issue", "created_at": "2099-01-01T00:00:00Z",
         "labels": [{"name": "bug"}]},          # after before_date -> filtered out
        {"number": 101, "title": "the issue itself", "created_at": "2020-01-02T00:00:00Z",
         "labels": [{"name": "bug"}]},          # == exclude_number -> filtered out
    ]
    registry, holder = _registry_with(
        monkeypatch, None, before_date="2020-01-03T00:00:00Z", exclude_number=101, items=items,
    )
    out = registry["search_similar_issues"]["fn"]('cors "preflight" AND broken')

    q = holder["params"]["q"]
    assert "AND" not in q
    assert '"' not in q
    assert "cors preflight broken" in q
    assert "#100" in out
    assert "#101" not in out and "future issue" not in out
    assert "No matching" not in out