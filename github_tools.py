"""
Real GitHub REST API integration - this IS the "third-party app" the worker
agent operates on. Two distinct responsibilities, kept in one file because
they share the same auth/request plumbing:

  1. DATASET PREP (build_historical_dataset): a one-time pull, at session
     start, of real closed+labeled issues, used as held-out ground truth to
     simulate "time passing" across sequential batches. Uses the Search API
     because (as verified live against real repos before writing this file)
     recently-closed issues are mostly unlabeled - you have to search by
     label to find a labeled historical sample at all.

  2. RUNTIME TOOLS (build_tool_registry): what the worker agent can actually
     call while triaging ONE issue. Built fresh per-issue (not a static
     global registry) so we can bake in a temporal cutoff: search results
     are filtered to issues created strictly before the issue being judged,
     which is what makes this an honest simulation of "the agent could only
     have seen the past at this point in time" rather than a leaky one.

Auth: a GITHUB_TOKEN is strongly recommended (60 req/hr unauthenticated vs.
5000 req/hr with a token - a plain classic PAT with no scopes checked is
enough for read-only use). Live label-writing (apply_labels_live) requires a
token with `repo` or `public_repo` scope and is OFF by default everywhere in
this project - see the dry_run flag.
"""

import re
import time
import requests

from models import ToolCallRecord

GITHUB_API = "https://api.github.com"

# GitHub's Search API rejects these boolean operators (422 "Validation
# Failed") - it uses implicit AND with space-separated terms instead.
# LLM-written queries frequently include them, which turns a legit search
# into a tool error.
_UNSUPPORTED_OPERATORS = re.compile(r"\b(?:AND|OR|NOT)\b", re.IGNORECASE)


def _headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_repo_labels(owner: str, repo: str, token: str | None = None) -> list[dict]:
    """Real labels that exist on this repo right now. Fetched once per
    session and handed to the worker agent - this is what stops it from
    hallucinating a plausible-sounding label that doesn't actually exist."""
    resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/labels",
        headers=_headers(token), params={"per_page": 100}, timeout=15,
    )
    resp.raise_for_status()
    return [{"name": l["name"], "description": l.get("description") or ""} for l in resp.json()]


def build_historical_dataset(
    owner: str, repo: str, label_pool: list[str], token: str | None = None,
    per_label: int = 12, total_target: int = 40,
) -> list[dict]:
    """
    Pull real closed issues that carry at least one label from `label_pool`,
    by querying the Search API once per label and merging/deduping results.
    Returns them sorted oldest-first, which is what lets us slice them into
    sequential batches that stand in for "time passing."
    """
    seen: dict[int, dict] = {}
    for label in label_pool:
        if len(seen) >= total_target:
            break
        query = f'repo:{owner}/{repo} is:issue is:closed label:"{label}"'
        resp = requests.get(
            f"{GITHUB_API}/search/issues",
            headers=_headers(token),
            params={"q": query, "per_page": per_label, "sort": "created", "order": "asc"},
            timeout=15,
        )
        if resp.status_code != 200:
            continue  # e.g. rate limited on one label - skip, don't kill the whole pull
        for item in resp.json().get("items", []):
            if item["number"] not in seen:
                seen[item["number"]] = {
                    "number": item["number"],
                    "title": item["title"],
                    "body": (item.get("body") or "")[:1500],  # cap body length -> cost control
                    "created_at": item["created_at"],
                    "actual_labels": [l["name"] for l in item["labels"]],
                }
        time.sleep(0.3)  # be polite to the unauthenticated/low rate-limit case

    return sorted(seen.values(), key=lambda x: x["created_at"])


# ---------------------------------------------------------------------------
# Runtime tools available to the worker agent while it triages ONE issue
# ---------------------------------------------------------------------------

def sanitize_search_query(query: str) -> str:
    """
    Make an LLM-written search query safe to hand to GitHub's Search API.

    Two failure modes killed real tool calls during diagnosis:
      1. Boolean operators (AND/OR/NOT) - GitHub returns HTTP 422
         "Validation Failed" for them (it uses implicit AND).
      2. Unbalanced double quotes - also 422; the lexer never closes the
         phrase. Strip all quotes (a broader, valid search beats an error).

    Deterministic, dependency-free.
    """
    q = query or ""
    q = _UNSUPPORTED_OPERATORS.sub(" ", q)
    q = q.replace('"', "")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _search_error_message(resp: requests.Response) -> str:
    """
    Classify a failed GitHub Search API response so the error the worker (and
    the Reflector) sees names the REAL cause instead of a generic
    "possibly rate-limited" - otherwise tool_usage memory learns the wrong
    lesson. GitHub 422 = bad query syntax, 403/429 = rate limit, 401 = auth.
    """
    code = resp.status_code
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = str(body.get("message", "") or "").strip()
    except Exception:
        pass

    if code in (403, 429):
        return f"SEARCH ERROR: HTTP {code} - rate limited or quota exhausted"
    if code == 422:
        reason = detail or "invalid search query syntax"
        return f"SEARCH ERROR: HTTP 422 - invalid search query syntax ({reason})"
    if code == 401:
        return "SEARCH ERROR: HTTP 401 - GITHUB_TOKEN missing, invalid, or expired"
    if code == 404:
        return "SEARCH ERROR: HTTP 404 - repo not found or search unavailable"
    return f"SEARCH ERROR: HTTP {code} - {detail or 'GitHub search failed'}"


def build_tool_registry(
    owner: str, repo: str, labels: list[dict], token: str | None,
    before_date: str, exclude_number: int,
) -> dict:
    """
    Returns {tool_name: {"schema": ..., "fn": ...}} scoped to one issue's
    evaluation. `before_date` + `exclude_number` enforce the temporal cutoff
    described in the module docstring.
    """

    def list_labels() -> str:
        return "\n".join(f"- {l['name']}: {l['description']}" for l in labels)

    def search_similar_issues(query: str) -> str:
        clean_query = sanitize_search_query(query)
        if not clean_query:
            return ("SEARCH ERROR: empty search query - include keywords from "
                    "the issue's title or body")
        search_query = f'repo:{owner}/{repo} is:issue is:closed {clean_query}'
        resp = requests.get(
            f"{GITHUB_API}/search/issues",
            headers=_headers(token),
            params={"q": search_query, "per_page": 8, "sort": "created", "order": "desc"},
            timeout=15,
        )
        if resp.status_code != 200:
            return _search_error_message(resp)
        results = []
        for item in resp.json().get("items", []):
            # Temporal cutoff: only issues strictly before this one, and not itself.
            if item["number"] == exclude_number or item["created_at"] >= before_date:
                continue
            label_names = [l["name"] for l in item["labels"]]
            results.append(f"#{item['number']} \"{item['title'][:70]}\" -> labels: {label_names}")
            if len(results) >= 5:
                break
        return "\n".join(results) if results else "No matching prior issues found."

    return {
        "list_labels": {
            "schema": {
                "name": "list_labels",
                "description": "List the exact valid labels that exist on this repo. "
                                "Always check this before proposing a label name.",
                "input_schema": {"type": "object", "properties": {}},
            },
            "fn": lambda: list_labels(),
        },
        "search_similar_issues": {
            "schema": {
                "name": "search_similar_issues",
                "description": "Search this repo's PAST closed issues for precedent "
                                "(e.g. similar error messages or symptoms) to inform labeling.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            "fn": lambda query: search_similar_issues(query),
        },
    }


def apply_labels_live(owner: str, repo: str, issue_number: int, labels: list[str],
                       token: str, dry_run: bool = True) -> str:
    """
    The ONE write action this project supports, and it is dry_run=True by
    default everywhere it's called. Only flip dry_run=False deliberately, on
    a repo you own, for a live demo moment - never during evaluation, since
    evaluation replays historical issues that are already closed and labeled.
    """
    if dry_run:
        return f"[DRY RUN] would apply labels {labels} to {owner}/{repo}#{issue_number}"
    resp = requests.post(
        f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/labels",
        headers=_headers(token), json={"labels": labels}, timeout=15,
    )
    resp.raise_for_status()
    return f"Applied labels {labels} to {owner}/{repo}#{issue_number}"
