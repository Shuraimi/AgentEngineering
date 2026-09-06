"""
Deterministic, dependency-free memory quality gate for AgentForge.

The LLM Reflector PROPOSES memory; this module DISPOSES it. Everything here
is pure string/heuristic logic (no LLM call, no vector DB, no new deps), so
the quality rules are stable, testable, and cheap:

  - near-duplicate proposals merge into the existing entry (the existing
    statement is kept, its evidence_count is bumped) instead of bloating
    memory with two copies of the same lesson;
  - contradictory proposals are DETECTED and resolved: the newer proposal
    replaces the stale entry (counted as a revision), which is exactly what
    "the agent corrected itself after seeing new evidence" means;
  - overly generic statements ("read the issue carefully") are rejected
    before they ever enter memory - generic advice that would apply to any
    repo is not learned knowledge about THIS repo;
  - relevance ranking decides which memories are injected into a worker
    prompt: per-issue token overlap first, then reinforcement strength and
    recency, capped so the prompt stays small and focused;
  - a hard cap (MAX_MEMORY_SIZE) keeps total growth bounded - because memory
    is injected into the prompt, unbounded growth would degrade the agent.
"""

import re

MAX_MEMORY_SIZE = 24          # hard cap on total entries, keeps the prompt stable
MAX_MEMORY_PER_PROMPT = 12    # how many ranked entries a worker prompt may show

# Phrases that signal instruction-restatement rather than repo-specific
# knowledge. Lowercase, matched against the normalized statement.
GENERIC_PHRASES = [
    "read the issue", "read the title", "read the body", "understand the issue",
    "be careful", "be thoughtful", "make sure", "always check", "remember to",
    "consider the", "think about", "good practice", "common sense",
    "use your judgment", "pay attention", "look at the", "first step",
    "it is important", "the best way", "in general", "generally speaking",
    "carefully", "thoroughly",
]

# Function words that carry no topical content. NOTE: negation words are
# deliberately NOT here - "not"/"never"/"avoid" are the signal we use to
# detect contradictions between otherwise overlapping statements.
STOPWORDS = {
    "the", "a", "an", "is", "are", "be", "to", "of", "for", "on", "in",
    "with", "and", "or", "this", "that", "it", "its", "as", "at", "by",
    "from", "you", "your", "should", "will", "would", "can", "could", "may",
    "might", "must", "do", "does", "did", "when", "where", "which", "who",
    "how", "what", "than", "then", "there", "their", "they", "them", "we",
    "our", "us", "if", "but", "so", "up", "down", "out", "off", "over",
    "under", "again", "further", "once", "having", "been", "being", "have",
    "has", "had", "into", "onto", "about", "than", "s", "t", "ll", "re",
    "ve", "d", "m",
}

NEGATION_WORDS = {
    "not", "never", "isnt", "arent", "dont", "doesnt", "didnt", "wont",
    "cannot", "cant", "shouldnt", "wouldnt", "avoid", "avoiding", "no",
    "without", "instead",
}

REL_DUPLICATE = "duplicate"
REL_CONTRADICTION = "contradiction"
REL_INDEPENDENT = "independent"


def normalize_statement(text: str) -> str:
    """Lowercase, strip punctuation/quotes, collapse whitespace."""
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokens(text: str) -> set[str]:
    """Topical tokens: >=3 chars, not stopwords (negations kept)."""
    return {w for w in normalize_statement(text).split() if len(w) >= 3 and w not in STOPWORDS}


def _negated(text: str) -> bool:
    return any(w in NEGATION_WORDS for w in normalize_statement(text).split())


def relation(existing: str, proposed: str) -> str:
    """
    Classify the relationship between two memory statements:

      - duplicate: same lesson, stated the same way or paraphrased
      - contradiction: same topic, conflicting claim (different label,
        direct negation, ...)
      - independent: unrelated or overlapping-but-compatible

    Deterministic token-overlap classification; see module docstring.
    """
    te, tp = _tokens(existing), _tokens(proposed)
    union = te | tp
    if not union:
        return REL_INDEPENDENT
    inter = te & tp
    overlap = len(inter) / len(union)

    # Direct negation of an otherwise-overlapping claim is always a conflict,
    # even when the negated statement is a word-subset of the other.
    if _negated(existing) != _negated(proposed) and overlap >= 0.45:
        return REL_CONTRADICTION

    if overlap >= 0.6:
        extra_e = te - tp
        extra_p = tp - te
        if not extra_e or not extra_p:
            # One statement's content is a subset of the other's - paraphrase.
            return REL_DUPLICATE
        # Both sides contribute differing specifics (e.g. different labels).
        return REL_CONTRADICTION

    # Very short statements (e.g. "auth issues 'bug'" vs "auth issues
    # 'feature'") land at ~0.5 overlap because the single differing token is a
    # large fraction of the union; a shared core with differing specifics is
    # still a conflict, not a paraphrase.
    if (overlap >= 0.5 and len(union) <= 4
            and (te - tp) and (tp - te)):
        return REL_CONTRADICTION

    return REL_INDEPENDENT


def is_duplicate(existing: str, proposed: str) -> bool:
    return relation(existing, proposed) == REL_DUPLICATE


def is_contradiction(existing: str, proposed: str) -> bool:
    return relation(existing, proposed) == REL_CONTRADICTION


def is_generic(statement: str) -> bool:
    """True when a proposed statement is repo-agnostic boilerplate."""
    n = normalize_statement(statement)
    if not n:
        return True
    hits = sum(1 for p in GENERIC_PHRASES if p in n)
    substantive = _tokens(statement)
    if hits >= 2:
        return True
    if len(n) < 15 and hits >= 1:
        return True
    if len(substantive) < 2:
        return True
    return False


def rank_memories(memory: list, issue_text: str = "", top_k: int = MAX_MEMORY_PER_PROMPT) -> list:
    """
    Order memory for injection into a worker prompt.

    Primary: per-issue relevance - how many of the statement's topical tokens
    appear in the current issue's title/body. This is what lets the worker
    "receive relevant learned memories" instead of a static dump.
    Tie-breakers: evidence_count (reinforcement strength), then recency.
    """
    issue_tokens = _tokens(issue_text) if issue_text else set()

    def key(m):
        overlap = len(_tokens(m.statement) & issue_tokens) if issue_tokens else 0
        return (overlap, m.evidence_count, m.last_reinforced_step, len(m.statement))

    return sorted(memory, key=key, reverse=True)[:top_k]