# AgentForge — Spec

## AgentState schema (source of truth: models.py)

- `owner`, `repo`: which repo this agent's identity belongs to
- `core_instructions`: stable base prompt, patched only for structural fixes
- `memory: list[MemoryEntry]`: grows over time, never wholesale replaced
- `step`: how many time steps this agent has lived through

## MemoryEntry kinds

- `domain_fact`: something true about this repo's own labeling conventions
- `tool_usage`: something about calling the tools effectively (avoiding a
  previously-hit tool error, a search-query pattern that finds precedent)

## Tools available to the worker agent (source of truth: github_tools.py)

| Tool | Real API call | Notes |
|---|---|---|
| list_labels | none (uses labels fetched once at session start) | prevents label hallucination |
| search_similar_issues | GET /search/issues | temporally filtered - no future/self leakage |

## Scoring (source of truth: evaluator.py)

Jaccard similarity between predicted and real (ground-truth) label sets.
Deterministic, no extra LLM call.

## Loop parameters (source of truth: learn_loop.py)

- `num_batches` (time steps), `batch_size` (issues per step)
- `DEFAULT_LABEL_POOL_SIZE = 6`: how many of the repo's most-used labels the
  demo focuses on, to keep the label space tractable

## Team roles

_(fill in)_ — who owns github_tools/runner, who owns evaluator/reflector,
who owns the UI, who owns the AO task queue + PR review, who owns the demo.
