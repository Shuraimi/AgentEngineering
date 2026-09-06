"""
Scores one prediction against real ground truth (the maintainers' own
labels on that historical issue).

Deliberately deterministic (Jaccard similarity over the two label sets) -
no LLM-judge call needed. This matters for the "cost-effectiveness" axis the
judges asked about: evaluation itself should be nearly free, so the API
budget goes toward the agent actually doing the task and reflecting on it,
not toward grading.
"""


def score_case(actual_labels: list[str], predicted_labels: list[str]) -> tuple[float, str]:
    actual = set(a.lower() for a in actual_labels)
    predicted = set(p.lower() for p in predicted_labels)

    if not actual and not predicted:
        return 1.0, "both empty - trivially correct"
    if not actual or not predicted:
        return 0.0, f"expected {sorted(actual)}, got {sorted(predicted)}"

    intersection = actual & predicted
    union = actual | predicted
    score = len(intersection) / len(union)
    reason = f"expected {sorted(actual)}, got {sorted(predicted)}, overlap {sorted(intersection)}"
    return score, reason
