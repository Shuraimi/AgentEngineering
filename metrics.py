"""
Pure arithmetic for learning metrics - kept dependency-free so the numbers
are testable in isolation and the semantics are impossible to misread.

All improvements are reported in PERCENTAGE POINTS (pp), not percent:
an agent going from 42% to 67% accuracy improved by 25pp. Saying "+25%"
would be wrong - that is the RELATIVE change (.42 -> .67 is +59%).
"""


def percentage_points(new: float, old: float) -> float:
    """Absolute accuracy delta in percentage points, e.g. .42 -> .67 = 25.0.

    Negative when the new measurement is WORSE than the old one - a
    regression is just as visible as an improvement.
    """
    return round((new - old) * 100, 2)


def improvement_from_baseline(cycle_accuracies: list[float], baseline_accuracy: float) -> list[float]:
    """pp each cycle gained over the pre-learning baseline. len == len(cycle_accuracies)."""
    return [percentage_points(acc, baseline_accuracy) for acc in cycle_accuracies]


def improvement_from_previous_cycle(cycle_accuracies: list[float], baseline_accuracy: float) -> list[float]:
    """pp each cycle gained over the PREVIOUS measurement: cycle 1 is compared
    to the baseline, every later cycle to the cycle before it, so a plateau or
    regression between cycles is visible, not hidden by a cumulative number.
    len == len(cycle_accuracies).
    """
    previous = baseline_accuracy
    deltas = []
    for acc in cycle_accuracies:
        deltas.append(percentage_points(acc, previous))
        previous = acc
    return deltas