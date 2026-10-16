"""Evaluation thresholds matching the reference platform."""

TARGETS = ("broad-spectrum", "escherichia-coli", "staphylococcus-aureus")
AMP_THRESHOLD = 0.998
ECOLI_THRESHOLD = 0.9980798959732056
SAUREUS_THRESHOLD = 0.9989857077598572
NON_HEMOLYTIC_THRESHOLD = 0.5


def evaluate(
    amp: float, ecoli: float, saureus: float, non_hemolytic: float, target: str = "broad-spectrum"
) -> tuple[int, str]:
    """Score only the criteria relevant to the selected generation target."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; choose from {TARGETS}.")
    points = int(amp > AMP_THRESHOLD) + int(non_hemolytic >= NON_HEMOLYTIC_THRESHOLD)
    if target in ("broad-spectrum", "escherichia-coli"):
        points += int(ecoli > ECOLI_THRESHOLD)
    if target in ("broad-spectrum", "staphylococcus-aureus"):
        points += int(saureus > SAUREUS_THRESHOLD)
    maximum = 4 if target == "broad-spectrum" else 3
    category = "Good" if points == maximum else "Mediocre" if points >= 2 else "Bad"
    return points, category
