"""Outcome label from a tumor-size trajectory (live tumor cells per frame)."""


def classify(tumor):
    final, peak = tumor[-1], max(tumor)
    if final == 0:
        return "elimination"
    if final < 0.05 * peak:
        return "residual"  # deep regression, small population left
    q3 = tumor[len(tumor) * 3 // 4]
    if final >= 0.95 * peak or (final > 0.8 * peak and final > 1.5 * q3):
        return "escape"
    return "equilibrium"
