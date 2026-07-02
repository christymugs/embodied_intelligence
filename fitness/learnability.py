"""Learnability metrics computed from a learning curve.

A learning curve is (timesteps, returns) recorded during the inner RL loop.
Three views of "how well did this body learn", all logged every generation so
the thesis can compare them; one of them drives selection.

  auc                : area under the learning curve, normalised by the budget.
                       = average eval return across training. Rewards FAST early
                       improvement, not just the endpoint. (Default selection.)
  final              : eval return at the end of the fixed budget (Gupta-style).
  steps_to_threshold : timesteps to first reach a target return ("convergence
                       speed"). Capped at the budget if never reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2 renamed it


@dataclass
class LearnabilityResult:
    auc: float
    final: float
    steps_to_threshold: float
    threshold: float
    reached_threshold: bool
    curve_t: list = field(default_factory=list)
    curve_r: list = field(default_factory=list)

    def score(self, metric: str = "auc") -> float:
        """Higher is always better, so steps_to_threshold is negated."""
        if metric == "auc":
            return self.auc
        if metric == "final":
            return self.final
        if metric == "steps_to_threshold":
            return -self.steps_to_threshold
        raise ValueError(f"unknown metric: {metric}")


def compute_learnability(timesteps, returns, threshold: float | None = None,
                         threshold_frac: float = 0.6) -> LearnabilityResult:
    t = np.asarray(timesteps, dtype=float)
    r = np.asarray(returns, dtype=float)
    if t.size == 0:
        return LearnabilityResult(0.0, 0.0, 0.0, 0.0, False, [], [])

    budget = float(t[-1]) if t[-1] > 0 else 1.0

    # Average eval return over training (area under curve / budget).
    auc = float(_trapz(r, t) / budget) if t.size > 1 else float(r[-1])
    final = float(r[-1])

    # Default threshold: a fraction of the best return this body ever reached.
    if threshold is None:
        threshold = threshold_frac * float(np.max(r))

    crossed = np.where(r >= threshold)[0]
    if crossed.size > 0:
        stt = float(t[crossed[0]])
        reached = True
    else:
        stt = budget  # never reached within budget
        reached = False

    return LearnabilityResult(
        auc=auc, final=final, steps_to_threshold=stt, threshold=float(threshold),
        reached_threshold=reached, curve_t=list(t), curve_r=list(r),
    )