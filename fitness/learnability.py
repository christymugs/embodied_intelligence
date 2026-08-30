"""Learnability metrics computed from a learning curve.

A learning curve is (timesteps, returns) recorded during the inner RL loop.
Three views of "how well did this body learn", all logged every generation so
the thesis can compare them; one of them drives selection.

  auc                : area under the GAIN curve (return minus its own t=0
                       starting value), normalised by the budget. Rewards fast
                       IMPROVEMENT due to training, not raw ability level.
                       (Default selection.)
  final              : eval return at the end of the fixed budget (Gupta-style,
                       deliberately an ABSOLUTE-performance metric, not a gain
                       one -- this is the intended contrast for --metric auc:
                       if selecting on auc evolves faster learners than
                       selecting on final does, that is the effect).
  steps_to_threshold : timesteps to first reach a target GAIN ("convergence
                       speed"). Capped at the budget if never reached.

auc and steps_to_threshold are both computed on the gain curve, not the raw
return curve. A body whose UNTRAINED (t=0) policy already scores well -- e.g.
from passive dynamics like toppling forward at reset, nothing to do with
learning -- must not score as "learned fast" just because its average level is
high; confirmed happening in practice (see experiments/run14: untrained
baseline climbed from -5.6 to +52.5 across generations while actual learning
gain stayed flat at ~3-4, because raw-return auc rewarded starting good over
improving). Subtracting r[0] before integrating means a flat curve scores ~0
regardless of the height it is flat at.

UNTRAINED_BASELINE_FLOOR guards the mirror-image exploit: a body whose
untrained policy scores catastrophically (not just mediocre) gets "gain"
credit for improving off of it, even if the trained result never becomes
good in absolute terms. Confirmed happening in practice on experiments/
run27: the champion's r[0] (untrained) got MORE negative every generation
(-2040 at gen0 -> -3070 at gen49) while its actual trained return (r[-1])
stayed flat and bad the whole run (-13.8 -> -4.7, never positive) -- auc
kept climbing (1564 -> 3036) purely because the baseline it was measured
against kept getting worse, not because anything learned to walk. The
best individual in that run's ENTIRE final population scored final=30.26;
every other top-10-by-auc individual scored final<=0. Clamping the
baseline used for gain to a floor caps how much "improvement" credit a
catastrophic start can buy, so genuinely reaching a good absolute return
becomes necessary again, the same way the run14 fix (above) closed off
the opposite direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A body scoring below this at t=0 is thrashing catastrophically (see
# experiments/run27, r[0] as low as -3070), not meaningfully differentiated
# from another catastrophic body at, say, -500 -- both are "started broken,"
# and treating -3070 as MORE room to gain than -500 rewards worse starts.
# -50 is comfortably below a real "fell in a few steps" outcome (run27's
# best-by-final individual, which does something, scored final=30.26; the
# uniformly-failed rest of that population scored <=0 but nowhere near
# -1000s) and comfortably above the catastrophic-thrashing floor.
UNTRAINED_BASELINE_FLOOR = -50.0

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

    # Gain over this body's OWN untrained starting point, not raw return level
    # (see module docstring for why) -- floored so a catastrophic start can't
    # buy artificial "room to gain" (see UNTRAINED_BASELINE_FLOOR above).
    baseline = max(r[0], UNTRAINED_BASELINE_FLOOR)
    gain = r - baseline

    # Average gain over training (area under the gain curve / budget).
    auc = float(_trapz(gain, t) / budget) if t.size > 1 else float(gain[-1])
    final = float(r[-1])

    # Default threshold: a fraction of the best GAIN this body ever achieved.
    if threshold is None:
        threshold = threshold_frac * float(np.max(gain))

    crossed = np.where(gain >= threshold)[0]
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