"""Turn an evolution run's history.json into thesis figures + a diagnostic.

    python -m scripts.plot_run --run experiments/run1

Produces, in the run directory:
  fitness_over_generations.png  best / mean / worst fitness vs generation
  learning_curves.png           best-body learning curve per generation
                                (light = early gen, dark = late gen) — the
                                morphological-Baldwin plot
  steps_to_threshold.png        convergence speed of the best body vs generation
  best_body_size.png            limb count of the best body vs generation

It also prints a short diagnostic, because a rising fitness alone does NOT prove
the Baldwin effect: AUC mixes "learns fast" with "is just a good locomotor". The
learning-curve SHAPE is what distinguishes them — see the printed notes.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(run_dir: str) -> list:
    path = os.path.join(run_dir, "history.json")
    with open(path) as f:
        return json.load(f)


def _finite(xs):
    return [x for x in xs if np.isfinite(x)]


def plot_fitness(history, out):
    gens = [h["gen"] for h in history]
    best = [h["best_fitness"] for h in history]
    mean, worst = [], []
    for h in history:
        fits = _finite([p["fitness"] for p in h["population"]])
        mean.append(np.mean(fits) if fits else np.nan)
        worst.append(np.min(fits) if fits else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(gens, worst, best, alpha=0.12, color="C0", label="worst–best range")
    ax.plot(gens, best, "-o", color="C3", label="best")
    ax.plot(gens, mean, "-o", color="C0", label="mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (AUC = mean eval return over training)")
    ax.set_title("Learnability fitness over generations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_learning_curves(history, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(history)
    cmap = plt.cm.viridis
    plotted = 0
    for h in history:
        c = h.get("best_curve")
        if not c or not c.get("t"):
            continue
        ax.plot(c["t"], c["r"], color=cmap(h["gen"] / max(1, n - 1)), alpha=0.85, lw=1.5)
        plotted += 1
    ax.set_xlabel("training timesteps (one lifetime)")
    ax.set_ylabel("evaluation return")
    ax.set_title("Best-body learning curves across generations")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(1, n - 1)))
    fig.colorbar(sm, ax=ax, label="generation (light = early, dark = late)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return plotted


def plot_steps_to_threshold(history, out):
    gens = [h["gen"] for h in history]
    stt = [h["best_metrics"].get("steps_to_threshold", np.nan) for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, stt, "-o", color="C2")
    ax.set_xlabel("generation")
    ax.set_ylabel("steps to reach 60% of own best return")
    ax.set_title("Convergence speed of the best body (lower = learns faster)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_body_size(history, out):
    gens = [h["gen"] for h in history]
    limbs = [h["best_limbs"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, limbs, "-o", color="C4")
    ax.set_xlabel("generation")
    ax.set_ylabel("limb count of best body")
    ax.set_title("Best-body morphological complexity over generations")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def diagnose(history) -> None:
    """Distinguish real learnability gains from the passive-locomotion pathology.

    Two things can make AUC rise across generations:
      (1) bodies learn MORE / FASTER within a lifetime  -> the Baldwin effect
      (2) bodies just start higher untrained (better passive movers) -> a confound
    We separate them by tracking, across generations, the within-lifetime GAIN
    (trained - untrained) and the untrained BASELINE (the t=0 point), plus the
    trend in steps_to_threshold.
    """
    print("\n--- diagnostic ---")

    gens, gains, baselines, stt = [], [], [], []
    for h in history:
        c = h.get("best_curve")
        if not c or len(c.get("r", [])) < 2:
            continue
        gens.append(h["gen"])
        baselines.append(float(c["r"][0]))                 # untrained start
        gains.append(float(c["r"][-1] - c["r"][0]))        # within-lifetime gain
        stt.append(float(h["best_metrics"].get("steps_to_threshold", np.nan)))

    if len(gens) < 3:
        print("  not enough generations with curves to assess a trend.")
        return

    # Compare the first third vs the last third of the run.
    k = max(1, len(gens) // 3)
    early_gain, late_gain = np.mean(gains[:k]), np.mean(gains[-k:])
    early_base, late_base = np.mean(baselines[:k]), np.mean(baselines[-k:])
    stt_slope = float(np.polyfit(gens, np.nan_to_num(stt, nan=0.0), 1)[0])

    print(f"  within-lifetime GAIN  (trained - untrained):  early {early_gain:+.2f}  ->  late {late_gain:+.2f}")
    print(f"  untrained BASELINE    (t=0 score):            early {early_base:+.2f}  ->  late {late_base:+.2f}")
    print(f"  steps_to_threshold trend per generation:      {stt_slope:+.0f}  (negative = converges faster)")

    learns = late_gain > 1.0                       # do late bodies actually learn within a life?
    gain_grew = late_gain - early_gain > 0.5       # does within-life learning grow over generations?
    baseline_grew = late_base - early_base > 2.0   # are bodies mainly getting better untrained?
    faster = stt_slope < 0

    print()
    if not learns and baseline_grew:
        print("  VERDICT: PASSIVE-LOCOMOTION CONFOUND. Curves are ~flat within a lifetime")
        print("  but their untrained baseline rises across generations. Evolution improved")
        print("  bodies that move well with ANY policy, not bodies that LEARN. (This is the")
        print("  run1 pathology.) Make the task harder to solve untrained before claiming a")
        print("  Baldwin effect.")
    elif learns and (gain_grew or faster):
        print("  VERDICT: BALDWIN SIGNAL. Bodies learn within a lifetime, AND later")
        print("  generations learn more and/or reach threshold faster. This is the result")
        print("  your thesis is after — back it with learning_curves.png and steps_to_threshold.png.")
    elif learns:
        print("  VERDICT: bodies learn within a lifetime, but there's no clear speed-up")
        print("  across generations yet. Selection may need more generations, or a shorter")
        print("  lifetime budget so faster learners stand out more.")
    else:
        print("  VERDICT: little learning in either early or late bodies. Likely the lifetime")
        print("  budget is too short, or most bodies can't get off the ground. Check a few")
        print("  with scripts.check_learning before reading into the evolution result.")

    best_fit = [h["best_fitness"] for h in history]
    print(f"\n  best fitness: {best_fit[0]:.2f} (gen {history[0]['gen']}) -> "
          f"{best_fit[-1]:.2f} (gen {history[-1]['gen']})")
    print("  For a clean control, also run evolution with --metric final and compare:")
    print("  if --metric auc evolves FASTER learners than --metric final does, that's the effect.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Run directory containing history.json")
    a = p.parse_args()

    history = load_history(a.run)
    if not history:
        print("history.json is empty.")
        return

    plot_fitness(history, os.path.join(a.run, "fitness_over_generations.png"))
    n = plot_learning_curves(history, os.path.join(a.run, "learning_curves.png"))
    plot_steps_to_threshold(history, os.path.join(a.run, "steps_to_threshold.png"))
    plot_body_size(history, os.path.join(a.run, "best_body_size.png"))

    print(f"Wrote 4 figures to {a.run}/ ({len(history)} generations, "
          f"{n} learning curves)")
    diagnose(history)


if __name__ == "__main__":
    main()