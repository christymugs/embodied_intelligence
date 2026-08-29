"""Run learnability-driven morphology evolution.

Quick pipeline check (seconds):
    python -m scripts.run_evolution --smoke

A small real run (local; minutes-to-hours depending on budget):
    python -m scripts.run_evolution --population 16 --generations 20 \
        --lifetime-steps 100000 --out experiments/run1

Scaling up (cluster): increase --population, --generations, --lifetime-steps and
--eval-seeds. Cost ~ population * generations * lifetime_steps * eval_seeds env
steps, so a serious run is cluster-scale — parallelise individual evaluations
(each is independent) across workers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone

from learning.env import TaskConfig
from learning.train import TrainConfig
from evolution.evolve import run_evolution, EvolutionConfig


def _git_commit() -> str:
    """Current commit hash, with a '-dirty' suffix if the working tree has
    uncommitted changes -- so a reported run can never be silently mistaken
    for one it doesn't actually match (e.g. a fix made after the last commit
    but before this run)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, check=True, text=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception as e:
        return f"unknown ({e})"


def _save_run_manifest(out_dir: str, args: argparse.Namespace, task: TaskConfig,
                        train: TrainConfig, evo: EvolutionConfig) -> None:
    """Write the exact code version + fully-resolved config a run used, so a
    reported result is traceable later -- previously only the CLI args
    appeared as text in the log header, with no record of which commit
    produced them, even though the reward function and morphology genome
    have both changed multiple times over the life of this project."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "cli_args": vars(args),
        "task_config": dataclasses.asdict(task),
        "train_config": dataclasses.asdict(train),
        "evolution_config": dataclasses.asdict(evo),
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Task defaults come from TaskConfig itself, not a second hardcoded number
    # here, so this script and anything else constructing TaskConfig() directly
    # (e.g. scripts/diagnose_selfcollision.py) can't silently drift apart.
    _default_task = TaskConfig()
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--tournament-k", type=int, default=3)
    p.add_argument("--metric", choices=["auc", "final", "steps_to_threshold"], default="auc")
    p.add_argument("--eval-seeds", type=int, default=1)
    p.add_argument("--lifetime-steps", type=int, default=200_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--horizon", type=int, default=_default_task.horizon)
    p.add_argument("--action-rate-weight", type=float, default=_default_task.action_rate_weight,
                   help="0 = off (default). Penalizes action CHANGE between consecutive "
                        "steps (not just magnitude), discouraging a jittery/vibrating gait. "
                        "Disabled by default after 0.1 stalled learning entirely in a prior "
                        "A/B test (see learning/env.py) -- try a much smaller value (e.g. "
                        "0.01) as a gentler nudge, not assumed safe at any weight.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--novelty-weight", type=float, default=0.0,
                   help="0 = pure fitness selection (default). >0 blends in a hybrid "
                        "morphology+behavior novelty bonus, ranked against an archive "
                        "of past individuals, so selection keeps exploring different "
                        "body plans instead of only refining the current best.")
    p.add_argument("--novelty-k", type=int, default=5,
                   help="k nearest neighbors used to score novelty.")
    p.add_argument("--novelty-archive-size", type=int, default=500)
    p.add_argument("--morphology-novelty-weight", type=float, default=0.5,
                   help="Split within the novelty score: 1.0 = morphology-only, "
                        "0.0 = behavior-only, 0.5 = even hybrid (default).")
    p.add_argument("--complexity-weight", type=float, default=0.0,
                   help="0 = off (default). Subtracted from fitness as "
                        "complexity_weight * num_actuators -- parsimony pressure, "
                        "so a more complex body must clearly outperform a simpler "
                        "one to still win, not just tie it.")
    p.add_argument("--out", type=str, default="experiments/run")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers for evaluation. 0 = auto (all CPUs).")
    p.add_argument("--serial", action="store_true",
                   help="Force single-process evaluation (clearer tracebacks).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny settings to verify the pipeline in seconds.")
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    a = parse_args()

    if a.smoke:
        a.population, a.generations, a.elitism = 4, 2, 1
        a.lifetime_steps, a.eval_every, a.horizon = 1500, 500, 120
        a.out = "experiments/smoke"

    task = TaskConfig(horizon=a.horizon, action_rate_weight=a.action_rate_weight)
    train = TrainConfig(
        algo=a.algo, lifetime_steps=a.lifetime_steps, eval_every=a.eval_every,
        n_steps=256 if a.smoke else 2048, seed=a.seed,
    )
    evo = EvolutionConfig(
        population_size=a.population, generations=a.generations, elitism=a.elitism,
        tournament_k=a.tournament_k, selection_metric=a.metric,
        eval_seeds=a.eval_seeds, seed=a.seed, out_dir=a.out,
        n_workers=1 if a.serial else a.workers,
        novelty_weight=a.novelty_weight, novelty_k=a.novelty_k,
        novelty_archive_size=a.novelty_archive_size,
        morphology_novelty_weight=a.morphology_novelty_weight,
        complexity_weight=a.complexity_weight,
    )

    _save_run_manifest(a.out, a, task, train, evo)
    print(f"Evolving for learnability (metric={a.metric}) -> {a.out}")
    res = run_evolution(evo=evo, task=task, train_cfg=train)
    best = res["best"]
    print(f"\nBest body: {best.genome.count_limbs()} limbs, depth {best.genome.max_depth()}, "
          f"fitness {best.fitness:.3f}")
    print(f"Artifacts in {a.out}/  (per-gen best XML + history.json)")


if __name__ == "__main__":
    main()