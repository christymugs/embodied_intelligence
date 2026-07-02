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
import warnings

from learning.env import TaskConfig
from learning.train import TrainConfig
from evolution.evolve import run_evolution, EvolutionConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--tournament-k", type=int, default=3)
    p.add_argument("--metric", choices=["auc", "final", "steps_to_threshold"], default="auc")
    p.add_argument("--eval-seeds", type=int, default=1)
    p.add_argument("--lifetime-steps", type=int, default=200_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--target-distance", type=float, default=0.5,
                   help="Finish line in +x metres (same for every body).")
    p.add_argument("--seed", type=int, default=0)
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

    task = TaskConfig(horizon=a.horizon, target_distance=a.target_distance)
    train = TrainConfig(
        algo=a.algo, lifetime_steps=a.lifetime_steps, eval_every=a.eval_every,
        n_steps=256 if a.smoke else 2048, seed=a.seed,
    )
    evo = EvolutionConfig(
        population_size=a.population, generations=a.generations, elitism=a.elitism,
        tournament_k=a.tournament_k, selection_metric=a.metric,
        eval_seeds=a.eval_seeds, seed=a.seed, out_dir=a.out,
        n_workers=1 if a.serial else a.workers,
    )

    print(f"Evolving for learnability (metric={a.metric}) -> {a.out}")
    res = run_evolution(evo=evo, task=task, train_cfg=train)
    best = res["best"]
    print(f"\nBest body: {best.genome.count_limbs()} limbs, depth {best.genome.max_depth()}, "
          f"fitness {best.fitness:.3f}")
    print(f"Artifacts in {a.out}/  (per-gen best XML + history.json)")


if __name__ == "__main__":
    main()