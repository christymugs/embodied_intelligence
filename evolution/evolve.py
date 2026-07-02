"""Outer evolutionary loop: evolve morphologies for LEARNABILITY.

For each individual, train its controller from scratch (inner loop), measure how
efficiently it learned, and select on that signal — not on peak performance.
Over generations this applies the morphological Baldwin pressure the project
investigates: bodies that are intrinsically easier to learn to control come to
dominate the population.

Per-individual evaluations are independent and run in parallel (see
evolution/parallel.py). Everything that happens is logged to {out_dir}/run.log,
{out_dir}/generation_stats.csv, and {out_dir}/history.json.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from morphology.genome import Genome, MorphologyBounds
from morphology.mujoco_builder import MujocoBuilder
from evolution.mutation import mutate, MutationRates
from evolution.population import Individual, tournament_select
from evolution.parallel import worker_init, evaluate_jobs, resolve_workers
from learning.env import TaskConfig
from learning.train import TrainConfig


@dataclasses.dataclass
class EvolutionConfig:
    population_size: int = 32
    generations: int = 50
    elitism: int = 2
    tournament_k: int = 3
    selection_metric: str = "auc"    # "auc" | "final" | "steps_to_threshold"
    eval_seeds: int = 1              # average learnability over this many seeds
    seed: int = 0
    out_dir: str = "experiments/run"
    reevaluate_elites: bool = False  # re-score elites each gen to fight metric noise
    n_workers: int = 0               # 0 => auto (all CPUs); 1 => serial


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def _setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(f"evolve.{out_dir}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(os.path.join(out_dir, "run.log"), mode="w")
    fh.setLevel(logging.DEBUG)        # full detail (incl. per-individual) to file
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)         # summaries to console
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _seed_for(evo_seed: int, individual_id: int, gen: int) -> int:
    return evo_seed + individual_id * 7 + gen * 101


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_evolution(evo: EvolutionConfig | None = None, task: TaskConfig | None = None,
                  train_cfg: TrainConfig | None = None,
                  bounds: MorphologyBounds | None = None,
                  rates: MutationRates | None = None) -> dict:
    evo = evo or EvolutionConfig()
    task = task or TaskConfig()
    train_cfg = train_cfg or TrainConfig()
    bounds = bounds or MorphologyBounds()
    rates = rates or MutationRates()

    rng = np.random.default_rng(evo.seed)
    os.makedirs(evo.out_dir, exist_ok=True)
    # Trained policies are written here (absolute path, so spawned workers on any
    # cwd hit the same place). Keyed by individual id; the best of each generation
    # is promoted to gen{g}_best_policy.zip. The rest stay as an internal cache you
    # don't need to look at (a few MB each) in case you ever want a specific body.
    policies_dir = os.path.abspath(os.path.join(evo.out_dir, "_policies"))
    os.makedirs(policies_dir, exist_ok=True)
    builder = MujocoBuilder()
    log = _setup_logger(evo.out_dir)

    n_workers = resolve_workers(evo.n_workers)
    serial = n_workers <= 1

    # ---- run header ----
    log.info("=" * 64)
    log.info("Learnability-driven morphology evolution")
    log.info("  population=%d  generations=%d  metric=%s  eval_seeds=%d",
             evo.population_size, evo.generations, evo.selection_metric, evo.eval_seeds)
    log.info("  inner: algo=%s  lifetime_steps=%d  eval_every=%d  horizon=%d",
             train_cfg.algo, train_cfg.lifetime_steps, train_cfg.eval_every, task.horizon)
    log.info("  workers=%s  seed=%d  out=%s",
             "serial" if serial else n_workers, evo.seed, evo.out_dir)
    log.info("=" * 64)

    # ---- init population ----
    next_id = 0
    pop: list[Individual] = []
    for _ in range(evo.population_size):
        pop.append(Individual(Genome.random(bounds, rng), id=next_id))
        next_id += 1

    # ---- csv header ----
    csv_path = os.path.join(evo.out_dir, "generation_stats.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "gen", "best_fitness", "mean_fitness", "worst_fitness",
            "best_id", "best_limbs", "best_depth", "n_invalid",
            "gen_seconds", "cumulative_seconds",
        ])

    history = []
    executor = None
    run_start = time.time()
    gen_times: list[float] = []
    try:
        if not serial:
            executor = ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init)

        for gen in range(evo.generations):
            gen_start = time.time()

            # Build the job list (skip already-evaluated unless re-evaluating).
            jobs = []
            for ind in pop:
                if ind.evaluated and not evo.reevaluate_elites:
                    continue
                jobs.append((
                    ind.id, ind.genome, task, train_cfg,
                    evo.eval_seeds, evo.selection_metric,
                    _seed_for(evo.seed, ind.id, gen),
                    os.path.join(policies_dir, f"ind_{ind.id}.zip"),
                ))
            log.info("gen %d: evaluating %d/%d individuals (%s)",
                     gen, len(jobs), len(pop), "serial" if serial else f"{n_workers} workers")

            results = evaluate_jobs(jobs, executor)

            # Assign results back.
            by_id = {ind.id: ind for ind in pop}
            for ind_id, res in results.items():
                ind = by_id[ind_id]
                ind.fitness = res["fitness"]
                ind.metrics = res["metrics"]
                ind.metrics["_curve"] = res.get("curve")
                ind.evaluated = True
                log.debug("  id=%-4d fitness=%9.3f limbs=%2d depth=%d metrics=%s",
                          ind.id, ind.fitness, ind.genome.count_limbs(),
                          ind.genome.max_depth(),
                          {k: v for k, v in ind.metrics.items() if k != "_curve"})

            # Rank + stats.
            pop.sort(key=lambda i: i.fitness, reverse=True)
            best = pop[0]
            finite = [i.fitness for i in pop if np.isfinite(i.fitness)]
            n_invalid = sum(1 for i in pop if not np.isfinite(i.fitness))
            mean_fit = float(np.mean(finite)) if finite else float("-inf")
            worst_fit = float(min(finite)) if finite else float("-inf")
            gen_seconds = time.time() - gen_start
            gen_times.append(gen_seconds)
            cum_seconds = time.time() - run_start

            log.info("gen %d done | best %.3f (id=%d, %d limbs) | mean %.3f | worst %.3f "
                     "| invalid %d | %.1fs",
                     gen, best.fitness, best.id, best.genome.count_limbs(),
                     mean_fit, worst_fit, n_invalid, gen_seconds)

            # ETA.
            if gen < evo.generations - 1:
                eta = np.mean(gen_times) * (evo.generations - gen - 1)
                log.info("  ~%.1f min/gen, est. %.1f min remaining",
                         np.mean(gen_times) / 60, eta / 60)

            # Persist: best XML, best genome (to rebuild the env), best policy
            # (to replay it), CSV row, rich history.
            builder.save(best.genome.to_robot_tree(),
                         os.path.join(evo.out_dir, f"gen{gen:03d}_best.xml"))
            with open(os.path.join(evo.out_dir, f"gen{gen:03d}_best_genome.pkl"), "wb") as f:
                pickle.dump(best.genome, f)
            # The best body's policy was saved by its worker to _policies/ind_{id}.zip.
            # For an elite carried over unevaluated, that file still exists from the
            # generation it was trained in, so this resolves either way.
            src_policy = os.path.join(policies_dir, f"ind_{best.id}.zip")
            if os.path.exists(src_policy):
                shutil.copy(src_policy, os.path.join(evo.out_dir, f"gen{gen:03d}_best_policy.zip"))
            else:
                log.debug("  no saved policy for best id=%d (invalid?); skipped promote", best.id)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    gen, best.fitness, mean_fit, worst_fit, best.id,
                    best.genome.count_limbs(), best.genome.max_depth(),
                    n_invalid, round(gen_seconds, 2), round(cum_seconds, 2),
                ])
            history.append(_history_entry(gen, pop, best, gen_seconds))
            with open(os.path.join(evo.out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

            if gen == evo.generations - 1:
                break

            elites = pop[: evo.elitism]
            next_pop = [
                Individual(e.genome.copy(), id=e.id, fitness=e.fitness,
                           evaluated=True, metrics=dict(e.metrics))
                for e in elites
            ]
            while len(next_pop) < evo.population_size:
                parent = tournament_select(pop, evo.tournament_k, rng)
                next_pop.append(Individual(mutate(parent.genome, rng, rates), id=next_id))
                next_id += 1
            pop = next_pop
    finally:
        if executor is not None:
            executor.shutdown()

    total_min = (time.time() - run_start) / 60
    log.info("=" * 64)
    log.info("Finished %d generations in %.1f min. Best fitness %.3f (%d limbs).",
             evo.generations, total_min, pop[0].fitness, pop[0].genome.count_limbs())
    log.info("Artifacts: run.log, generation_stats.csv, history.json, and per generation "
             "gen*_best.xml + gen*_best_genome.pkl + gen*_best_policy.zip in %s",
             evo.out_dir)
    return {"history": history, "best": pop[0]}


def _history_entry(gen: int, pop: list[Individual], best: Individual, gen_seconds: float) -> dict:
    """Rich per-generation record: summary + per-individual rows + best curve."""
    def clean(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "_curve"}

    return {
        "gen": gen,
        "gen_seconds": round(gen_seconds, 2),
        "best_id": best.id,
        "best_fitness": best.fitness,
        "best_limbs": best.genome.count_limbs(),
        "best_depth": best.genome.max_depth(),
        "best_metrics": clean(best.metrics),
        "best_curve": best.metrics.get("_curve"),   # (t, r) — for Baldwin-effect plots
        "population": [
            {
                "id": i.id,
                "fitness": i.fitness,
                "limbs": i.genome.count_limbs(),
                "depth": i.genome.max_depth(),
                "metrics": clean(i.metrics),
            }
            for i in pop
        ],
    }