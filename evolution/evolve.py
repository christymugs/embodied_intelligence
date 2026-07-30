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
from morphology.descriptor import morphology_descriptor
from evolution.mutation import mutate, MutationRates
from evolution.population import Individual, tournament_select
from evolution.parallel import worker_init, evaluate_jobs, resolve_workers
from evolution.novelty import NoveltyArchive, compute_novelty, percentile_ranks
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
    # Novelty bonus blended into SELECTION only (tournament/elitism), never into
    # reported fitness. 0.0 = pure fitness selection (today's behavior, exact
    # passthrough). >0.0 blends in a hybrid morphology+behavior novelty score,
    # ranked against an archive of past individuals, so the search keeps
    # exploring different body plans instead of only refining the current best.
    novelty_weight: float = 0.0
    novelty_k: int = 5
    novelty_archive_size: int = 500
    morphology_novelty_weight: float = 0.5
    # Parsimony pressure: subtracted from fitness as complexity_weight *
    # num_actuators, applied where fitness itself is computed (parallel.py), so
    # it shapes which body is reported/promoted as "best" each generation, not
    # just an invisible tie-breaker. 0.0 = off (today's behavior). A simpler
    # body no longer just ties a complex one on raw task performance -- the
    # complex one has to clear this bar too to still win.
    complexity_weight: float = 0.0


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
            "gen_seconds", "cumulative_seconds", "mean_novelty", "best_novelty",
        ])

    history = []
    executor = None
    run_start = time.time()
    gen_times: list[float] = []
    archive = NoveltyArchive(max_size=evo.novelty_archive_size)
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
                    evo.complexity_weight,
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
                          {k: v for k, v in ind.metrics.items()
                           if k not in ("_curve", "_morphology_descriptor", "_behavior_descriptor")})

            # ---- Novelty scoring: informs SELECTION only, never reported fitness. ----
            for ind in pop:
                ind.metrics["_morphology_descriptor"] = morphology_descriptor(ind.genome).tolist()
                if ind.id in results:  # elites keep their carried-over descriptor otherwise
                    ind.metrics["_behavior_descriptor"] = results[ind.id].get("behavior_descriptor")

            novelty_scores = compute_novelty(
                [np.array(ind.metrics["_morphology_descriptor"]) for ind in pop],
                [ind.metrics.get("_behavior_descriptor") for ind in pop],
                archive, k=evo.novelty_k, morphology_weight=evo.morphology_novelty_weight,
            )
            for ind, score in zip(pop, novelty_scores):
                ind.metrics["novelty"] = score

            # Archive only genuinely-new-this-generation individuals (not elites
            # re-seen every gen), so long-lived elites don't dominate the archive.
            for ind_id in results:
                ind = by_id[ind_id]
                if np.isfinite(ind.fitness):
                    archive.add(np.array(ind.metrics["_morphology_descriptor"]),
                                ind.metrics.get("_behavior_descriptor"))

            if evo.novelty_weight <= 0.0:
                for ind in pop:
                    ind.selection_score = ind.fitness
            else:
                finite_idx = [i for i, ind in enumerate(pop) if np.isfinite(ind.fitness)]
                fitness_ranks = percentile_ranks([pop[i].fitness for i in finite_idx])
                novelty_ranks = percentile_ranks([pop[i].metrics["novelty"] for i in finite_idx])
                for ind in pop:
                    ind.selection_score = float("-inf")
                w = evo.novelty_weight
                for rank_pos, i in enumerate(finite_idx):
                    pop[i].selection_score = (1 - w) * fitness_ranks[rank_pos] + w * novelty_ranks[rank_pos]

            # Rank + stats (reporting stays pure fitness -- novelty never
            # contaminates the learnability measurement being written out).
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
            mean_novelty = float(np.mean([ind.metrics.get("novelty", 0.0) for ind in pop
                                          if np.isfinite(ind.fitness)])) if finite else 0.0
            best_novelty = best.metrics.get("novelty", 0.0)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    gen, best.fitness, mean_fit, worst_fit, best.id,
                    best.genome.count_limbs(), best.genome.max_depth(),
                    n_invalid, round(gen_seconds, 2), round(cum_seconds, 2),
                    round(mean_novelty, 4), round(best_novelty, 4),
                ])
            history.append(_history_entry(gen, pop, best, gen_seconds))
            with open(os.path.join(evo.out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

            if gen == evo.generations - 1:
                break

            elites = sorted(pop, key=lambda i: i.selection_score, reverse=True)[: evo.elitism]
            next_pop = [
                Individual(e.genome.copy(), id=e.id, fitness=e.fitness,
                           selection_score=e.selection_score,
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
    """Rich per-generation record: summary + per-individual rows + best curve.

    Per-individual rows keep their morphology/behavior descriptors (needed to
    measure population diversity across a run, not just track the champion),
    but not the learning curve -- that one's genuinely large and only useful
    for the champion. `best_metrics` excludes both descriptors since they're
    already promoted as their own top-level `best_*` keys below.
    """
    def clean_best(m: dict) -> dict:
        return {k: v for k, v in m.items()
                if k not in ("_curve", "_morphology_descriptor", "_behavior_descriptor")}

    def clean_population(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "_curve"}

    return {
        "gen": gen,
        "gen_seconds": round(gen_seconds, 2),
        "best_id": best.id,
        "best_fitness": best.fitness,
        "best_limbs": best.genome.count_limbs(),
        "best_depth": best.genome.max_depth(),
        "best_metrics": clean_best(best.metrics),
        "best_curve": best.metrics.get("_curve"),   # (t, r) — for Baldwin-effect plots
        "best_morphology_descriptor": best.metrics.get("_morphology_descriptor"),
        "best_behavior_descriptor": best.metrics.get("_behavior_descriptor"),
        "population": [
            {
                "id": i.id,
                "fitness": i.fitness,
                "limbs": i.genome.count_limbs(),
                "depth": i.genome.max_depth(),
                "metrics": clean_population(i.metrics),
            }
            for i in pop
        ],
    }