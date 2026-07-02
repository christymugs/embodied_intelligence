"""Parallel evaluation of individuals across processes.

Each body trains on its own, so there is no reason to do them one at a time.
This file sends different bodies to different worker processes so multiple
robots can train at once.

A couple important details:

1. Each worker is forced to use one torch thread. The speedup comes from
   training many bodies at the same time, not from letting one body grab the
   whole CPU. If every worker tries to use all cores, they fight each other and
   everything gets slower.

2. The worker function has to stay at the top level of the file. Python needs
   to import it cleanly when it starts child processes, so no nested
   functions, lambdas, or weird closure stuff here.

Workers return simple Python dicts with the fitness, metrics, and learning
curve. They do not return the trained SB3 model directly because that object is
too big and messy to pass between processes. When we want to replay a trained
robot later, the worker saves the policy to a .zip file and returns the path.
"""
from __future__ import annotations

import dataclasses
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from morphology.genome import Genome
from learning.env import TaskConfig
from learning.train import train_morphology, TrainConfig
from fitness.learnability import compute_learnability


def worker_init() -> None:
    """Run once per spawned worker: silence warnings, pin to a single thread."""
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def evaluate_one(genome: Genome, task: TaskConfig, train_cfg: TrainConfig,
                 eval_seeds: int, metric: str, base_seed: int,
                 policy_path: str | None = None) -> dict:
    """Train + score a single morphology. Returns a plain, picklable result dict.

    If `policy_path` is given, the best-scoring seed's trained policy is saved
    there (a .zip) so it can be replayed later. The model itself is never
    returned across the process boundary — only written to disk.
    """
    if genome.num_actuators() == 0:
        return {"fitness": float("-inf"), "metrics": {"invalid": "no_actuators"},
                "curve": None, "policy_path": None}

    scores, results = [], []
    best_model, best_score = None, float("-inf")
    for s in range(eval_seeds):
        cfg = dataclasses.replace(train_cfg, seed=base_seed + s)
        try:
            out = train_morphology(genome, task, cfg)
            lr = compute_learnability(out["timesteps"], out["returns"])
            sc = lr.score(metric)
            scores.append(sc)
            results.append(lr)
            if policy_path is not None and sc > best_score:
                best_score, best_model = sc, out["model"]
        except Exception as e:  # an unstable body must not kill the worker
            scores.append(float("-inf"))
            results.append(None)

    finite = [x for x in scores if np.isfinite(x)]
    fitness = float(np.mean(finite)) if finite else float("-inf")

    saved = None
    if policy_path is not None and best_model is not None:
        try:
            best_model.save(policy_path)
            saved = policy_path
        except Exception:
            saved = None

    valid = [r for r in results if r is not None]
    metrics, curve = {}, None
    if valid:
        metrics = {
            "auc": float(np.mean([r.auc for r in valid])),
            "final": float(np.mean([r.final for r in valid])),
            "steps_to_threshold": float(np.mean([r.steps_to_threshold for r in valid])),
            "reached_threshold": float(np.mean([r.reached_threshold for r in valid])),
        }
        best = max(valid, key=lambda r: r.score(metric))
        curve = {"t": list(best.curve_t), "r": list(best.curve_r)}

    return {"fitness": fitness, "metrics": metrics, "curve": curve, "policy_path": saved}


def _eval_star(args):
    """Top-level unpacking wrapper so ProcessPoolExecutor can pickle the target."""
    return evaluate_one(*args)


def resolve_workers(n_workers: int) -> int:
    """0 (or negative) => auto = all CPUs; otherwise the requested count."""
    if n_workers and n_workers > 0:
        return n_workers
    return max(1, os.cpu_count() or 1)


def evaluate_jobs(jobs: list[tuple], executor: ProcessPoolExecutor | None) -> dict:
    """Run all the evaluation jobs and return the results by id.

    Each job contains one body plus everything needed to train and score it:
    the genome, task settings, training config, metric, seed, and optional policy
    save path.

    If policy_path is None, we just train/score the body and move on.
    If policy_path is a .zip path, we save the trained policy there so we can replay
    that robot later in the visualizer.

    If there is no executor, the jobs run one at a time. If there is an executor,
    the jobs run in parallel. Results are stored by key/id, so it does not matter
    which worker finishes first.
    """
    keys = [j[0] for j in jobs]
    payloads = [j[1:] for j in jobs]

    if executor is None:
        return {k: evaluate_one(*p) for k, p in zip(keys, payloads)}

    out = {}
    for k, res in zip(keys, executor.map(_eval_star, payloads)):
        out[k] = res
    return out