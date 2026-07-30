"""Novelty scoring: how different is an individual from what's already been
tried, in morphology-space and in behavior-space.

Standard novelty search (Lehman & Stanley): novelty(i) = mean distance to the
k nearest neighbors, drawn from an archive of past individuals so novelty is
judged against history, not just whoever's alive right now (otherwise a whole
population can drift together and each member still looks "novel" against a
population that moved with it). The neighbor pool used here is archive ∪ the
rest of the current generation, which is the standard formulation and avoids
that collapse mode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoveltyArchive:
    max_size: int = 500
    _morph: deque = field(init=False, repr=False)
    _behavior: deque = field(init=False, repr=False)

    def __post_init__(self):
        self._morph = deque(maxlen=self.max_size)
        self._behavior = deque(maxlen=self.max_size)

    def add(self, morph_vec: np.ndarray, behavior_vec: np.ndarray | None) -> None:
        self._morph.append(morph_vec)
        if behavior_vec is not None:
            self._behavior.append(behavior_vec)

    def morph_array(self) -> np.ndarray:
        return np.array(self._morph) if self._morph else np.empty((0,))

    def behavior_array(self) -> np.ndarray:
        return np.array(self._behavior) if self._behavior else np.empty((0,))


def _knn_mean_dist(query: np.ndarray, pool: np.ndarray, k: int) -> float:
    """Mean Euclidean distance from `query` to its k nearest rows in `pool`."""
    if pool.size == 0 or pool.shape[0] == 0:
        return 0.0
    dists = np.linalg.norm(pool - query, axis=1)
    k = min(k, dists.shape[0])
    nearest = np.partition(dists, k - 1)[:k]
    return float(np.mean(nearest))


def compute_novelty(
    pop_morph: list[np.ndarray],
    pop_behavior: list[list[float] | None],
    archive: NoveltyArchive,
    k: int = 5,
    morphology_weight: float = 0.5,
) -> list[float]:
    """One novelty score per individual, same order as pop_morph/pop_behavior.

    Morphology pool = archive's morphology vectors + every other individual's
    (everyone has one). Behavior pool = archive's behavior vectors + every
    other individual's non-None one. An individual with no behavior vector
    (invalid genome / training crashed) gets behavior_novelty = 0.0.
    """
    n = len(pop_morph)
    morph_stack = np.array(pop_morph)
    archive_morph = archive.morph_array()
    archive_behavior = archive.behavior_array()

    scores = []
    for i in range(n):
        others_morph = np.delete(morph_stack, i, axis=0)
        morph_pool = np.concatenate([archive_morph, others_morph]) if archive_morph.size else others_morph
        morph_novelty = _knn_mean_dist(pop_morph[i], morph_pool, k)

        if pop_behavior[i] is None:
            behavior_novelty = 0.0
        else:
            others_behavior = np.array([b for j, b in enumerate(pop_behavior) if j != i and b is not None])
            behavior_pool = (
                np.concatenate([archive_behavior, others_behavior])
                if archive_behavior.size and others_behavior.size
                else (archive_behavior if archive_behavior.size else others_behavior)
            )
            behavior_novelty = _knn_mean_dist(np.array(pop_behavior[i]), behavior_pool, k)

        scores.append(morphology_weight * morph_novelty + (1 - morphology_weight) * behavior_novelty)

    return scores


def percentile_ranks(values: list[float]) -> list[float]:
    """Rank each value into [0,1] (0 = smallest). len<=1 -> [0.5, ...]."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = np.argsort(np.argsort(values))
    return (order / (n - 1)).tolist()
