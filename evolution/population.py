"""Small helpers for keeping track of the evolving population.

An Individual is one robot body in the population. It stores the body's genome,
its id, its fitness score, whether it has already been evaluated, and any extra
metrics we want to save.

This file also has the selection helpers evolution uses to decide which bodies
get copied or mutated into the next generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morphology.genome import Genome


@dataclass
class Individual:
    genome: Genome
    id: int
    fitness: float = float("-inf")
    evaluated: bool = False
    metrics: dict = field(default_factory=dict)


def tournament_select(pop: list[Individual], k: int, rng: np.random.Generator) -> Individual:
    """Pick the fittest of k random contenders."""
    idx = rng.integers(0, len(pop), size=k)
    contenders = [pop[i] for i in idx]
    return max(contenders, key=lambda ind: ind.fitness)


def truncation_select(pop: list[Individual], frac: float) -> list[Individual]:
    """Return the top `frac` of the population by fitness."""
    n = max(1, int(len(pop) * frac))
    return sorted(pop, key=lambda ind: ind.fitness, reverse=True)[:n]