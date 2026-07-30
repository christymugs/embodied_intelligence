"""Fixed-length structural summary of a genome, for novelty comparisons.

Two genomes can have completely different tree shapes (different limb counts,
depths, branching), so they can't be compared limb-by-limb. This reduces any
genome to the same fixed-length vector of aggregate body-plan statistics —
sizes, joint layout, branching — so morphology distance is just a Euclidean
distance between two arrays, regardless of tree structure. Every value is
normalized to roughly [0,1] against the genome's own MorphologyBounds so no
one dimension (e.g. density in the hundreds) dominates the distance purely by
scale.
"""

from __future__ import annotations

import numpy as np

from .genome import Genome


def _norm(x: float, lohi: tuple[float, float]) -> float:
    lo, hi = lohi
    return (x - lo) / max(hi - lo, 1e-9)


def morphology_descriptor(genome: Genome) -> np.ndarray:
    """Return a fixed-length (14,) descriptor vector for `genome`."""
    b = genome.bounds
    limbs = genome.all_limbs()
    root, non_root = limbs[0], limbs[1:]

    limb_count_norm = genome.count_limbs() / b.max_limbs
    depth_norm = genome.max_depth() / b.max_depth
    actuator_norm = genome.num_actuators() / b.max_limbs

    radii = [_norm(root.radius, b.root_radius)] + [_norm(l.radius, b.limb_radius) for l in non_root]
    heights = [_norm(root.height, b.root_height)] + [_norm(l.height, b.limb_height) for l in non_root]
    densities = [_norm(l.density, b.density) for l in limbs]

    radius_mean, radius_std = float(np.mean(radii)), float(np.std(radii))
    height_mean, height_std = float(np.mean(heights)), float(np.std(heights))
    density_mean, density_std = float(np.mean(densities)), float(np.std(densities))

    attach_mean = float(np.mean([l.attach_frac for l in non_root])) if non_root else 0.0

    jointed = [l for l in limbs if l.joint is not None]
    if jointed and b.axis_choices is not None:
        axes = np.array([l.joint.axis for l in jointed])
        axis_fracs = [
            float(np.mean(np.all(np.isclose(axes, choice), axis=1)))
            for choice in b.axis_choices
        ]
    else:
        axis_fracs = [0.0, 0.0, 0.0]

    branching_norm = float(np.mean([len(l.children) for l in limbs])) / max(b.max_children, 1)

    return np.array([
        limb_count_norm, depth_norm, actuator_norm,
        radius_mean, radius_std, height_mean, height_std, density_mean, density_std,
        attach_mean, branching_norm, *axis_fracs,
    ], dtype=np.float64)
