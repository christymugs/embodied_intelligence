"""Mutation operators over the morphology genome.

Each operator respects `MorphologyBounds` so mutants stay simulable. The set
covers the full open design space the project targets:

  grow_limb     : a new limb sprouts from an existing one (recursive growth)
  prune_limb    : remove a non-root limb (and its subtree)
  resize_limb   : jitter a random limb's radius/height/density
  resize_torso  : jitter the root specifically (torso stays open to change)
  perturb_joint : jitter a joint's axis/range/gear
  move_attach   : slide where a limb attaches along its parent

`mutate()` composes these stochastically. It always works on a *copy*, so the
parent genome is never altered in place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphology.genome import Genome, LimbGene, JointGene, MorphologyBounds, _u


@dataclass
class MutationRates:
    """Per-call probability of applying each operator."""
    grow_limb: float = 0.25
    prune_limb: float = 0.15
    resize_limb: float = 0.6
    resize_torso: float = 0.3
    perturb_joint: float = 0.5
    move_attach: float = 0.3
    # Relative scale of Gaussian jitter (fraction of each bound's span).
    sigma: float = 0.1


def mutate(genome: Genome, rng: np.random.Generator, rates: MutationRates | None = None) -> Genome:
    """Return a mutated copy of `genome`."""
    rates = rates or MutationRates()
    g = genome.copy()
    b = g.bounds

    if rng.random() < rates.resize_torso:
        _resize_limb(g.root, b, rng, rates.sigma, is_root=True)

    if rng.random() < rates.resize_limb:
        target = _pick(g.all_limbs()[1:], rng)  # any non-root limb
        if target is not None:
            _resize_limb(target, b, rng, rates.sigma, is_root=False)

    if rng.random() < rates.perturb_joint:
        jointed = [l for l in g.all_limbs() if l.joint is not None]
        target = _pick(jointed, rng)
        if target is not None:
            _perturb_joint(target.joint, b, rng, rates.sigma)

    if rng.random() < rates.move_attach:
        target = _pick(g.all_limbs()[1:], rng)
        if target is not None and target.attach_frac is not None:
            target.attach_frac = _clip(
                target.attach_frac + rng.normal(0, rates.sigma), b.attach_frac
            )

    if rng.random() < rates.grow_limb:
        _grow_limb(g, b, rng)

    if rng.random() < rates.prune_limb:
        _prune_limb(g, rng)

    return g


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def _resize_limb(limb: LimbGene, b: MorphologyBounds, rng, sigma, is_root: bool) -> None:
    rad_bounds = b.root_radius if is_root else b.limb_radius
    hgt_bounds = b.root_height if is_root else b.limb_height
    limb.radius = _jitter(limb.radius, rad_bounds, rng, sigma)
    limb.height = _jitter(limb.height, hgt_bounds, rng, sigma)
    limb.density = _jitter(limb.density, b.density, rng, sigma)


def _perturb_joint(joint: JointGene, b: MorphologyBounds, rng, sigma) -> None:
    # Jitter the symmetric range half-width.
    half = _jitter(joint.range_max, b.joint_range, rng, sigma)
    joint.range_min, joint.range_max = -half, half
    joint.gear = _jitter(joint.gear, b.gear, rng, sigma)
    # Occasionally flip to a different principal axis.
    if b.axis_choices is not None and rng.random() < 0.2:
        joint.axis = list(b.axis_choices[rng.integers(len(b.axis_choices))])


def _grow_limb(g: Genome, b: MorphologyBounds, rng) -> None:
    if g.count_limbs() >= b.max_limbs:
        return
    # Candidate parents: limbs whose depth leaves room for a child.
    depths = _depths(g)
    candidates = [l for l, d in depths.items() if d < b.max_depth and len(l.children) < b.max_children]
    parent = _pick(candidates, rng)
    if parent is None:
        return
    child = LimbGene(
        radius=_u(rng, b.limb_radius),
        height=_u(rng, b.limb_height),
        density=_u(rng, b.density),
        attach_frac=_u(rng, b.attach_frac),
        joint=_random_joint(b, rng),
    )
    parent.children.append(child)


def _prune_limb(g: Genome, rng) -> None:
    # Map each non-root limb to (parent, index) so we can detach it.
    detachable: list[tuple[LimbGene, int]] = []
    for parent in g.all_limbs():
        for i, _ in enumerate(parent.children):
            detachable.append((parent, i))
    if not detachable:
        return
    parent, idx = detachable[rng.integers(len(detachable))]
    del parent.children[idx]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _random_joint(b: MorphologyBounds, rng) -> JointGene:
    if b.axis_choices is not None:
        axis = list(b.axis_choices[rng.integers(len(b.axis_choices))])
    else:
        v = rng.normal(size=3)
        axis = list(v / (np.linalg.norm(v) + 1e-9))
    r = _u(rng, b.joint_range)
    return JointGene(axis=axis, range_min=-r, range_max=r, gear=_u(rng, b.gear))


def _depths(g: Genome) -> dict[LimbGene, int]:
    out: dict[LimbGene, int] = {}

    def walk(limb: LimbGene, d: int):
        out[limb] = d
        for c in limb.children:
            walk(c, d + 1)

    walk(g.root, 1)
    return out


def _pick(items, rng):
    if not items:
        return None
    return items[rng.integers(len(items))]


def _clip(x, lohi):
    lo, hi = lohi
    return float(min(max(x, lo), hi))


def _jitter(x, lohi, rng, sigma):
    lo, hi = lohi
    span = hi - lo
    return _clip(x + rng.normal(0, sigma * span), lohi)