"""Evolvable morphology genome.

The genome is a *recipe* for building a RobotTree. It is deliberately open:

  - Recursive tree    : limbs grow from limbs, arbitrary depth/branching.
  - Heterogeneous size: every limb carries its own radius/height/density.
  - Open torso        : the root limb's size is part of the genome too.
  - Free attachment   : a child attaches at `attach_frac` along its parent's
                        long axis (0 = base, 1 = tip), so limbs can sprout
                        anywhere on a parent, not just the tip.

Design choices are intentionally minimal so the search can find body plans a
human wouldn't hand-design. Hard bounds (in `MorphologyBounds`) exist only to
keep models physically simulable, not to bias the shape.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .robotic_tree import Limb, RobotTree


# --------------------------------------------------------------------------- #
# Bounds for the open design space. These are *simulability* limits, not a
# prior on good shapes. Widen them to open the space further.
# --------------------------------------------------------------------------- #
@dataclass
class MorphologyBounds:
    # Torso (root) tends to be chunkier than limbs.
    root_radius: tuple[float, float] = (0.04, 0.12)
    root_height: tuple[float, float] = (0.20, 0.60)
    # Non-root limbs.
    limb_radius: tuple[float, float] = (0.02, 0.06)
    limb_height: tuple[float, float] = (0.10, 0.50)
    density: tuple[float, float] = (400.0, 1200.0)
    # Joint range (degrees, symmetric: [-r, +r]).
    joint_range: tuple[float, float] = (20.0, 120.0)
    # Where a child attaches along its parent's axis (0=base, 1=tip).
    attach_frac: tuple[float, float] = (0.0, 1.0)
    gear: tuple[float, float] = (0.5, 2.0)

    # Tree-shape limits keep evaluation tractable. Loosen to open further.
    max_limbs: int = 12
    max_depth: int = 4
    max_children: int = 3

    # Principal axes only (interpretable). Set to None to sample any unit vector.
    axis_choices = ([1, 0, 0], [0, 1, 0], [0, 0, 1])


@dataclass
class JointGene:
    axis: list[float]
    range_min: float
    range_max: float
    actuator_type: str = "position"
    gear: float = 1.0


@dataclass(eq=False)
class LimbGene:
    radius: float
    height: float
    density: float
    # None for the root limb; otherwise where on the parent this attaches.
    attach_frac: float | None = None
    # None => rigid weld to parent; otherwise an actuated joint.
    joint: JointGene | None = None
    children: list["LimbGene"] = field(default_factory=list)


class Genome:
    """A whole morphology: one root LimbGene plus helpers to build/mutate it."""

    def __init__(self, root: LimbGene, bounds: MorphologyBounds | None = None):
        self.root = root
        self.bounds = bounds or MorphologyBounds()

    # ----------------------- construction ----------------------- #
    @classmethod
    def random(
        cls, bounds: MorphologyBounds | None = None, rng: np.random.Generator | None = None
    ) -> "Genome":
        bounds = bounds or MorphologyBounds()
        rng = rng or np.random.default_rng()

        root = LimbGene(
            radius=_u(rng, bounds.root_radius),
            height=_u(rng, bounds.root_height),
            density=_u(rng, bounds.density),
            attach_frac=None,
            joint=None,
        )
        g = cls(root, bounds)
        # Grow a random subtree under the torso.
        budget = [bounds.max_limbs - 1]  # remaining non-root limbs
        g._grow(root, depth=1, rng=rng, budget=budget)
        # Guarantee the body is trainable: a torso with no actuated limb can't
        # locomote or learn anything, so force one rather than waste an eval on it.
        if g.num_actuators() == 0:
            root.children.append(
                LimbGene(
                    radius=_u(rng, bounds.limb_radius),
                    height=_u(rng, bounds.limb_height),
                    density=_u(rng, bounds.density),
                    attach_frac=_u(rng, bounds.attach_frac),
                    joint=g._random_joint(rng),
                )
            )
        return g

    def _grow(self, parent: LimbGene, depth: int, rng, budget: list[int]) -> None:
        if depth > self.bounds.max_depth or budget[0] <= 0:
            return
        # Branching shrinks with depth so trees don't always max out.
        max_kids = min(self.bounds.max_children, budget[0])
        n_children = rng.integers(0, max_kids + 1) if max_kids > 0 else 0
        for _ in range(int(n_children)):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            child = LimbGene(
                radius=_u(rng, self.bounds.limb_radius),
                height=_u(rng, self.bounds.limb_height),
                density=_u(rng, self.bounds.density),
                attach_frac=_u(rng, self.bounds.attach_frac),
                joint=self._random_joint(rng),
            )
            parent.children.append(child)
            self._grow(child, depth + 1, rng, budget)

    def _random_joint(self, rng) -> JointGene:
        if self.bounds.axis_choices is not None:
            axis = list(self.bounds.axis_choices[rng.integers(len(self.bounds.axis_choices))])
        else:
            v = rng.normal(size=3)
            axis = list(v / (np.linalg.norm(v) + 1e-9))
        r = _u(rng, self.bounds.joint_range)
        return JointGene(
            axis=axis,
            range_min=-r,
            range_max=r,
            actuator_type="position",
            gear=_u(rng, self.bounds.gear),
        )

    # ----------------------- phenotype ----------------------- #
    def to_robot_tree(self) -> RobotTree:
        """Compile the genome into a RobotTree ready for MujocoBuilder."""
        root_limb = Limb(
            parent_body=None, name=None,
            radius=self.root.radius, height=self.root.height, density=self.root.density,
        )
        tree = RobotTree(root_limb)
        for child_gene in self.root.children:
            self._attach(tree, root_limb, child_gene)
        return tree

    def _attach(self, tree: RobotTree, parent_limb: Limb, gene: LimbGene) -> None:
        z = (gene.attach_frac or 0.0) * parent_limb.height
        site = parent_limb.add_site(pos=(0, 0, z))
        child_limb, conn = tree.add_limb(
            parent_limb, site, gene.radius, gene.height, gene.density
        )
        if gene.joint is not None:
            j = tree.add_joint_to_connection(
                conn, "hinge", gene.joint.axis, gene.joint.range_min, gene.joint.range_max
            )
            tree.add_actuator_to_connection(
                conn, j, actuator_type=gene.joint.actuator_type, gear=gene.joint.gear
            )
        for c in gene.children:
            self._attach(tree, child_limb, c)

    # ----------------------- introspection ----------------------- #
    def all_limbs(self) -> list[LimbGene]:
        """Flat list of every LimbGene (root first), for mutation targeting."""
        out: list[LimbGene] = []

        def walk(g: LimbGene):
            out.append(g)
            for c in g.children:
                walk(c)

        walk(self.root)
        return out

    def count_limbs(self) -> int:
        return len(self.all_limbs())

    def max_depth(self) -> int:
        def depth(g: LimbGene) -> int:
            return 1 + max((depth(c) for c in g.children), default=0)

        return depth(self.root)

    def num_actuators(self) -> int:
        return sum(1 for g in self.all_limbs() if g.joint is not None)

    def copy(self) -> "Genome":
        return Genome(copy.deepcopy(self.root), self.bounds)


def _u(rng: np.random.Generator, lohi: tuple[float, float]) -> float:
    lo, hi = lohi
    return float(rng.uniform(lo, hi))