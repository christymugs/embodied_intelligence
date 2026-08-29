"""Evolvable morphology genome.

The genome is a *recipe* for building a RobotTree. It is deliberately open:

  - Recursive tree    : limbs grow from limbs, arbitrary depth/branching.
  - Heterogeneous size: every limb carries its own radius/height/density.
  - Open torso        : the root limb's size is part of the genome too.
  - Free attachment   : a child attaches at `attach_frac` along its parent's
                        long axis (0 = base, 1 = tip) and `attach_azimuth`
                        around its circumference, so limbs can sprout anywhere
                        on the parent's SURFACE, not just the tip and not
                        embedded on its centerline -- a hinge bracket bolted to
                        the side, not a limb growing from inside the body.

Design choices are intentionally minimal so the search can find body plans a
human wouldn't hand-design. Hard bounds (in `MorphologyBounds`) exist only to
keep models physically simulable, not to bias the shape.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .robotic_tree import Limb, RobotTree


# --------------------------------------------------------------------------- #
# Actuator realism: cap peak actuator force from the geometry of the limb it
# drives, so a bigger/heavier limb needs a bigger motor to match -- instead of
# the previous flat, unbounded kp, where a hinge on a tiny limb could apply
# the same force as one on the torso for free. The evolved `gear` (mechanical
# advantage, see MorphologyBounds.gear) then scales the *delivered* torque
# within this cap, rather than scaling an unbounded one.
# TORQUE_DENSITY_NM_PER_KG is an order-of-magnitude stand-in for small
# hobby/robotic servos (Nm of stall torque per kg of actuator mass) -- a
# tunable realism knob, not a measured constant.
# --------------------------------------------------------------------------- #
TORQUE_DENSITY_NM_PER_KG = 5.0

# Small explicit gap added on top of true tangency in _attach's surface offset,
# so a joint at rest has real daylight from its parent instead of sitting
# exactly on the contact boundary (where normal jitter constantly registers as
# "touching" even though it isn't doing anything -- confirmed on run13: 97% of
# steps showed contact but with near-zero correlation to displacement, i.e.
# not exploited, just noisy). A few mm is a reasonable clearance for a real
# hinge bracket at this size scale, not a measured constant.
ATTACHMENT_CLEARANCE_M = 0.004


def _capsule_mass(radius: float, height: float, density: float) -> float:
    """Mass of the capsule MuJoCo will build for this limb (cylinder + two
    hemispherical caps), matching the geometry emitted by MujocoBuilder."""
    volume = math.pi * radius ** 2 * height + (4.0 / 3.0) * math.pi * radius ** 3
    return density * volume


def _subtree_mass(gene: "LimbGene") -> float:
    """Total mass this limb's own joint has to actually hold up: itself PLUS
    everything hanging further down the chain from it. A joint sized only
    from its immediate limb's own mass has no idea it may also need to
    support a heavy sub-limb further out -- that starves proximal joints of
    the torque they need to hold the body up at all, leaving gravity-driven
    toppling (lean forward, sink, bank the fall as forward reward) as the
    only strategy that produces any reward, since actively supporting the
    chain is physically impossible for the motor it was given."""
    m = _capsule_mass(gene.radius, gene.height, gene.density)
    for c in gene.children:
        m += _subtree_mass(c)
    return m


def _actuator_force_cap(subtree_mass: float) -> float:
    """Max actuator force (pre-gear) for a joint carrying this much mass."""
    return TORQUE_DENSITY_NM_PER_KG * subtree_mass


# Reference angular speed used only to calibrate joint damping below -- kept
# numerically in sync with TaskConfig.max_joint_speed (learning/env.py) by
# convention, not by import (genome.py must not depend on learning.env, which
# itself imports from morphology.genome -- that would be circular).
REFERENCE_JOINT_SPEED_RAD_S = 8.0

# Joint damping used to be a flat 5.0 Nm/(rad/s) for every joint regardless of
# its own actuator's torque budget -- a constant sized (if ever deliberately)
# for the project's original unbounded kp=100 actuators. Every realism fix
# since (mass-based force capping, subtree-mass accounting, torque-speed
# limiting) made actuators progressively weaker without ever revisiting this,
# so damping silently became a dominant drag: measured directly on
# experiments/run20's champion, one joint had a 9.53 Nm peak torque against a
# 10-20 Nm damping resistance at just 2-4 rad/s -- damping alone exceeding the
# ENTIRE torque budget of the motor meant to overcome it. Smaller, more
# "buildable" bodies (lower torque budgets from complexity pressure) were hit
# hardest, and no amount of extra training could fix a physical force the
# motor can't out-torque. Damping now scales with each joint's OWN peak
# torque instead: calibrated so damping resistance at a reference angular
# speed is a fixed FRACTION of that joint's peak torque -- proportional drag,
# not an absolute one that dwarfs weak actuators. First attempt used 0.15
# (matching experiments/damping_fix_policy.zip) but that measurably made
# things WORSE, not better -- damping isn't purely a drag, some of it
# provides passive stability against small wobbles growing into falls, and
# 0.15 likely went far enough the other way to under-damp the joints. Raised
# to 0.5 as a more moderate middle ground: still a fraction of the old flat
# 5.0's effective drag on weak actuators, without cutting it so far that
# passive stability is lost.
DAMPING_TORQUE_FRACTION = 0.5


def _joint_damping(peak_torque: float) -> float:
    """Damping coefficient so damping torque at REFERENCE_JOINT_SPEED_RAD_S
    is DAMPING_TORQUE_FRACTION of this joint's own peak actuator torque."""
    return DAMPING_TORQUE_FRACTION * peak_torque / REFERENCE_JOINT_SPEED_RAD_S


# --------------------------------------------------------------------------- #
# Bounds for the open design space. These are *simulability* limits, not a
# prior on good shapes. Widen them to open the space further.
# --------------------------------------------------------------------------- #
@dataclass
class MorphologyBounds:
    # Torso (root) tends to be chunkier than limbs.
    root_radius: tuple[float, float] = (0.04, 0.12)
    root_height: tuple[float, float] = (0.20, 0.60)
    # Non-root limbs. Absolute simulability clamps -- actual sampling is
    # tapered from the parent's own size (taper_radius_frac/taper_height_frac
    # above); these just bound how far that tapering can drift.
    limb_radius: tuple[float, float] = (0.02, 0.06)
    limb_height: tuple[float, float] = (0.10, 0.50)
    density: tuple[float, float] = (400.0, 1200.0)
    # Joint range (degrees, symmetric: [-r, +r]).
    joint_range: tuple[float, float] = (20.0, 120.0)
    # Where a child attaches along its parent's axis (0=base, 1=tip). Full
    # range stays legal (a body can still evolve mid-shaft attachment), but
    # _sample_attach_frac below biases sampling toward the ends -- a real
    # jointed chain (hip->knee->ankle) attaches at segment ends, not
    # partway up a shaft, and independent-uniform sampling was producing
    # limbs sprouting out of another limb's middle by pure chance on every
    # attachment.
    attach_frac: tuple[float, float] = (0.0, 1.0)
    # A child's radius/height is drawn as a FRACTION OF ITS PARENT's own
    # radius/height (tapering, like a real jointed limb: thigh -> shin ->
    # foot each comparable to, usually a bit smaller than, what it's
    # attached to) -- not independently from limb_radius/limb_height below.
    # Independent sampling let a leaf limb two levels deep roll bigger than
    # the torso it ultimately hangs off, or a torso end up with legs a
    # fraction of its own thickness, by pure chance on every limb (see
    # experiments/run25's champion and the "huge torso, tiny legs, then a
    # giant leg" bodies this produced). Still clipped to limb_radius/
    # limb_height as absolute simulability rails.
    taper_radius_frac: tuple[float, float] = (0.45, 0.95)
    taper_height_frac: tuple[float, float] = (0.55, 1.05)
    # Mounting angle around the parent's circumference: a small set of
    # standard positions (like a real bracket's bolt-hole pattern), not an
    # arbitrary continuous angle. A body can still evolve any number of
    # limbs, any sizes, any tree shape, fully asymmetric -- but whichever
    # angles it picks are always one of these 8 (45 degree steps), so a real
    # build only ever needs a handful of repeated brackets, never one
    # custom-machined angle per joint. Set to None to allow any continuous
    # angle (fully open, but each joint may need its own custom bracket).
    azimuth_choices: tuple[float, ...] | None = tuple(
        i * math.pi / 4 for i in range(8)
    )
    gear: tuple[float, float] = (0.5, 2.0)

    # Tree-shape limits keep evaluation tractable. Loosen to open further.
    # Was raised to 12, then dropped to 8 after run19 showed a lot of
    # unearned dead-weight limbs -- but a hand-built symmetric quadruped
    # (4 legs x thigh+shin = 8 non-root limbs) sits EXACTLY at 8, leaving no
    # headroom for anything beyond that minimal reference shape. Back to 12:
    # complexity_weight (raised alongside this) is now the primary pressure
    # against dead weight, so the cap doesn't also need to do that job --
    # it's just enough room for a mirrored-pair body plan plus something
    # extra, without dictating what that extra thing is.
    max_limbs: int = 12
    # Was 4: let evolution chain limbs off limbs off limbs off limbs, which
    # produced spindly structures where a limb three levels down the chain
    # ends up longer than the torso it's ultimately hanging off -- no visible
    # "trunk" to read as a buildable body, just a tangle of comparably-sized
    # rods (see experiments/run15 gen4 champion). Capped at 2 (torso -> limb
    # -> sub-limb, e.g. an upper leg + lower leg/foot segment) so every body
    # keeps a legible torso-and-limbs shape, the kind you could actually
    # bracket-mount and build.
    max_depth: int = 2
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
    attach_azimuth: float | None = None
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
        # Sample how many non-root limbs THIS body gets, uniformly over the legal
        # range, then grow toward that target. Letting per-node branching decide
        # organically (the old approach) makes tree size a Galton-Watson process
        # under a hard cap: those are bimodal -- growth either dies out almost
        # immediately or runs away to the cap, badly under-sampling every value
        # in between. Sampling the target first keeps every limb count reachable
        # with roughly equal probability, per individual, with nothing hardcoded.
        target_non_root = int(rng.integers(1, bounds.max_limbs))  # 1 .. max_limbs-1
        g._grow_to_target(root, target_non_root, rng)
        if g.num_actuators() == 0:
            radius, height = sample_tapered_size(rng, root.radius, root.height, bounds)
            root.children.append(
                LimbGene(
                    radius=radius,
                    height=height,
                    density=_u(rng, bounds.density),
                    attach_frac=sample_attach_frac(rng, bounds),
                    attach_azimuth=sample_azimuth_avoiding(
                        rng, bounds, [c.attach_azimuth for c in root.children]
                    ),
                    joint=g._random_joint(rng),
                )
            )
        return g

    def _grow_to_target(self, root: LimbGene, target: int, rng) -> None:
        """Attach `target` limbs one at a time, each to a uniformly random
        eligible existing limb (any limb under max_depth/max_children). Growth
        order is randomized across the whole tree rather than depth-first, so
        no single node's branching roll can determine the final size."""
        depth = {id(root): 0}
        pool = [root]
        added = 0
        while added < target:
            eligible = [l for l in pool if depth[id(l)] < self.bounds.max_depth
                        and len(l.children) < self.bounds.max_children]
            if not eligible:
                break  # tree-shape limits reached before hitting the target
            parent = eligible[rng.integers(len(eligible))]
            radius, height = sample_tapered_size(rng, parent.radius, parent.height, self.bounds)
            child = LimbGene(
                radius=radius,
                height=height,
                density=_u(rng, self.bounds.density),
                attach_frac=sample_attach_frac(rng, self.bounds),
                attach_azimuth=sample_azimuth_avoiding(
                    rng, self.bounds, [c.attach_azimuth for c in parent.children]
                ),
                joint=self._random_joint(rng),
            )
            parent.children.append(child)
            depth[id(child)] = depth[id(parent)] + 1
            pool.append(child)
            added += 1

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
        theta = gene.attach_azimuth or 0.0
        # Site sits just clear of the parent's surface -- offset by parent
        # radius PLUS the child's own radius, not parent radius alone. The
        # child is itself a capsule with a rounded end-cap of radius
        # gene.radius; anchoring that cap's CENTER exactly on the parent's
        # surface (the old, buggy offset) leaves the cap's own volume
        # overlapping the parent by up to gene.radius, guaranteed by
        # construction on every attachment, not something evolution found.
        # This offset is the tangent case: cap surface just touches parent
        # surface, like a ball resting against a cylinder, zero overlap.
        # ATTACHMENT_CLEARANCE_M adds a bit of real daylight on top of that,
        # so a resting joint doesn't sit exactly on the contact boundary.
        offset = parent_limb.radius + gene.radius + ATTACHMENT_CLEARANCE_M
        x = offset * math.cos(theta)
        y = offset * math.sin(theta)
        # Base orientation: rotate the child's local +z (the axis its own
        # capsule grows along) to point radially outward from that surface
        # point -- a hinge bracket mounted on the side, not a limb sprouting
        # parallel to the parent's own long axis. The target direction
        # (cos theta, sin theta, 0) is always exactly 90 deg from +z, about
        # axis (-sin theta, cos theta, 0) -- a fixed closed-form quaternion,
        # no rotation library needed. The joint's own evolved range then
        # rotates the limb away from this resting, surface-normal pose.
        half = math.pi / 4  # cos/sin of half the 90 deg rotation angle
        quat = (
            math.cos(half),
            -math.sin(half) * math.sin(theta),
            math.sin(half) * math.cos(theta),
            0.0,
        )
        site = parent_limb.add_site(pos=(x, y, z), quat=quat)
        child_limb, conn = tree.add_limb(
            parent_limb, site, gene.radius, gene.height, gene.density
        )
        if gene.joint is not None:
            j = tree.add_joint_to_connection(
                conn, "hinge", gene.joint.axis, gene.joint.range_min, gene.joint.range_max
            )
            force_cap = _actuator_force_cap(_subtree_mass(gene))
            tree.add_actuator_to_connection(
                conn, j, actuator_type=gene.joint.actuator_type, gear=gene.joint.gear,
                forcerange=(-force_cap, force_cap),
            )
            j.damping = _joint_damping(force_cap * gene.joint.gear)
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

    def max_reach(self) -> float:
        """Longest root-to-leaf sum of limb heights, in meters.

        A physically-motivated "how tall could this body stand if its
        longest kinematic chain were extended in a straight line" estimate
        -- used as the absolute standing-height reference in
        learning/env.py's forward-reward shaping. root.height alone (the
        trunk capsule's own length) badly undersells this for a body whose
        actual legs are chained sub-limbs off a short trunk, letting a
        collapsed sprawl already sit at ~100% of "standing" by that measure
        alone (see run25's champion, root.height <= 0.3 -- confirmed on a
        controlled A/B retrain that reusing root.height as the reference
        wasn't a tall-enough bar to change anything)."""
        def reach(g: LimbGene) -> float:
            return g.height + max((reach(c) for c in g.children), default=0.0)

        return reach(self.root)

    def num_actuators(self) -> int:
        return sum(1 for g in self.all_limbs() if g.joint is not None)

    def copy(self) -> "Genome":
        return Genome(copy.deepcopy(self.root), self.bounds)


def _u(rng: np.random.Generator, lohi: tuple[float, float]) -> float:
    lo, hi = lohi
    return float(rng.uniform(lo, hi))


def _sample_azimuth(rng: np.random.Generator, bounds: "MorphologyBounds") -> float:
    if bounds.azimuth_choices is None:
        return _u(rng, (0.0, 2 * math.pi))
    choices = bounds.azimuth_choices
    return float(choices[rng.integers(len(choices))])


def _angular_dist(a: float, b: float) -> float:
    """Shortest distance between two angles, in [0, pi]."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def sample_azimuth_avoiding(
    rng: np.random.Generator, bounds: "MorphologyBounds", taken: list[float],
) -> float:
    """Like _sample_azimuth, but avoids azimuths already used by sibling
    limbs on the same parent where possible -- a real bracket only has one
    bolt per hole, so multiple children landing on the same or an adjacent
    mounting point (previously possible on every attachment, independent
    random choice) produced overlapping/clumped limbs rather than a spread
    stance. Falls back to allowing a repeat only when every option is
    already taken (e.g. more children than azimuth_choices has slots)."""
    if bounds.azimuth_choices is None:
        min_sep = math.pi / 6  # 30 degrees
        candidate = _u(rng, (0.0, 2 * math.pi))
        for _ in range(8):
            if all(_angular_dist(candidate, t) >= min_sep for t in taken):
                return candidate
            candidate = _u(rng, (0.0, 2 * math.pi))
        return candidate  # give up after 8 tries, accept whatever we have
    choices = bounds.azimuth_choices
    free = [c for c in choices if not any(_angular_dist(c, t) < 1e-6 for t in taken)]
    pool = free or list(choices)
    return float(pool[rng.integers(len(pool))])


def sample_attach_frac(rng: np.random.Generator, bounds: "MorphologyBounds") -> float:
    """Where along the parent's axis a child attaches, biased toward the
    ends (0=base, 1=tip) rather than uniform -- a real jointed chain
    attaches at segment ends (hip->knee->ankle), not partway up a shaft.
    Beta(0.4, 0.4) is U-shaped: still covers the full range (mid-shaft
    attachment stays legal, just less likely), rather than hard-restricting
    to exactly the two ends."""
    lo, hi = bounds.attach_frac
    return float(lo + rng.beta(0.4, 0.4) * (hi - lo))


def sample_tapered_size(
    rng: np.random.Generator, parent_radius: float, parent_height: float,
    bounds: "MorphologyBounds",
) -> tuple[float, float]:
    """A child limb's (radius, height), drawn as a fraction of its PARENT's
    own size rather than independently from limb_radius/limb_height -- see
    MorphologyBounds.taper_radius_frac/taper_height_frac. Clipped to
    limb_radius/limb_height as absolute simulability rails, so tapering
    from an extreme parent can't drift outside what the rest of the system
    (actuator force caps, etc.) was tuned for."""
    radius = _clip(parent_radius * _u(rng, bounds.taper_radius_frac), bounds.limb_radius)
    height = _clip(parent_height * _u(rng, bounds.taper_height_frac), bounds.limb_height)
    return radius, height


def _clip(x: float, lohi: tuple[float, float]) -> float:
    lo, hi = lohi
    return float(min(max(x, lo), hi))