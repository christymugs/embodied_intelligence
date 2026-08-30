"""Gymnasium locomotion environment over an arbitrary evolved morphology.

A morphology is compiled to MuJoCo on construction. Observation and action
spaces are sized from the compiled model, so every body — whatever its limb
count — gets its own correctly-shaped policy trained from scratch.

The task is a fair, standardized "run as far as you can" locomotion task:
  - Every body starts at the same place, under the same fixed time cap (horizon) —
    so scores are comparable across morphologies. There is no finish line: reward
    is just distance covered in +x, continuously, over that fixed window.
  - Because every body gets the exact same amount of time, "further" and "faster"
    are the same thing — speed pressure falls out naturally from the fixed-time,
    unbounded-distance reward, with no arbitrary target distance to tune and no
    bonus-shaping needed for it.
  - On reset the body settles under gravity before reward counts; the episode ends
    early if the root collapses (terminate_on_fall), so faceplant-and-drift can't
    win. An untrained policy scores ~0, so scoring requires learning to move.
  - A continuous height-maintenance term (height_reward_weight) penalizes drifting
    below the body's own settled height, well before the hard fall cutoff -- this
    gives PPO a learnable gradient against "controlled falling" (banking distance
    reward from a one-time topple) instead of only a flat penalty at the moment of
    termination. It's gait-agnostic: it rewards staying near YOUR OWN resting
    height, whatever that is, so a genuinely low crawl is not penalized for being
    low, only for progressively sinking during locomotion.
  - Observations include a per-body floor-contact vector (which limbs are touching
    the ground right now), since the base qpos/qvel alone don't expose that and
    real legged/crawling controllers generally get a contact signal to work with.
Direction is a single axis (+x); locomotion style is left open (crawl/shuffle/roll).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from morphology.genome import Genome
from morphology.mujoco_builder import MujocoBuilder


@dataclass
class TaskConfig:
    horizon: int = 2500          # fixed time cap every body gets, in env steps
    frame_skip: int = 5          # sim substeps per env step
    forward_reward_weight: float = 1.0
    # Penalizes MEAN squared action across actuators (not sum), so this stays
    # actuator-COUNT-independent -- it targets how hard each joint is working,
    # not how many joints exist (that's what complexity_weight in evolution.py
    # is for; conflating the two would double-penalize bigger bodies for the
    # same thing). Bounded in [0, ctrl_cost_weight] per step since actions are
    # in [-1, 1]. Was 0.001 against a SUM over actuators, which for a ~7-10
    # actuator body was at most ~0.01 -- negligible next to a typical per-step
    # forward reward of ~0.1-1.0, so it wasn't actually discouraging flailing.
    ctrl_cost_weight: float = 0.1
    # ctrl_cost only penalizes action MAGNITUDE, never how fast an action
    # changes between consecutive steps -- so rapidly jittering a joint's
    # target back and forth costs exactly the same as moving it smoothly.
    # That let a champion (experiments/run22, gen9) win by vibrating in
    # place to creep forward via stick-slip friction rather than a
    # coordinated swinging gait. Tried weight=0.1 (experiments/run23): on a
    # controlled A/B retrain of the SAME body/budget that broke through to
    # full-horizon survival under fall_penalty_frac alone (return 283,
    # 4/5 seeds), adding this on top left it stuck at exactly 0.0 return for
    # the entire 150k-step budget -- solving "don't fall" and "don't jitter"
    # at once was too hard within budget, not just slower. DISABLED by
    # default until tried at a much smaller weight (e.g. 0.01) as a gentler
    # nudge rather than a hard constraint.
    action_rate_weight: float = 0.0
    # No finish line: same start, same fixed horizon for every body (fair +
    # standardized), reward is just distance covered in that time. Removing the
    # target distance removes an arbitrary number to tune -- speed pressure comes
    # for free from "more distance in the same fixed time" rather than from a
    # bonus for crossing a line.
    # Alive bonus DISABLED by default: a per-step survival reward dominates the
    # return (0.05 * 1000 steps = 50), lets a body score highly by lying still,
    # and removes any need to learn. Keep at 0 so scoring requires moving.
    healthy_reward: float = 0.0
    terminate_on_fall: bool = True    # end the episode if the root collapses — without
                                      # this, a body can faceplant and still bank forward
                                      # reward by drifting, so no gait needs to be learned
    # "fallen" = root below this fraction of its SETTLED height (measured
    # per-body after the settle phase). Was 0.6 -- too lenient, let a body
    # topple most of the way down before cutoff, banking a lot of
    # "controlled fall" distance reward on the way; raised to 0.75 to close
    # that off. Briefly tried loosening to 0.65 (experiments/run19) to give
    # partial recoveries more room -- reverted: it let a body survive far
    # longer (up to 951/1000 steps) by rocking in place near-zero net
    # displacement instead of moving forward, since avoiding the fall/height
    # penalties that way was cheaper than risking a fall by actually
    # advancing. 0.75 doesn't have that failure mode.
    min_z_frac: float = 0.75
    # Continuous penalty for dropping below settled height (see module
    # docstring); 0 disables it. QUADRATIC in the height deficit (not linear)
    # and weighted much higher than the original 2.0 -- a linear penalty at
    # that weight was cheap enough that a slow, continuous lean-and-sink
    # ("controlled falling": bank forward-x reward from gravity pulling the
    # body forward while it topples, get cut off right at the fall line)
    # actually out-earned standing still, confirmed on experiments/run16's
    # champion: height dropped every single step from frame 0, reward kept
    # climbing as it sank, right up to termination. Quadratic keeps small
    # natural bobbing cheap (a real gait dips a little) while making a
    # sustained sink toward the fall line increasingly expensive, so leaning
    # into a slow topple is a net loss well before it reaches the cutoff.
    height_reward_weight: float = 20.0
    # Falling used to cost a flat -1, then a fixed -8 scaled by fraction of
    # horizon forfeited -- neither scales with how much reward the episode
    # had already banked. A fast morphology can lunge for a burst of forward
    # reward (proportional to distance covered, via the /dt term below) worth
    # far more than any fixed penalty, then eat the capped fall cost and
    # still net a large positive return -- confirmed on experiments/run22's
    # champion, which fell at ~1s into every 25s episode and still won
    # selection, because +100+ banked from the lunge dwarfed an at-most -8
    # fixed penalty. Clawing back a fraction of THIS EPISODE'S OWN banked
    # reward on a fall closes that off regardless of how large the banked
    # amount is -- a lunge worth 10 or 1000 is equally unprofitable to follow
    # with a fall, so this doesn't need re-tuning as bodies get faster.
    # Tried 0.5 (half clawback) after experiments/run24 showed a third of ALL
    # individuals across a fresh population-scale run, including the reported
    # champion, stuck at final=0.0 -- hypothesis was that full clawback erases
    # the gradient between "no progress" and "real partial progress" before a
    # fall. Controlled A/B on two known bodies disproved it: run24's own stuck
    # champion (a 2-limb/1-actuator body) fell at step 8 regardless of the
    # weight -- it's morphologically incapable of standing, not reward-
    # starved, so there was no gradient to rescue. Worse, run22's champion --
    # which reliably reached full-horizon survival (return 283, 4/5 seeds) at
    # frac=1.0 -- regressed to falling in ALL 5 seeds at frac=0.5, because
    # half a clawback still leaves a profitable lunge-and-fall (banks ~100,
    # keeps ~50). Reverted to full clawback.
    fall_penalty_frac: float = 1.0
    # fall_penalty_frac=1.0 claws back ALL of an episode's own positive
    # banked reward on a fall, by design (see above) -- but that means EVERY
    # fall nets to exactly the same reward (0), whether it happened after 5
    # steps or 500. PPO gets no gradient distinguishing "closer to
    # succeeding" from "failed immediately," since both look identical.
    # Confirmed happening in practice, not just in theory: experiments/
    # run28's champion had its return lock at EXACTLY 0.00 from step 5,000
    # of training onward, all the way to 200,000 -- not still learning, just
    # stuck with nothing to climb. This adds a small, ALWAYS-preserved (not
    # subject to the clawback above) bonus for absolute distance reached
    # before a fall, so different failures are no longer indistinguishable.
    # Small and capped well below what continuing to walk would earn (a
    # successful full episode's accumulated forward_reward), so it rewards
    # getting further, without making "get some distance, then fall on
    # purpose" competitive with "don't fall" -- that comparison is exactly
    # what fall_penalty_frac=1.0 exists to keep unprofitable, and this sits
    # alongside it rather than undoing it. 0 disables it.
    fall_distance_bonus_weight: float = 0.5
    reset_noise: float = 0.01
    settle_steps: int = 30            # let the body settle before reward counts, so the
                                      # one-time "topple forward" drop isn't free reward
    init_z: float | None = None       # None => derived from morphology height
    # Real motors trade off torque against speed -- they can't deliver full
    # torque at any speed simultaneously the way our flat forcerange cap does.
    # Each actuator's available force is scaled down linearly toward 0 as its
    # joint's speed approaches max_joint_speed (a simple, standard DC-motor
    # torque-speed line), on top of the existing static cap from limb mass.
    # An order-of-magnitude default for small hobby/robotic servos, not a
    # measured constant -- same spirit as TORQUE_DENSITY_NM_PER_KG.
    max_joint_speed: float = 8.0      # rad/s; force cap reaches 0 at this speed
    # Real encoders aren't perfect -- Gaussian noise on the joint-angle
    # observation only (not velocity/contact), since "rotation sensor" was the
    # specific missing piece. ~0.01 rad (~0.6 deg) is a plausible order of
    # magnitude for a low-cost encoder/potentiometer.
    joint_angle_noise_std: float = 0.01


class MorphologyLocomotionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, genome: Genome, task: TaskConfig | None = None, seed: int | None = None):
        self.task = task or TaskConfig()
        self.genome = genome

        xml = MujocoBuilder().to_xml_string(genome.to_robot_tree())
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        if self.model.nu == 0:
            raise ValueError("Morphology has no actuators; it is not trainable.")

        self._init_z = self.task.init_z if self.task.init_z is not None \
            else max(genome.root.height, 0.3)
        # Reference for the standing-height reward below -- root.height alone
        # (the trunk capsule's own length) is too weak a bar whenever the
        # actual legs are sub-limbs chained off a short trunk, so a body can
        # sit at ~100% of "standing" already collapsed flat (see genome.py's
        # max_reach() docstring for how this was found).
        # 0.6x, not the full reach: max_reach() is a straight-line upper
        # bound (the chain fully extended in one direction), but a real
        # jointed leg standing on the ground is bent, not a rigid vertical
        # pole -- demanding the full straight-line length as "100% standing"
        # is an unreachable bar for most joint geometries. Confirmed via a
        # controlled A/B on run25's champion: at the full reach (1.07m) it
        # never got past ~0.4m (37% of target) and returns collapsed
        # (~1100 vs. the original 4021); the discount was punishing a body
        # for a bar no bent leg could plausibly reach, not rewarding real
        # standing.
        self._stand_ref_z = max(0.6 * genome.max_reach(), self._init_z, 0.1)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        # Action -> ctrl mapping from the actuators' control ranges.
        cr = self.model.actuator_ctrlrange.copy()
        limited = self.model.actuator_ctrllimited.astype(bool)
        self._ctrl_low = np.where(limited, cr[:, 0], -1.0)
        self._ctrl_high = np.where(limited, cr[:, 1], 1.0)

        # Static per-actuator force cap from limb mass (see genome.py), and the
        # qvel index each actuator's joint lives at, so the torque-speed limit
        # in step() can scale the former from the latter each step.
        self._static_forcerange = self.model.actuator_forcerange.copy()
        self._actuator_dof = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])

        obs0 = self._get_obs()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=obs0.shape, dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)  # kept for any non-reset sampling
        self._step_count = 0
        self._settled_z = self._init_z
        self._healthy_z = 0.0  # set per-episode in reset() from the settled height
        self._episode_reward = 0.0  # set per-episode in reset()
        self._prev_action = np.zeros(self.model.nu, dtype=np.float32)  # set per-episode in reset()

    # ---------------------------------------------------------------- #
    def _contact_vector(self) -> np.ndarray:
        """1.0 per body (root + every limb) currently touching the floor, else
        0.0 -- a ground-contact signal for the policy, since qpos/qvel alone
        don't expose it. Fixed length per body (nbody-1, excluding world), so
        this stays consistent with the rest of the per-body observation sizing."""
        contact = np.zeros(self.model.nbody - 1, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._floor_gid:
                contact[self.model.geom_bodyid[c.geom2] - 1] = 1.0
            elif c.geom2 == self._floor_gid:
                contact[self.model.geom_bodyid[c.geom1] - 1] = 1.0
        return contact

    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self.data.qpos, self.data.qvel
        # Drop absolute root x,y for translation invariance; keep height, orientation,
        # all joint angles, all velocities, and per-body floor contact.
        root_z = qpos[2:3]
        root_quat = qpos[3:7]
        # A real encoder isn't noiseless -- see TaskConfig.joint_angle_noise_std.
        # np_random auto-initializes (Gymnasium API) even before reset() has
        # run, which is fine here since __init__ calls _get_obs() once just to
        # size the observation space.
        joint_pos = qpos[7:].copy()
        if self.task.joint_angle_noise_std > 0:
            joint_pos += self.np_random.normal(0.0, self.task.joint_angle_noise_std, size=joint_pos.shape)
        contact = self._contact_vector()
        return np.concatenate([root_z, root_quat, joint_pos, qvel, contact]).astype(np.float32)

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, -1.0, 1.0)
        return self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)  # seeds self.np_random per the Gymnasium API
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self._init_z
        n = self.task.reset_noise
        self.data.qpos[:] += self.np_random.uniform(-n, n, size=self.model.nq)
        self.data.qvel[:] += self.np_random.uniform(-n, n, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        # Let the body settle under gravity with no control, so the initial
        # topple/drop doesn't hand out free forward reward in the first steps.
        for _ in range(self.task.settle_steps):
            mujoco.mj_step(self.model, self.data)
        # The body's natural resting height defines "healthy" for THIS morphology,
        # so the fall check adapts to tall and low bodies alike.
        self._settled_z = float(self.data.qpos[2])
        self._healthy_z = self.task.min_z_frac * self._settled_z
        self._step_count = 0
        self._episode_reward = 0.0  # tracked so a fall can claw back what THIS episode banked
        self._prev_action = np.zeros(self.model.nu, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        x_before = float(self.data.qpos[0])
        self.data.ctrl[:] = self._map_action(action)
        # Torque-speed limit: scale each actuator's static force cap down
        # linearly toward 0 as its joint's current speed approaches
        # max_joint_speed, like a real DC motor's torque-speed line. Computed
        # once per env step (from the speed at the start of it) rather than
        # every physics substep, as a pragmatic cost/accuracy tradeoff.
        joint_speed = np.abs(self.data.qvel[self._actuator_dof])
        speed_frac = np.clip(1.0 - joint_speed / self.task.max_joint_speed, 0.0, 1.0)
        self.model.actuator_forcerange[:, 0] = self._static_forcerange[:, 0] * speed_frac
        self.model.actuator_forcerange[:, 1] = self._static_forcerange[:, 1] * speed_frac
        for _ in range(self.task.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = float(self.data.qpos[0])

        dt = self.model.opt.timestep * self.task.frame_skip
        forward = (x_after - x_before) / dt
        # Forward reward is credited in proportion to how upright the body
        # currently is (root height as a fraction of its own longest-limb-
        # chain reach, self._stand_ref_z -- see genome.max_reach()). Without
        # this, "cover distance" and "stay upright" are entirely independent,
        # so a body that collapses onto a stable, low sprawl (cheap to keep
        # "healthy" -- height_reward_weight above only guards against SINKING
        # below wherever it already settled, not against never rising in the
        # first place) earns full forward credit for creeping along on the
        # ground. Confirmed exactly this on experiments/run25's champion: it
        # flattened out within the first ~5m of a 43m walk and never rose
        # again for the rest of the episode. Scaling forward reward directly,
        # rather than adding a separate standing bonus, avoids opening a new
        # "stand still and collect reward" exploit (see healthy_reward's
        # comment above) -- standing upright without moving still earns zero
        # from this term, same as before. Scales with the body's OWN
        # geometry, not a universal height, so a genuinely short-legged
        # design isn't penalized for being short -- only for being lower
        # than ITS OWN reach, same gait-agnostic principle as
        # height_reward_weight.
        stand_frac = np.clip(self.data.qpos[2] / self._stand_ref_z, 0.0, 1.0)
        forward_reward = self.task.forward_reward_weight * forward * stand_frac
        ctrl_cost = self.task.ctrl_cost_weight * float(np.mean(np.square(action)))
        action_rate_cost = self.task.action_rate_weight * float(np.mean(np.square(action - self._prev_action)))
        self._prev_action = np.asarray(action, dtype=np.float32).copy()
        # Only penalizes dropping BELOW the body's own settled height, never being
        # above it, so this doesn't bias tall-standing over a genuinely low crawl --
        # it targets "progressively sinking during a fall" specifically.
        height_frac = self.data.qpos[2] / max(self._settled_z, 1e-6)
        height_deficit = max(0.0, 1.0 - height_frac)
        height_penalty = -self.task.height_reward_weight * height_deficit ** 2
        reward = (
            forward_reward
            - ctrl_cost
            - action_rate_cost
            + self.task.healthy_reward
            + height_penalty
        )
        self._episode_reward += reward

        self._step_count += 1
        terminated = False
        if self.task.terminate_on_fall and self.data.qpos[2] < self._healthy_z:
            terminated = True
            # Claw back a fraction of THIS episode's own banked reward -- makes a
            # fall unprofitable regardless of how large the lunge that preceded it
            # was, instead of competing against a fixed penalty that has to be
            # re-tuned every time a faster/bigger lunge is discovered.
            fall_penalty = self.task.fall_penalty_frac * max(0.0, self._episode_reward)
            reward -= fall_penalty
            self._episode_reward -= fall_penalty
            # Immune to the clawback above -- see fall_distance_bonus_weight.
            distance_bonus = self.task.fall_distance_bonus_weight * max(0.0, x_after)
            reward += distance_bonus
            self._episode_reward += distance_bonus
        truncated = self._step_count >= self.task.horizon

        obs = self._get_obs()
        # Unstable bodies can blow up; clamp and end the episode rather than crash.
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated, reward = True, -1.0

        return obs, float(reward), terminated, truncated, {
            "forward": forward, "x": x_after, "steps": self._step_count,
        }