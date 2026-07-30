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
    # Falling used to cost a flat -1 regardless of when it happened -- nowhere
    # near enough to discourage "sprint hard, then let it fall" once a body had
    # already banked a good burst of forward reward, since it only forfeits a
    # tiny fixed amount for throwing away the REST of the episode's distance
    # opportunity. Now scaled by the fraction of the horizon thrown away, so
    # falling early (most time left, most distance opportunity forfeited)
    # costs close to the full penalty, and falling near the natural end costs
    # almost nothing.
    fall_penalty_weight: float = 8.0
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
        ctrl_cost = self.task.ctrl_cost_weight * float(np.mean(np.square(action)))
        # Only penalizes dropping BELOW the body's own settled height, never being
        # above it, so this doesn't bias tall-standing over a genuinely low crawl --
        # it targets "progressively sinking during a fall" specifically.
        height_frac = self.data.qpos[2] / max(self._settled_z, 1e-6)
        height_deficit = max(0.0, 1.0 - height_frac)
        height_penalty = -self.task.height_reward_weight * height_deficit ** 2
        reward = (
            self.task.forward_reward_weight * forward
            - ctrl_cost
            + self.task.healthy_reward
            + height_penalty
        )

        self._step_count += 1
        terminated = False
        if self.task.terminate_on_fall and self.data.qpos[2] < self._healthy_z:
            frac_time_forfeited = max(0.0, 1.0 - self._step_count / self.task.horizon)
            terminated = True
            reward -= self.task.fall_penalty_weight * frac_time_forfeited
        truncated = self._step_count >= self.task.horizon

        obs = self._get_obs()
        # Unstable bodies can blow up; clamp and end the episode rather than crash.
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated, reward = True, -1.0

        return obs, float(reward), terminated, truncated, {
            "forward": forward, "x": x_after, "steps": self._step_count,
        }