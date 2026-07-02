"""Gymnasium locomotion environment over an arbitrary evolved morphology.

A morphology is compiled to MuJoCo on construction. Observation and action
spaces are sized from the compiled model, so every body — whatever its limb
count — gets its own correctly-shaped policy trained from scratch.

The task is a fair, standardized reach-a-target locomotion task:
  - Every body starts at the same place, aims for the same finish line
    (target_distance in +x), under the same time cap (horizon) — so scores are
    comparable across morphologies.
  - Distance is rewarded continuously (dense signal PPO can learn from), and the
    episode ENDS with a speed bonus the moment the body crosses the target — the
    bonus shrinks the longer it took, so evolution is pressured toward SPEED.
  - A body that never reaches the target is still graded on distance covered, and
    in a fixed time cap "more distance" means "faster" — so the task discriminates
    on speed at every level and becomes pure reach-target-by-time once bodies are
    strong enough to reach the finish line.
  - On reset the body settles under gravity before reward counts; the episode ends
    early if the root collapses (terminate_on_fall), so faceplant-and-drift can't
    win. An untrained policy scores ~0, so scoring requires learning to move.
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
    horizon: int = 1000          # max env steps to reach the target (the time cap)
    frame_skip: int = 5          # sim substeps per env step
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.001
    # Reach-a-target task: same start, same finish line, same clock for every body
    # (fair + standardized). The episode ENDS when the root crosses target_distance
    # in +x, with a bonus that shrinks the longer it took -> pressure on SPEED.
    # Distance is also rewarded continuously, so a body that never reaches the
    # target still gets graded on how far it got (and faster = further in the same
    # time). Set target_distance as a stretch goal; it stays dormant until bodies
    # are strong enough to reach it, at which point the speed bonus activates.
    target_distance: float = 0.5    # metres in +x that "finishes" the episode
    arrival_bonus: float = 5.0      # reward for finishing, scaled by fraction of time left
    # Alive bonus DISABLED by default: a per-step survival reward dominates the
    # return (0.05 * 1000 steps = 50), lets a body score highly by lying still,
    # and removes any need to learn. Keep at 0 so scoring requires moving.
    healthy_reward: float = 0.0
    terminate_on_fall: bool = True    # end the episode if the root collapses — without
                                      # this, a body can faceplant and still bank forward
                                      # reward by drifting, so no gait needs to be learned
    min_z_frac: float = 0.6           # "fallen" = root below this fraction of its SETTLED
                                      # height (measured per-body after the settle phase)
    reset_noise: float = 0.01
    settle_steps: int = 30            # let the body settle before reward counts, so the
                                      # one-time "topple forward" drop isn't free reward
    init_z: float | None = None       # None => derived from morphology height


class MorphologyLocomotionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, genome: Genome, task: TaskConfig | None = None, seed: int | None = None):
        self.task = task or TaskConfig()
        self.genome = genome

        xml = MujocoBuilder().to_xml_string(genome.to_robot_tree())
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

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

        obs0 = self._get_obs()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=obs0.shape, dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)  # kept for any non-reset sampling
        self._step_count = 0
        self._settled_z = self._init_z
        self._healthy_z = 0.0  # set per-episode in reset() from the settled height

    # ---------------------------------------------------------------- #
    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self.data.qpos, self.data.qvel
        # Drop absolute root x,y for translation invariance; keep height, orientation,
        # all joint angles, and all velocities.
        root_z = qpos[2:3]
        root_quat = qpos[3:7]
        joint_pos = qpos[7:]
        return np.concatenate([root_z, root_quat, joint_pos, qvel]).astype(np.float32)

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
        for _ in range(self.task.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = float(self.data.qpos[0])

        dt = self.model.opt.timestep * self.task.frame_skip
        forward = (x_after - x_before) / dt
        ctrl_cost = self.task.ctrl_cost_weight * float(np.sum(np.square(action)))
        reward = (
            self.task.forward_reward_weight * forward
            - ctrl_cost
            + self.task.healthy_reward
        )

        self._step_count += 1
        terminated = False
        reached = x_after >= self.task.target_distance
        if reached:
            # Finished: bonus scaled by how much of the time cap is left -> faster wins.
            frac_time_left = max(0.0, 1.0 - self._step_count / self.task.horizon)
            reward += self.task.arrival_bonus * frac_time_left
            terminated = True
        elif self.task.terminate_on_fall and self.data.qpos[2] < self._healthy_z:
            terminated, reward = True, reward - 1.0
        truncated = self._step_count >= self.task.horizon

        obs = self._get_obs()
        # Unstable bodies can blow up; clamp and end the episode rather than crash.
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated, reward = True, -1.0

        return obs, float(reward), terminated, truncated, {
            "forward": forward, "x": x_after,
            "reached": reached, "steps": self._step_count,
        }