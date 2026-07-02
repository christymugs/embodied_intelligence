"""Inner loop: train a controller for ONE morphology from scratch and record
its learning curve (evaluation return vs training timesteps).

The learning curve — not just the final number — is the object the learnability
metrics in `fitness/learnability.py` consume. Every morphology is trained
independently with a freshly initialised policy, which is what makes the
measured quantity *learnability* rather than transferred skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from morphology.genome import Genome
from learning.env import MorphologyLocomotionEnv, TaskConfig


@dataclass
class TrainConfig:
    algo: str = "ppo"               # "ppo" | "sac"
    lifetime_steps: int = 200_000   # SHORT fixed budget => selects for fast learners
    eval_every: int = 10_000
    eval_episodes: int = 3
    policy: str = "MlpPolicy"
    n_steps: int = 2048             # PPO rollout length
    batch_size: int = 64
    learning_rate: float = 3e-4
    seed: int = 0
    verbose: int = 0
    device: str = "cpu"


class _CurveCallback(BaseCallback):
    """Evaluate the current policy every `eval_every` steps; store the curve."""

    def __init__(self, eval_env, eval_every: int, eval_episodes: int):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.timesteps: list[int] = []
        self.returns: list[float] = []
        self._last_eval = 0

    def _evaluate(self) -> float:
        rets = []
        for ep in range(self.eval_episodes):
            obs, _ = self.eval_env.reset(seed=10_000 + ep)
            done, total = False, 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = self.eval_env.step(action)
                total += r
                done = term or trunc
            rets.append(total)
        return float(np.mean(rets))

    def _on_training_start(self) -> None:
        # t=0 point: performance of the *untrained* policy (the learning baseline).
        self.timesteps.append(0)
        self.returns.append(self._evaluate())

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_every:
            self._last_eval = self.num_timesteps
            self.timesteps.append(int(self.num_timesteps))
            self.returns.append(self._evaluate())
        return True


def train_morphology(genome: Genome, task: TaskConfig | None = None,
                     cfg: TrainConfig | None = None,
                     save_path: str | None = None) -> dict:
    """Train `genome`'s controller and return its learning curve + final model.

    If `save_path` is given, the trained policy is written there as a .zip so it
    can be reloaded and replayed later (e.g. by scripts/visualizer.py).
    """
    task = task or TaskConfig()
    cfg = cfg or TrainConfig()

    train_env = Monitor(MorphologyLocomotionEnv(genome, task, seed=cfg.seed))
    eval_env = MorphologyLocomotionEnv(genome, task, seed=cfg.seed + 999)

    if cfg.algo == "ppo":
        model = PPO(
            cfg.policy, train_env, n_steps=cfg.n_steps, batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate, seed=cfg.seed,
            verbose=cfg.verbose, device=cfg.device,
        )
    elif cfg.algo == "sac":
        model = SAC(
            cfg.policy, train_env, learning_rate=cfg.learning_rate,
            seed=cfg.seed, verbose=cfg.verbose, device=cfg.device,
        )
    else:
        raise ValueError(f"unknown algo: {cfg.algo}")

    cb = _CurveCallback(eval_env, cfg.eval_every, cfg.eval_episodes)
    model.learn(total_timesteps=cfg.lifetime_steps, callback=cb, progress_bar=False)

    if save_path is not None:
        model.save(save_path)

    train_env.close()
    eval_env.close()
    return {"timesteps": cb.timesteps, "returns": cb.returns, "model": model}