"""Train ONE body and inspect its learning curve — fast task-design iteration.

The point: before running a multi-hour evolution, confirm a single body can
actually LEARN to locomote on the current task. If its curve is flat (trained
≈ untrained), the task doesn't require learning and evolution will select for
passive movers, not learnable ones. Tune the task here until the curve rises,
THEN run evolution.

    python -m scripts.check_learning --steps 300000
    python -m scripts.check_learning --steps 300000 --no-terminate --healthy-reward 0.0
    python -m scripts.check_learning --steps 300000 --min-z-frac 0.5 --horizon 1000

Writes single_curve.png in the run dir and prints the curve + a verdict.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from morphology.genome import Genome, MorphologyBounds
from learning.env import TaskConfig
from learning.train import TrainConfig, train_morphology
from fitness.learnability import compute_learnability


def main():
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0, help="Which random body to build.")
    p.add_argument("--steps", type=int, default=300_000, help="Training budget (one lifetime).")
    p.add_argument("--eval-every", type=int, default=20_000)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--healthy-reward", type=float, default=0.0)
    p.add_argument("--min-z-frac", type=float, default=0.6)
    p.add_argument("--no-terminate", action="store_true",
                   help="Disable fall termination (to compare against the broken task).")
    p.add_argument("--out", type=str, default="experiments/check")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Build one body that actually has actuators.
    rng = np.random.default_rng(a.seed)
    genome = Genome.random(MorphologyBounds(), rng)
    print(f"Body: {genome.count_limbs()} limbs, depth {genome.max_depth()}, "
          f"{genome.num_actuators()} actuators")

    # Save body so it can be inspected in MuJoCo (named by seed, not by
    # generation — it is a random test body, unrelated to any evolved gen*_best.xml).
    from morphology.mujoco_builder import MujocoBuilder
    body_path = os.path.join(a.out, f"seed{a.seed}_body.xml")
    MujocoBuilder().save(genome.to_robot_tree(), body_path)
    print(f"  saved this body: {body_path}")
    print(f"  view it: mjpython -m mujoco.viewer --mjcf {body_path}")

    task = TaskConfig(
        horizon=a.horizon,
        healthy_reward=a.healthy_reward,
        terminate_on_fall=not a.no_terminate,
        min_z_frac=a.min_z_frac,
    )
    cfg = TrainConfig(algo=a.algo, lifetime_steps=a.steps, eval_every=a.eval_every,
                      eval_episodes=3, seed=a.seed)

    print(f"Training for {a.steps:,} steps "
          f"(terminate_on_fall={task.terminate_on_fall}, healthy_reward={task.healthy_reward})...")
    out = train_morphology(genome, task, cfg)
    lr = compute_learnability(out["timesteps"], out["returns"])

    print("\n  timestep      return")
    for t, r in zip(lr.curve_t, lr.curve_r):
        print(f"  {int(t):>9,}   {r:8.2f}")

    gain = lr.curve_r[-1] - lr.curve_r[0]
    final = lr.curve_r[-1]
    print(f"\nuntrained -> trained gain: {gain:+.2f}")
    print(f"steps_to_threshold       : {lr.steps_to_threshold:,.0f}")

    # Roll out the trained policy a few times to report the interpretable
    # task numbers: how far it actually gets, and whether/how fast it finishes.
    from learning.env import MorphologyLocomotionEnv
    model = out["model"]
    eval_env = MorphologyLocomotionEnv(genome, task, seed=a.seed + 7)
    dists, fell_steps = [], []
    for ep in range(5):
        obs, _ = eval_env.reset(seed=20_000 + ep)
        done = False
        info = {"x": 0.0, "steps": 0}
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(act)
            done = term or trunc
        dists.append(info["x"])
        if info["steps"] < task.horizon:
            fell_steps.append(info["steps"])
    print(f"\ntrained rollout over 5 episodes (fixed horizon = {task.horizon} steps, no finish line):")
    print(f"  mean distance covered : {np.mean(dists):+.3f} m  (max {max(dists):+.3f})  <- the speed number")
    if fell_steps:
        print(f"  fell early in {len(fell_steps)}/5 episodes (mean step {np.mean(fell_steps):.0f})")
    else:
        print(f"  never fell -- ran the full horizon in all 5 episodes")

    if gain > 0.5:
        print("VERDICT: this body LEARNS within a lifetime (curve rises). Good — evolution")
        print("  can select for bodies that learn faster (earlier-rising curves).")
    elif final < 0.5:
        print("VERDICT: flat and stuck near the floor. This body can't get off the ground —")
        print("  it collapses immediately, so the policy never gets a gradient. That's almost")
        print("  certainly an UNLEARNABLE morphology, not a task bug, and it's a fair loser")
        print("  under learnability selection. Only worry if NEARLY ALL random bodies look")
        print(f"  like this. Watch it fall: mjpython -m mujoco.viewer --mjcf {body_path}")
    else:
        print("VERDICT: flat but already scoring high untrained — the task doesn't require")
        print("  learning for this body (reward-hackable). Make the task harder before trusting it.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lr.curve_t, lr.curve_r, "-o", color="C0")
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("evaluation return")
    ax.set_title(f"Single-body learning curve (seed {a.seed})")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "single_curve.png"), dpi=150)
    print(f"\nWrote {a.out}/single_curve.png")


if __name__ == "__main__":
    main()