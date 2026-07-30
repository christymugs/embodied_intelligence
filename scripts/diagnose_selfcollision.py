"""Diagnose whether a champion's reward is coming from self-collision jitter
rather than a learned gait.

The morphology builder (morphology/mujoco_builder.py) sets no contype/
conaffinity/exclude, so MuJoCo's defaults apply: any two limb geoms that are
NOT direct parent/child in the kinematic tree can collide, and nothing in
genome.py / mutation.py prevents evolution from placing limbs so they
physically overlap. This script rolls out one episode under a saved champion's
trained policy and classifies every contact each step as floor-vs-limb or
limb-vs-limb (self-collision), then reports how much of the forward
displacement happens on self-collision steps vs. floor-only steps. It also
checks for overlap at the settled resting pose, before any control is applied.

Usage:
    ./.env/bin/python -m scripts.diagnose_selfcollision --run experiments/run3 --gen 24
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import mujoco
from stable_baselines3 import PPO, SAC

from learning.env import MorphologyLocomotionEnv, TaskConfig


def _resolve_paths(args) -> tuple[str, str]:
    if args.genome and args.policy:
        return args.genome, args.policy
    if args.run is not None and args.gen is not None:
        g = os.path.join(args.run, f"gen{args.gen:03d}_best_genome.pkl")
        p = os.path.join(args.run, f"gen{args.gen:03d}_best_policy.zip")
        return g, p
    raise SystemExit("Give either --run RUN --gen N, or --genome FILE --policy FILE.")


def _load_policy(path: str, algo: str):
    if not os.path.exists(path):
        raise SystemExit(f"Policy not found: {path}")
    return (SAC if algo == "sac" else PPO).load(path, device="cpu")


def _floor_geom_id(model) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def _classify_contacts(model, data, floor_gid: int):
    """Split this step's contacts into floor-vs-limb force and limb-vs-limb
    (self-collision) force, and name the self-colliding body pairs."""
    floor_force = 0.0
    self_force = 0.0
    self_pairs = set()
    result = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, result)
        force_mag = float(np.linalg.norm(result[:3]))
        if c.geom1 == floor_gid or c.geom2 == floor_gid:
            floor_force += force_mag
        else:
            self_force += force_mag
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            self_pairs.add(tuple(sorted((model.body(b1).name, model.body(b2).name))))
    return floor_force, self_force, self_pairs


def diagnose(genome_path, policy_path, algo, horizon, seed):
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    policy = _load_policy(policy_path, algo)

    task = TaskConfig(horizon=horizon)
    env = MorphologyLocomotionEnv(genome, task, seed=seed)
    floor_gid = _floor_geom_id(env.model)

    print(f"Body: {genome.count_limbs()} limbs, {genome.num_actuators()} actuators, "
          f"max depth {genome.max_depth()}")

    # --- Static overlap check: settled pose, before any control ---
    obs, _ = env.reset(seed=seed)
    mujoco.mj_forward(env.model, env.data)
    _, static_self_force, static_pairs = _classify_contacts(env.model, env.data, floor_gid)
    print(f"\nStatic (settled, zero control) self-collision force: {static_self_force:.4f}")
    if static_pairs:
        print(f"  self-contact pairs at rest: {sorted(static_pairs)}")

    # --- Rollout under the trained policy ---
    x_prev = float(env.data.qpos[0])
    rows = []
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        floor_force, self_force, self_pairs = _classify_contacts(env.model, env.data, floor_gid)
        x_now = info["x"]
        rows.append({
            "dx": x_now - x_prev, "forward": info["forward"], "reward": reward,
            "floor_force": floor_force, "self_force": self_force, "self_pairs": self_pairs,
        })
        x_prev = x_now
        done = terminated or truncated

    dx = np.array([r["dx"] for r in rows])
    self_force = np.array([r["self_force"] for r in rows])
    self_active = self_force > 1e-6

    total_dx = dx.sum()
    dx_during_self = dx[self_active].sum() if self_active.any() else 0.0
    frac_steps_self = self_active.mean()

    if info["steps"] >= horizon:
        outcome = f"ran full horizon, x={info['x']:.3f} m"
    else:
        outcome = f"fell at step {info['steps']}, x={info['x']:.3f} m"

    print(f"\nEpisode: {len(rows)} steps, {outcome}")
    print(f"Total x displacement: {total_dx:.4f} m")
    print(f"Steps with self-collision contact: {frac_steps_self * 100:.1f}% of episode")
    print(f"x displacement during self-collision steps: {dx_during_self:.4f} m "
          f"({100 * dx_during_self / total_dx if total_dx else 0:.1f}% of total)")
    if self_active.any():
        corr = np.corrcoef(self_force, dx)[0, 1]
        print(f"Correlation(self-collision force, per-step dx): {corr:.3f}")
        all_pairs = set()
        for r in rows:
            all_pairs |= r["self_pairs"]
        print(f"Distinct self-colliding body pairs seen: {sorted(all_pairs)}")
    else:
        print("No self-collision contacts detected during the episode.")

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default=None, help="Run directory, e.g. experiments/run3")
    p.add_argument("--gen", type=int, default=None, help="Which generation's champion to check")
    p.add_argument("--genome", type=str, default=None, help="Explicit genome .pkl")
    p.add_argument("--policy", type=str, default=None, help="Explicit policy .zip")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    # Default comes from TaskConfig() itself (see scripts/run_evolution.py), so this
    # can't silently diverge from what a run actually trained with.
    _default_task = TaskConfig()
    p.add_argument("--horizon", type=int, default=_default_task.horizon)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    genome_path, policy_path = _resolve_paths(args)
    diagnose(genome_path, policy_path, args.algo, args.horizon, args.seed)


if __name__ == "__main__":
    main()
