"""Render a saved champion's episode straight to a GIF, camera auto-framed to
the robot (not the floor). Sidesteps the interactive viewer's camera entirely
-- no scrolling/aligning needed, just open the file.

Usage:
    ./.env/bin/python -m scripts.render_episode --run experiments/run5 --gen 24
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import mujoco
from PIL import Image
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


def _fit_camera(model, data) -> mujoco.MjvCamera:
    """Frame the camera on the robot's actual geoms, ignoring the (huge) floor
    plane, so small bodies don't end up as a speck or a giant close-up blob."""
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pts = np.array([data.geom_xpos[g] for g in range(model.ngeom) if g != floor_gid])
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1))) + 0.25  # + margin for geom extent

    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = max(radius * 3.0, 0.6)
    cam.azimuth = 120
    cam.elevation = -20
    return cam


def render(genome_path, policy_path, algo, horizon, seed,
          out_path, width, height, hold_frames):
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    policy = _load_policy(policy_path, algo)

    task = TaskConfig(horizon=horizon)
    env = MorphologyLocomotionEnv(genome, task, seed=seed)
    obs, _ = env.reset(seed=seed)

    renderer = mujoco.Renderer(env.model, height=height, width=width)

    frames = []
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        # Recompute every frame so the camera tracks the body across the whole
        # episode -- a one-time fit (the old behaviour) freezes on the start
        # pose, so any body that travels more than a body-length just walks
        # out of frame and the rest of the GIF shows empty ground, looking
        # like it stopped moving even when the policy is fully active.
        cam = _fit_camera(env.model, env.data)
        renderer.update_scene(env.data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))
        done = terminated or truncated

    # Hold the final (outcome) frame a bit longer so it's easy to see how it ended.
    frames.extend([frames[-1]] * hold_frames)

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=50, loop=0)

    if info["steps"] >= horizon:
        outcome = f"ran full horizon, x={info['x']:.3f} m"
    else:
        outcome = f"fell at step {info['steps']}, x={info['x']:.3f} m"
    print(f"Body: {genome.count_limbs()} limbs, {genome.num_actuators()} actuators")
    print(f"Episode: {outcome}")
    print(f"Saved {len(frames)} frames -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--gen", type=int, default=None)
    p.add_argument("--genome", type=str, default=None)
    p.add_argument("--policy", type=str, default=None)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--out", type=str, default="episode.gif")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--hold-frames", type=int, default=15, help="extra copies of the last frame")
    args = p.parse_args()

    genome_path, policy_path = _resolve_paths(args)
    render(genome_path, policy_path, args.algo, args.horizon,
          args.seed, args.out, args.width, args.height, args.hold_frames)


if __name__ == "__main__":
    main()
