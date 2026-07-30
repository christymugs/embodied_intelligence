"""Watch a trained champion actually move — and stop when its episode ends.

This replaces the old constant-control viewer (which drove every actuator to a
fixed 0.5 and looped until you hit Ctrl-C). Instead it loads a saved champion —
its genome (to rebuild the exact body) and its trained policy (its learned
"brain") — and runs ONE real episode through the same MorphologyLocomotionEnv it
was trained in. The policy chooses the actions; the episode ends on its own when
the body reaches the target, runs out of time, or falls. That is the "have an
end instead of Ctrl-C" behaviour: the simulation terminates by itself, then the
window simply holds the final frame until you close it.

macOS note: MuJoCo's interactive viewer must be launched with `mjpython`, e.g.
    mjpython -m scripts.visualizer --run experiments/run2 --gen 24

Point it at a run + generation (loads that generation's champion), or give
explicit --genome / --policy paths.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC

from learning.env import MorphologyLocomotionEnv, TaskConfig


def _resolve_paths(args) -> tuple[str, str]:
    """Figure out the genome and policy files from --run/--gen or explicit paths."""
    if args.genome and args.policy:
        return args.genome, args.policy
    if args.run is not None and args.gen is not None:
        g = os.path.join(args.run, f"gen{args.gen:03d}_best_genome.pkl")
        p = os.path.join(args.run, f"gen{args.gen:03d}_best_policy.zip")
        return g, p
    raise SystemExit("Give either --run RUN --gen N, or --genome FILE --policy FILE.")


def _load_policy(path: str, algo: str):
    if not os.path.exists(path):
        raise SystemExit(f"Policy not found: {path}\n"
                         "(Only runs made AFTER policy-saving was added have these.)")
    return (SAC if algo == "sac" else PPO).load(path, device="cpu")


def run_episode(env, policy, viewer, dt: float) -> dict:
    """Step one episode under the policy, syncing the viewer, until it ends."""
    obs, _ = env.reset(seed=np.random.randint(0, 100_000))
    viewer.cam.lookat[:] = env.data.qpos[0:3]
    viewer.sync()
    done = False
    info = {"x": 0.0, "steps": 0}
    while viewer.is_running() and not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Camera tracks the robot's position every frame -- only the lookat
        # POINT follows (not distance/azimuth/elevation), so you can still
        # freely rotate/zoom by hand while it keeps the body centered as it
        # walks off, instead of leaving it behind the fixed initial frame.
        viewer.cam.lookat[:] = env.data.qpos[0:3]
        viewer.sync()
        time.sleep(dt)          # play back at roughly real time
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default=None, help="Run directory, e.g. experiments/run2")
    p.add_argument("--gen", type=int, default=None, help="Which generation's champion to watch")
    p.add_argument("--genome", type=str, default=None, help="Explicit genome .pkl (instead of --run/--gen)")
    p.add_argument("--policy", type=str, default=None, help="Explicit policy .zip (instead of --run/--gen)")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--horizon", type=int, default=1000, help="Time cap in env steps.")
    p.add_argument("--loop", action="store_true",
                   help="Replay the episode over and over instead of holding the final frame.")
    args = p.parse_args()

    genome_path, policy_path = _resolve_paths(args)
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    policy = _load_policy(policy_path, args.algo)

    task = TaskConfig(horizon=args.horizon)
    env = MorphologyLocomotionEnv(genome, task)
    dt = env.model.opt.timestep * task.frame_skip

    print(f"Body: {genome.count_limbs()} limbs, {genome.num_actuators()} actuators")
    print(f"No finish line -- run as far as possible, time cap {args.horizon} steps")
    print("Running episode... (close the window to quit)")

    # launch_passive lets us drive the sim ourselves and sync each frame.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while True:
            info = run_episode(env, policy, viewer, dt)
            if not viewer.is_running():
                break

            # Report the outcome — this is the "end" the episode reached.
            if info["steps"] >= args.horizon:
                outcome = f"ran full horizon, x={info['x']:.3f} m"
            else:
                outcome = f"fell at step {info['steps']}, x={info['x']:.3f} m"
            print(f"Episode ended: {outcome}")

            if not args.loop:
                # Hold the final frame so the end state is visible; no Ctrl-C needed.
                print("Holding final frame — close the window to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                break
            time.sleep(1.0)  # brief pause, then replay


if __name__ == "__main__":
    main()