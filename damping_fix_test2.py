"""Retest with DAMPING_TORQUE_FRACTION raised from 0.15 to 0.5 -- the first
attempt (0.15) made episode duration/distance measurably worse, likely by
under-damping. Same genome, same 500k budget."""
import pickle
from learning.env import TaskConfig
from learning.train import TrainConfig, train_morphology

with open("experiments/run20/gen003_best_genome.pkl", "rb") as f:
    genome = pickle.load(f)

task = TaskConfig()
cfg = TrainConfig(lifetime_steps=500_000, eval_every=10_000)
out = train_morphology(genome, task, cfg, save_path="experiments/damping_fix_policy2.zip")

print("=== learning curve ===")
for t, r in zip(out["timesteps"], out["returns"]):
    print(f"t={t:>9d}  return={r:.3f}")
