"""One-off test: does more inner-loop training turn run20's champion's
oscillation into sustained walking? Retrains its controller alone (not a
full evolutionary run) for 2M steps instead of 500k, and prints the curve."""
import pickle
from learning.env import TaskConfig
from learning.train import TrainConfig, train_morphology

with open("experiments/run20/gen003_best_genome.pkl", "rb") as f:
    genome = pickle.load(f)

task = TaskConfig()
cfg = TrainConfig(lifetime_steps=2_000_000, eval_every=20_000)
out = train_morphology(genome, task, cfg, save_path="experiments/run20_extended_policy.zip")

print("=== learning curve ===")
for t, r in zip(out["timesteps"], out["returns"]):
    print(f"t={t:>9d}  return={r:.3f}")
