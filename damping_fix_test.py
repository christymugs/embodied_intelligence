"""Does fixing the flat joint damping bug let this body sustain locomotion
longer? Same genome as experiments/run20 gen3 champion, same 500k training
budget as a normal run -- only the damping formula changed."""
import pickle
from learning.env import TaskConfig
from learning.train import TrainConfig, train_morphology

with open("experiments/run20/gen003_best_genome.pkl", "rb") as f:
    genome = pickle.load(f)

task = TaskConfig()
cfg = TrainConfig(lifetime_steps=500_000, eval_every=10_000)
out = train_morphology(genome, task, cfg, save_path="experiments/damping_fix_policy.zip")

print("=== learning curve ===")
for t, r in zip(out["timesteps"], out["returns"]):
    print(f"t={t:>9d}  return={r:.3f}")
