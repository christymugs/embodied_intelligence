"""Diagnostic: is the ~90-115 step survival ceiling a bug in our env/reward
code (would affect ANY body), or a body-quality issue (evolution hasn't
found a good one yet)? Build one clean, sensible quadruped by hand -- torso
+ 4 symmetric legs, each a thigh+shin pair -- and test under our exact
current physics/reward/training. If even this collapses at ~90 steps too,
the problem is systemic, not morphological.
"""
import math
import pickle
from morphology.genome import Genome, MorphologyBounds, LimbGene, JointGene
from learning.env import TaskConfig
from learning.train import TrainConfig, train_morphology

bounds = MorphologyBounds()

def leg(azimuth_deg):
    az = math.radians(azimuth_deg)
    return LimbGene(
        radius=0.035, height=0.25, density=700.0,
        attach_frac=0.15, attach_azimuth=az,
        joint=JointGene(axis=[1, 0, 0], range_min=-60.0, range_max=60.0,
                         actuator_type="position", gear=1.2),
        children=[
            LimbGene(
                radius=0.028, height=0.28, density=700.0,
                attach_frac=1.0, attach_azimuth=az,
                joint=JointGene(axis=[0, 1, 0], range_min=-70.0, range_max=70.0,
                                 actuator_type="position", gear=1.2),
                children=[],
            )
        ],
    )

root = LimbGene(radius=0.07, height=0.35, density=700.0, attach_frac=None, joint=None,
                 children=[leg(0), leg(90), leg(180), leg(270)])
genome = Genome(root, bounds)

with open("experiments/handbuilt_quadruped_genome.pkl", "wb") as f:
    pickle.dump(genome, f)

print(f"Body: {genome.count_limbs()} limbs, {genome.num_actuators()} actuators, depth {genome.max_depth()}")

task = TaskConfig()
cfg = TrainConfig(lifetime_steps=500_000, eval_every=10_000)
out = train_morphology(genome, task, cfg, save_path="experiments/handbuilt_quadruped_policy.zip")

print("=== learning curve ===")
for t, r in zip(out["timesteps"], out["returns"]):
    print(f"t={t:>9d}  return={r:.3f}")
