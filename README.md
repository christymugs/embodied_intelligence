# Embodying Intelligence in Real Robots — Learnability-Driven Evolution


This project explores how robot bodies can be evolved not just to perform well, but to **learn well**.

Instead of selecting morphologies only by their final task performance, this system evolves robot bodies based on how efficiently a controller can learn to use them within a fixed lifetime. The goal is to study whether evolutionary pressure on **learnability** can produce morphologies that acquire skills faster.

The project combines:

- evolutionary algorithms for robot body design
- recursive morphology generation
- MuJoCo simulation
- Gymnasium-compatible reinforcement learning environments
- learnability-based fitness metrics
- parallelized evaluation across populations

## Pipeline

```
genome  ──to_robot_tree()──►  RobotTree  ──MujocoBuilder──►  MJCF XML  ──►  MuJoCo
  ▲                                                                            │
  │ mutate()                                                  RL training (inner loop)
  │                                                                            │
  └────────────────── selection on LEARNABILITY ◄───── learnability metric ◄──┘
                              (outer EA loop)
```

The outer loop evolves bodies. For each body, the inner loop trains a controller
from scratch (RL) and measures **how efficiently it learned** — that score, not
final performance, drives selection.

## Layout

```
morphology/
.
├── README.md
├── requirements.txt
├── config
│   └── default.yaml
├── morphology
│   ├── genome.py
│   ├── mujoco_builder.py
│   └── robotic_tree.py
├── evolution
│   ├── evolve.py
│   ├── mutation.py
│   ├── parallel.py
│   └── population.py
├── learning
│   ├── env.py
│   └── train.py
├── fitness
│   └── learnability.py
├── scripts
│   ├── build_test_robot.py
│   ├── check_learning.py
│   ├── plot_run.py
│   ├── run_evolution.py
│   └── visualizer.py
├── tests
│   └── test_pipeline.py
└── experiments
    ├── check
    ├── run1
    ├── run2
    ├── run3
    ├── seed5.xml
    └── smoke
```

## Design space (intentionally open)

- **Recursive growth** — limbs sprout from limbs to arbitrary depth.
- **Heterogeneous limbs** — every limb has its own radius / height / density.
- **Open torso** — the root limb's size evolves like any other.
- **Free attachment** — children attach anywhere along a parent's long axis.

Bounds in `MorphologyBounds` are _simulability_ limits (keep MuJoCo happy), not a
prior on good shapes. Widen them to open the space further.

## Selecting for learnability, not performance

The inner loop trains each body's controller from scratch and records a learning
curve. `fitness/learnability.py` turns that curve into three logged metrics —
**auc** (area under the learning curve; the default selection signal), **final**
(return at the fixed budget, Gupta-style), and **steps_to_threshold**
(convergence speed). Selection uses one; all three are saved for analysis. The
task is forward locomotion, where standing still scores ~0, so there's no cheap
reward to hack — and each body is averaged over `eval_seeds` to denoise the metric.

## Status

End-to-end pipeline working and tested: morphology + MJCF compilation (200/200
random bodies compile), genome + mutation (500/500 mutated bodies compile,
copy-safe), Gymnasium env (passes the API checker), inner RL training with
learning-curve logging, learnability metrics, and the outer EA loop (saves
per-generation best body + history.json). Verified with a 4-body / 2-generation
smoke run.

## Run

```bash
pip install -r requirements.txt
python -m tests.test_pipeline                 # verify the MuJoCo pipeline
python -m scripts.run_evolution --smoke       # full co-evolution, tiny + fast
python -m scripts.run_evolution \             # a real run, parallel over CPUs
    --population 16 --generations 20 --lifetime-steps 100000 \
    --workers 8 --out experiments/run1
```

Evaluations run in parallel across processes (`--workers N`, default = all CPUs;
`--serial` forces one process for clearer tracebacks). Workers are single-threaded
for torch so they don't oversubscribe the CPU — speedup comes from many bodies at
once, roughly linear up to your core count. Cost scales as
population × generations × lifetime_steps × eval_seeds env steps, so a serious run
is still cluster-scale; the per-body independence is what makes it parallelise.

### What gets logged (per run, in `out_dir/`)

- `run.log` — full trace: run config, per-generation summaries, and per-individual
  fitness/metrics (file gets DEBUG detail; console shows INFO summaries + ETA).
- `generation_stats.csv` — one row per generation (best/mean/worst fitness, best
  body size, timing). Plot this for fitness-over-generations.
- `history.json` — rich record incl. each generation's best **learning curve**
  `(timesteps, returns)` — the data for the morphological-Baldwin plot (curves
  getting steeper across generations).
- `gen000_best.xml`, `gen001_best.xml`, … — the best body each generation, ready
  to open in MuJoCo's viewer.
# embodied_intelligence
