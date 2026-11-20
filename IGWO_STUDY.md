# IGWO CIFAR-10 Paper-Parameter Study

## Current protocol

The existing CIFAR-10 GWO runs in `results/main_experiment/GWO/run_01` through
`run_03` are the reference; they must never be rerun or overwritten. The saved
configuration is `results/main_experiment/config_used.json`: GWO population 8,
15 iterations, 6 search epochs, 50 final epochs, and seeds 42, 43, 44. Its
actual unique candidate-training counts are 124, 123, and 124 (with 4, 5, and
4 cache hits).

Earlier IGWO results are preserved in
`results/igwo_comparison/paper_config_6x10/` and
`results/igwo_comparison/paper_config_4x15/`. The current IGWO study uses
population 6, 15 iterations, and 2 Adaptive Guided Local Search candidates
per iteration. Each seed's
unique-training budget is capped at the corresponding existing GWO run's
cache-miss count. Since IGWO has at most 6 initial evaluations plus
`15 * (6 GWO candidates + 2 local candidates)`, this budget includes startup
and cache effects rather than assuming equal theoretical iteration counts.
When the cap is reached, the last local-search batch may be incomplete; this is
recorded and no extra training is allowed. Cache hits skip CNN training and
retry the same hyperparameter.

The launch script starts three independent Windows processes, one per seed.
They share an exclusive file lock so a single RTX 3060 trains one model at a
time instead of risking concurrent GPU out-of-memory failures. Waiting
processes and active training processes each have their own run log. The lock
is released even if a run errors, and the processes are started separately
from the Copilot session.

The archived GWO summaries do not contain raw initial-population positions.
The runner checks the archived config and software versions, reconstructs the
GWO initial population deterministically from the matching seed and search
bounds, then verifies that IGWO starts from the first six corresponding
positions. This reconstruction performs no fitness evaluations and launches
no GWO training.

## Adaptive Guided Local Search

Each iteration first performs the standard GWO position update for all six
wolves, followed by two unique local evaluations around the current Global Best.
For each candidate:

1. Select one mutable hyperparameter.
2. Copy the current Global Best and modify only that coordinate. A categorical
   coordinate moves to another allowed category; integer coordinates use a
   valid integer neighbor. Continuous coordinates use a bounded adaptive step
   radius: it starts at 10% of the coordinate range, halves after an
   improvement, and doubles otherwise, within 1% and 50% of the range. The
   actual displacement is sampled within that radius. Learning rate is
   searched in its existing log10 coordinate.
3. If the candidate improves fitness, immediately make it the Global Best and
   use it as the center for the next local candidate.
4. Count only unique candidate trainings toward the two-candidate local
   target. Cache hits and duplicate solutions are retried with the same
   parameter selected, without consuming training budget. A retry limit bounds
   attempts; any unfilled target is recorded.
5. Insert an improved Global Best into the population by replacing its worst
   wolf.

Each trace entry records the parameter, old/new values, actual step size and
adaptive step radius before/after, fitness, improvement, cache status, and
candidate positions. Categorical parameters have no numeric step size, which
is recorded as `null`.

## Training and evaluation behavior

The fitness metric remains validation accuracy. Candidate training uses six
epochs and final training uses 50 epochs with the existing CIFAR-10 splits,
HybridCNN architecture, optimizer, scheduler, and search space. Test metrics
are calculated only after optimization and final checkpoint selection.

The current training implementation selects checkpoints, schedules learning
rate, and early-stops on validation accuracy, not Macro-F1. This existing
behavior is preserved. `ReduceLROnPlateau` uses `mode="max"`, factor 0.5,
patience 4; early-stopping patience is 10. Training uses Adam and
cross-entropy loss. The final-training monitor is a seeded 10% split of the
combined training and validation examples.

The existing CIFAR-10 GWO results contain test Accuracy, Precision, Recall,
and F1, validation Accuracy, convergence history, unique-training/cache
counts, runtime, best configuration, and the exact protocol config. They do
not contain historical validation Macro-F1, so that metric is not imputed.

## Running and outputs

From the repository root, launch the three runs:

```powershell
& .\scripts\launch_igwo_paper_experiments.ps1
```

By default, the launcher uses
`C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe`; supply `-PythonPath` to
override it. The launcher checks the archived GWO config/results and refuses
to overwrite any target seed directory. Per-seed files go under
`results/igwo_comparison/paper_config_6x15_adaptive/seed_42`, `seed_43`, and
`seed_44`.
The parent contains `process_launch.json`, per-seed stdout/stderr logs, and a
shared GPU-lock file. Each run contains its config, metrics, history,
confusion matrix, IGWO trace, checkpoint, plots, and `training.log`.

The processes can also be started individually with
`scripts/run_igwo_existing_gwo_seed.py --seed 42|43|44`; the default is seed
42. The script does not launch other seeds automatically.

After all three summaries report `completed`, produce aggregate per-seed and
three-seed mean/sample-standard-deviation/best/worst reports with:

```powershell
& 'C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe' scripts\analyze_igwo_paper_comparison.py
```

The report includes validation/test metrics, convergence, unique training
counts, cache hits, runtime, and local-search Global-Best improvements. Three
seeds are exploratory evidence and do not support a claim of statistical
superiority.
