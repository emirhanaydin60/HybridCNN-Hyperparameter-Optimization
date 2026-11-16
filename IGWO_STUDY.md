# IGWO CIFAR-10 Comparison

This study compares the unchanged Grey Wolf Optimizer (GWO) with a
Global-Best-centered Improved GWO (IGWO). It does not add a new model search
space or use CIFAR-10 test metrics to select candidates.

## Current study plan

The earlier nine-run plan below is superseded for the current execution. The
existing GWO results for seeds 42–44 are the baseline and must not be rerun.
Run only the IGWO seed-42 comparison with
`scripts/run_igwo_existing_gwo_seed.py`; its default output is
`results/igwo_existing_gwo_seed42/`. It validates the saved GWO configuration,
checks software versions, reconstructs the seeded GWO initial population
without fitness evaluations, and compares it with the IGWO run. The script can
also run a single later seed explicitly with `--seed 43` or `--seed 44`; it
does not launch those seeds automatically.

Do not use `scripts/run_igwo_comparison.py` for this plan: that earlier
orchestrator re-runs GWO. Any partial results under
`results/igwo_comparison/` are retained and are not reused or overwritten.

## IGWO procedure

For each iteration, IGWO first applies the standard GWO position update to the
whole population. It then performs a sequential local search around the current
global best:

1. Select one mutable hyperparameter.
2. Copy the current global-best position and change only that coordinate.
   Categorical coordinates move to another allowed category; continuous
   coordinates receive a Gaussian perturbation with standard deviation equal
   to 10% of the coordinate range, then are clipped to their bounds.
3. Evaluate the candidate. A new best becomes the center for the next local
   candidate; otherwise the current best remains the center.
4. Continue until the number of unique local candidate trainings reaches the
   population size. A cache hit retries the same parameter rather than being
   counted as a new candidate. Retry exhaustion and budget-limited incomplete
   iterations are retained in the trace rather than reported as full searches.
5. If local search found a better global best, insert it into the population
   by replacing the current worst wolf.

The local-search candidate is formally

`x' = x_best + delta * e_j`,

where `e_j` is the unit vector for the selected dimension. Categorical
coordinates use a different valid category instead of a numeric perturbation.

## Matched comparison protocol

The source of the experiment settings is
`results/main_experiment/config_used.json`. The runner checks that the CIFAR-10
protocol remains population 8, 15 iterations, 6 search epochs, 50 final
epochs, 45,000 training examples, 5,000 validation examples, and three seeds
42, 43, and 44. GWO and IGWO receive the same seed; the runner compares both
the raw initial positions and their mapped solution signatures.

The study has two comparisons:

- **Equal iterations:** GWO and IGWO each run 8 wolves for 15 iterations.
- **Equal fitness budget:** IGWO runs with a per-seed budget equal to the
  corresponding equal-iteration GWO run's number of cache misses (unique
  candidate model trainings). It has at most 15 iterations. The budget counts
  unique fitness-model trainings, not cached calls or the separate final
  training.

The optimizer fitness is validation accuracy. Candidate fitness uses the
existing six-epoch training path and its existing split/loaders. Final model
training and test evaluation reuse `experiment_runner.py`. Final test
metrics are computed only after the final checkpoint has been selected.

### Existing pipeline behavior retained

Inspection of `train_model` shows that the repository's checkpoint selection,
`ReduceLROnPlateau`, and early stopping monitor validation **accuracy**, not
Macro-F1. This study preserves that existing behavior rather than silently
changing the transfer/training pipeline. Specifically, the scheduler uses
`mode="max"`, factor 0.5, patience 4; early stopping patience is 10. Training
uses Adam and cross-entropy loss. The final-training monitor is a seeded 10%
split of the combined training and validation data. The official CIFAR-10 test
set is not used in search or checkpoint selection.

`set_global_seed` seeds Python, NumPy, PyTorch, and CUDA, and enables cuDNN
deterministic mode while disabling cuDNN benchmarking. Some device/library
operations may still be nondeterministic; software versions and the selected
device are saved with each run.

## Running

From the repository root, run:

```powershell
& 'C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe' scripts\run_igwo_comparison.py
```

The script writes new files under `results/igwo_comparison/` and refuses to
reuse that directory if it exists without a matching study manifest. Completed
runs can be resumed from their saved summaries; a partial run directory without
a summary is preserved and causes a clear error instead of being overwritten.
After all nine runs pass validation, produce the descriptive tables and plots:

```powershell
& 'C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe' scripts\analyze_igwo_comparison.py
```

Each run exports its configuration, summary and metrics JSON, training history,
confusion matrix, IGWO trace where applicable, checkpoint, plots, and training
log. The study exports a protocol manifest, per-run CSV, aggregate analysis
JSON, convergence plot, and test-metric plot. Confusion-matrix shape, test-set
total, all reported test metrics, balanced-accuracy/macro-recall equality,
parameter count, seed, initial population, and evaluation budget are checked.

The analysis reports per-seed paired differences and descriptive means and
sample standard deviations. With three seeds it does not claim statistical
superiority or provide a confirmatory significance conclusion.
