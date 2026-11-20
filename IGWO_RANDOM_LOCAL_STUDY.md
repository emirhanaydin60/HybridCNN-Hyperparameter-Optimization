# IGWO 8x15 Random Local Search

This study is isolated from the active Adaptive IGWO 6x15 run. It writes to
`results/igwo_comparison/paper_config_8x15_random_local/` and refuses to reuse
an existing seed directory or launch claim.

## Configuration

- Population: 8
- Iterations: 15
- Seeds: 42, 43, 44
- Local candidates: 8 unique evaluations after each standard GWO update
- Local search: randomly select one mutable hyperparameter and uniformly
  reinitialize only that coordinate within its existing domain
- Learning rate: uniform sampling in the existing log10 coordinate, equivalent
  to log-uniform sampling in the original learning-rate domain
- Cache: the existing fitness cache keys the complete mapped hyperparameter
  configuration; hits skip CNN training and retry the same selected parameter
- Retry limit: 64 per local candidate; unfilled targets are recorded
- GPU lock: shares
  `results/igwo_comparison/paper_config_6x15_adaptive/gpu_training.lock`

The runner uses the repository's existing CIFAR-10 loaders, model, fitness
calculation, final training, metrics, and output writer. It registers the
random-local optimizer only in its own Python process; no files used by the
currently running Adaptive jobs are changed.

## Safe automatic launch

Start the detached supervisor after the three Adaptive worker PIDs have been
verified:

```powershell
.\scripts\start_igwo_random_local_supervisor.ps1
```

The supervisor checks the expected Adaptive PIDs, each seed's completed
manifest, final summary, metrics, trace, confusion matrix, and checkpoint. A
failed or incomplete run blocks launch and writes `failure_notice.json`. It
does not stop or signal any Adaptive process. Only after all three runs pass
validation and the protected workers exit does it invoke the random-local
launcher. File locks and one-time launch claims prevent duplicate supervisors
or duplicate experiment launches.

Supervisor status and logs are stored at the study root:

- `supervisor.log`
- `supervisor_status.json`
- `supervisor.stdout.log` and `supervisor.stderr.log`
- `launch_claim.json` and `process_launch.json` after launch

The runner creates `seed_42`, `seed_43`, and `seed_44` only after the gate
passes. Each seed has its own study manifest, run summary, training history,
IGWO trace, checkpoint, and training log.
