# Brain Tumor completion status

The Brain Tumor runner was prepared and started in the isolated
`results/brain_tumor_completion` directory. No ISIC-2019 training was started.

## Protocol used

- `TRAIN` only was split into a fixed-seed, class-preserving 90/10 train/validation split.
- `TEST` remained untouched until final evaluation.
- Checkpoint selection and early stopping monitor `val_macro_f1`.
- Seed: 42.
- Image size: 224x224, unchanged.
- HybridCNN learning rates and architecture hyperparameters were read from the existing CIFAR-10 optimization summaries.
- Baselines use non-pretrained torchvision models and learning rate 0.001.
- AMP was enabled.
- Micro-batch 8 was first attempted; it completed one epoch but took about 11 minutes 49 seconds.
- Micro-batch 16 was then preflighted successfully and used with gradient accumulation 2 to preserve effective batch size 32. Its first epoch also took about 11 minutes.

## Current result

The six-model suite is **not complete**. The run was stopped before any model
completed training because the measured epoch cost would require many hours
per model. The partial epoch checkpoint and metrics are not final results and
must not be used for comparison.

Observed first-epoch evidence for `HybridCNN_GWO_Run1`:

- Train: 5,040 images
- Validation: 560 images
- Test: 1,600 images
- Batch 8, AMP: validation Macro-F1 0.575151 after approximately 11m49s
- Batch 16, AMP, accumulation 2: validation Macro-F1 0.709297 after approximately 11m00s

These values are diagnostic only, not selected-checkpoint or test results.

No final `result.json`, final test confusion matrix, or completed master result
was produced by this interrupted BT run. Existing result directories under
`results/final_experiments/BrainTumor` remain unchanged.

## Required next decision

To complete all six models, one of the following must be explicitly approved:

1. Keep the current 50-epoch protocol and run for the estimated multi-day wall
   time on this GPU.
2. Pre-register a shorter, identical epoch budget for all six BT models.
3. Move the unchanged protocol to a faster/larger GPU.

No option was applied automatically.
