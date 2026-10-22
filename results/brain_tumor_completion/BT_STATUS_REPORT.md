# Brain Tumor Experiment Status Report

## Stop decision

All continuing Brain Tumor training was stopped safely. At the time of this
report there was no active BT training process. The unrelated long-lived Python
process visible in the process list was not a BT runner and was not terminated.

No new training was started for this status-only task. ISIC-2019 training was
not started and no ISIC-2019 files were changed.

The unchanged 50-epoch protocol costs approximately **11 minutes per epoch** on
the available GPU. For six models this is approximately:

```text
11 minutes × 50 epochs × 6 models = 3,300 minutes ≈ 55 hours
```

Additional validation, final TEST evaluation, checkpoint I/O, and possible
early-stopping variation are not included in that estimate.

## Protocol recorded for continuation

- Dataset: Brain Tumor only.
- `TRAIN` split into stratified 90% train and 10% validation.
- Split seed: 42.
- `TEST` excluded from training, checkpoint selection, and early stopping.
- Best checkpoint monitor: validation Macro-F1 (`val_macro_f1`).
- Requested budget: 50 epochs, patience 10.
- Image size: 224x224.
- AMP: enabled.
- Effective batch size: 32.
- Latest safe run setting: micro-batch 16, gradient accumulation 2.
- HybridCNN learning rates and architecture parameters came from the existing
  CIFAR-10 optimization summaries.
- Existing `results/final_experiments/BrainTumor` results were not reused as
  independent test results because they used TEST as validation.

## Per-model status

| Model | Status | Latest completed epoch | Last validation result | Checkpoint | Resume status |
|---|---|---:|---|---|---|
| HybridCNN_GWO_Run1 | incomplete/resource-limited | 1 | Accuracy 0.710714; Macro-F1 0.709297 | `HybridCNN_GWO_Run1/best_model.pth` | Not safely resumable: weights only |
| HybridCNN_GWO_Run3 | incomplete/resource-limited | 0 | Not available | None | Not resumable |
| HybridCNN_WOA_Run3 | incomplete/resource-limited | 0 | Not available | None | Not resumable |
| EfficientNet-B0 | incomplete/resource-limited | 0 | Not available in new protocol | None | Not resumable |
| MobileNetV3-Large | incomplete/resource-limited | 0 | Not available in new protocol | None | Not resumable |
| ShuffleNetV2-1.5x | incomplete/resource-limited | 0 | Not available in new protocol | None | Not resumable |

The GWO_Run1 checkpoint contains model weights only. Optimizer state,
scheduler state, scaler state, and an exact continuation cursor were not saved.
Therefore it must not be described as a resumable full-training checkpoint.
It is retained as an explicitly partial artifact and is not a final result.

## Preserved files and separation

Previous BT artifacts remain under:

- `results/final_experiments/BrainTumor`

The new status and partial-run records are under:

- `results/brain_tumor_completion`

No previous result, checkpoint, or experiment directory was deleted or
overwritten. The new status files do not merge the old invalid TEST-as-
validation results into new independent results.

## Remaining work

1. Select an approved execution option: preserve the 50-epoch protocol on a
   long-running GPU job, move it to a faster/larger GPU, or approve a new
   identical reduced-budget protocol for all six models.
2. Start each model from a fresh run or a deliberately documented weights-only
   warm start; do not claim exact optimizer-state continuation from the
   retained GWO_Run1 file.
3. Save optimizer, scheduler, AMP scaler, model, and epoch state if resumability
   is required.
4. Select checkpoints using validation Macro-F1 only.
5. Evaluate the untouched TEST folder once with the selected checkpoint.
6. Generate final confusion matrices and test Accuracy, Macro Precision, Macro
   Recall, and Macro-F1 only after all six runs complete.

## Machine-readable record

The authoritative machine-readable status is:

- `results/brain_tumor_completion/bt_status.json`

All six entries are `incomplete/resource-limited`; none is marked completed.
