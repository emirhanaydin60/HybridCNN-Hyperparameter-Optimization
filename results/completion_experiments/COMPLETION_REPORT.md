# CNN Experiment Completion Report

## Scope

This report covers the requested comparison of:

- HybridCNN_GWO_Run1
- HybridCNN_GWO_Run3
- HybridCNN_WOA_Run3
- EfficientNet-B0
- MobileNetV3-Large
- ShuffleNetV2-1.5x

across CIFAR-10, ISIC-2019, and Brain Tumor.

No completed experiment was rerun. Existing result directories were not
modified. New files are isolated below `results/completion_experiments`.

## Recovered and verified results

The three CIFAR-10 HybridCNN results were recovered from the legacy
`results/main_experiment` summaries, checkpoints, training curves, and
confusion-matrix images. The final-experiments `result.json` files were not
treated as authoritative because later smoke-test runs replaced some of them.

For CIFAR-10 and ISIC-2019 baseline models, Accuracy, Macro Precision, Macro
Recall, and Macro-F1 were recomputed from the stored confusion matrices. The
`0.0000` macro values in the logs and the null fields in the old master files
were not reused.

| Dataset | Model | Validation Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Status |
|---|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 | HybridCNN_GWO_Run1 | 0.8046 | 0.8793 | 0.87917 | 0.87930 | 0.87881 | Reused |
| CIFAR-10 | HybridCNN_GWO_Run3 | 0.8168 | 0.8722 | 0.87212 | 0.87220 | 0.87193 | Reused |
| CIFAR-10 | HybridCNN_WOA_Run3 | 0.8154 | 0.8768 | 0.87655 | 0.87680 | 0.87659 | Reused |
| CIFAR-10 | EfficientNet-B0 | 0.7374 | 0.7345 | 0.73521 | 0.73450 | 0.73386 | Metrics recomputed |
| CIFAR-10 | MobileNetV3-Large | 0.5570 | 0.5556 | 0.55867 | 0.55560 | 0.55400 | Metrics recomputed |
| CIFAR-10 | ShuffleNetV2-1.5x | 0.7154 | 0.7086 | 0.71097 | 0.70860 | 0.70918 | Metrics recomputed |
| ISIC-2019 | EfficientNet-B0 | 0.69036 | 0.67811 | 0.51061 | 0.40837 | 0.42424 | Metrics recomputed |
| ISIC-2019 | MobileNetV3-Large | 0.68088 | 0.66035 | 0.43856 | 0.40700 | 0.40404 | Metrics recomputed |
| ISIC-2019 | ShuffleNetV2-1.5x | 0.65916 | 0.65700 | 0.44843 | 0.37220 | 0.38324 | Metrics recomputed |

The recomputed accuracy values match the independently logged test accuracies
for the six baseline runs. The macro metrics are now available from the
confusion matrices rather than from the invalid zero/null fields.

## Brain Tumor protocol decision

The existing Brain Tumor loader used the `TEST` directory as validation. Those
results are therefore not independent test results and are not reused in the
comparison.

The required protocol is:

1. Split `TRAIN` into training and validation with a fixed seed and
   class-preserving sampling.
2. Select the best checkpoint using only that validation subset.
3. Evaluate the selected checkpoint once on the untouched `TEST` directory.

No Brain Tumor model was retrained under this corrected protocol during this
step.

## Missing experiments and failure evidence

| Dataset | Model group | Existing evidence | Decision |
|---|---|---|---|
| ISIC-2019 | All three HybridCNNs | CUDA out-of-memory during previous runs; GWO_Run1 also has a partial epoch log/checkpoint | Do not use partial checkpoint; requires a controlled rerun |
| Brain Tumor | All three HybridCNNs | CUDA out-of-memory during previous runs; no completed independent result | Requires a controlled rerun |
| Brain Tumor | Three baselines | Training completed, but TEST was used as validation | Requires retraining under the corrected split |

The isolated completion attempt confirmed that mixed precision and batch size 8
can execute a short HybridCNN forward/backward sequence on the available
12-GB RTX 3060. It did not produce a final experiment result. The attempt was
stopped before completing an epoch after measuring the practical runtime cost.

## Measured cost and estimates

The estimates below are planning estimates, not experimental results:

- ISIC-2019 train set: 17,732 images.
- Brain Tumor TRAIN set: 5,600 images; the corrected 90/10 split is
  approximately 5,040 training and 560 validation images.
- ISIC-2019 HybridCNN_GWO_Run1, batch 8, AMP: 5 batches in 5.54 seconds.
  This projects to roughly 41 minutes per training epoch and approximately
  34 hours for 50 epochs, excluding validation and test overhead.
- Brain Tumor HybridCNN_GWO_Run1, batch 8, AMP: 5 batches in 5.38 seconds.
  This projects to roughly 13 minutes per training epoch and approximately
  10.5 hours for 50 epochs, excluding validation and test overhead.

The other HybridCNN variants should be treated as a broad range rather than
exact forecasts:

| Missing experiment | Approximate 50-epoch cost on current GPU | Confidence |
|---|---:|---|
| ISIC-2019 / HybridCNN_GWO_Run1 | ~34–40 hours | Medium; directly benchmarked architecture family |
| ISIC-2019 / HybridCNN_GWO_Run3 | ~18–28 hours | Low/medium; smaller parameter count but same high-resolution activations |
| ISIC-2019 / HybridCNN_WOA_Run3 | ~34–40 hours | Medium; similar parameter count to GWO_Run1 |
| Brain Tumor / HybridCNN_GWO_Run1 | ~10–13 hours | Medium; directly benchmarked |
| Brain Tumor / HybridCNN_GWO_Run3 | ~6–10 hours | Low/medium |
| Brain Tumor / HybridCNN_WOA_Run3 | ~10–13 hours | Medium |

These estimates explain why the six experiments were not silently changed to
fewer epochs. Reducing epochs, changing image resolution, changing the
architecture, or using a subset would change the scientific comparison and
requires an explicit protocol decision.

## Scientifically valid alternatives

No alternative was applied automatically. The following choices remain
available for a later decision:

1. **Keep the current protocol:** run all missing experiments for the planned
   50 epochs with AMP, batch size 8, and gradient accumulation to effective
   batch size 32. This preserves the architecture and image resolution but
   requires multiple days.
2. **Pre-register a shorter-budget protocol:** use a fixed smaller epoch budget
   for every model and dataset, with the same early-stopping rule. This is
   faster but is a new comparison protocol and should not be mixed with the
   existing 50-epoch results without clearly labeling the protocol.
3. **Use a larger GPU or multi-GPU environment:** preserve the current
   resolution, architecture, and epoch budget while reducing wall-clock time.
4. **Redesign the HybridCNN for high-resolution inputs:** add scientifically
   justified downsampling and rerun all affected models. This is a new model,
   not a drop-in completion of the current experiment.

## Output files

- `master_results.csv`: one row per requested dataset/model pair.
- `master_results.json`: the same rows with structured source/configuration
  fields.
- `final_summary.json`: counts and output-root metadata.
- `preflight.json`: measured safe AMP/batch preflight configuration.
- `COMPLETION_REPORT.md`: this report.

Rows without independently verified results retain null metrics and include an
explicit reason. No partial checkpoint is represented as a successful model.
