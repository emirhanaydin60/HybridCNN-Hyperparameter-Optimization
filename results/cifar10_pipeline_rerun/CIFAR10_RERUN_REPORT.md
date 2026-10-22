# CIFAR-10 HybridCNN Pipeline Rerun

Fresh-scratch reruns of the three selected CIFAR-10 HybridCNN configurations using the shared MedMNIST transfer training/evaluation implementation. These are single runs, not repeated trials for a new statistical comparison.

## Data and protocol

- Official CIFAR-10 train/test splits are used. Each selected run retains its historical random train/validation split seed.
- HybridCNN_GWO_Run1: train/validation/test = 45000/5000/10000; split seed 42.
- HybridCNN_GWO_Run3: train/validation/test = 45000/5000/10000; split seed 44.
- HybridCNN_WOA_Run3: train/validation/test = 45000/5000/10000; split seed 44.
- CIFAR-10 test data is separate from the official training split and is evaluated only after selecting the checkpoint by validation Macro-F1.
- Input: 32x32 RGB; ToTensor only; no normalization, augmentation, or pretrained weights.
- Adam, per-run source learning rate, batch 32, maximum 50 epochs; ReduceLROnPlateau on validation Macro-F1 (factor 0.5, patience 4); early stopping patience 10.
- Determinism: Python, NumPy, PyTorch and CUDA seeds set to 42; cuDNN deterministic enabled and benchmark disabled. CUDA kernels and software/hardware versions are recorded per run.

## Results

| Configuration | Status | Seed | Accuracy | Macro precision | Macro recall | Macro-F1 | Balanced accuracy | Test loss | Best epoch | Trainable params | Train seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HybridCNN_GWO_Run1 | completed | 42 | 0.877900 | 0.879783 | 0.877900 | 0.878330 | 0.877900 | 0.856746 | 44 | 7429418 | 11247.47 |
| HybridCNN_GWO_Run3 | completed | 42 | 0.862400 | 0.865442 | 0.862400 | 0.863361 | 0.862400 | 0.724364 | 28 | 3239242 | 4706.64 |
| HybridCNN_WOA_Run3 | completed | 42 | 0.875300 | 0.876209 | 0.875300 | 0.875422 | 0.875300 | 0.841664 | 46 | 7411658 | 11253.22 |

## Run artifacts

- HybridCNN_GWO_Run1: `results\cifar10_pipeline_rerun\runs\HybridCNN_GWO_Run1` (completed)
- HybridCNN_GWO_Run3: `results\cifar10_pipeline_rerun\runs\HybridCNN_GWO_Run3` (completed)
- HybridCNN_WOA_Run3: `results\cifar10_pipeline_rerun\runs\HybridCNN_WOA_Run3` (completed)

## Comparison with the historical selected-run summaries

| Configuration | Metric | Historical | Rerun | Difference (rerun - historical) |
|---|---|---:|---:|---:|
| HybridCNN_GWO_Run1 | accuracy | 0.879300 | 0.877900 | -0.001400 |
| HybridCNN_GWO_Run1 | macro_precision | 0.879172 | 0.879783 | +0.000611 |
| HybridCNN_GWO_Run1 | macro_recall | 0.879300 | 0.877900 | -0.001400 |
| HybridCNN_GWO_Run1 | macro_f1 | 0.878805 | 0.878330 | -0.000475 |
| HybridCNN_GWO_Run3 | accuracy | 0.872200 | 0.862400 | -0.009800 |
| HybridCNN_GWO_Run3 | macro_precision | 0.872122 | 0.865442 | -0.006680 |
| HybridCNN_GWO_Run3 | macro_recall | 0.872200 | 0.862400 | -0.009800 |
| HybridCNN_GWO_Run3 | macro_f1 | 0.871926 | 0.863361 | -0.008565 |
| HybridCNN_WOA_Run3 | accuracy | 0.876800 | 0.875300 | -0.001500 |
| HybridCNN_WOA_Run3 | macro_precision | 0.876549 | 0.876209 | -0.000341 |
| HybridCNN_WOA_Run3 | macro_recall | 0.876800 | 0.875300 | -0.001500 |
| HybridCNN_WOA_Run3 | macro_f1 | 0.876592 | 0.875422 | -0.001171 |

Historical metrics are not used for selection. Differences can reflect the fresh random initialization and the transfer-pipeline preprocessing policy (ToTensor without the normalization used by the original CIFAR-10 loader). These are not repeated trials for a new statistical comparison.
