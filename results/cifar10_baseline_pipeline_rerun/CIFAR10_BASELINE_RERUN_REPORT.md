# CIFAR-10 Baseline Pipeline Rerun

Fresh-scratch reruns of EfficientNet-B0, ShuffleNetV2-1.5x, and MobileNetV3-Large using the shared MedMNIST transfer trainer.

## Protocol

- Official CIFAR-10 train/test splits; train/validation split is 45,000/5,000 with seed 42, and the official 10,000-image test split is kept separate.
- Baseline input follows the shared transfer pipeline at 224x224 RGB, bilinear resize, then ToTensor only. No normalization, augmentation, or pretrained weights.
- Adam, learning rate 0.001, batch 32, maximum 50 epochs; checkpoint and early stopping use validation Macro-F1. ReduceLROnPlateau monitors validation Macro-F1 (factor 0.5, patience 4); early stopping patience 10.
- Seed 42 is set for Python, NumPy, PyTorch, and CUDA when available. cuDNN deterministic mode is enabled and benchmark mode disabled.

## Results

| Model | Status | Accuracy | Macro precision | Macro recall | Macro-F1 | Balanced accuracy | Test loss | Best epoch | Parameters | Train seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| efficientnet-b0 | planned | — | — | — | — | — | — | — | — | — |
| shufflenetv2-1.5x | planned | — | — | — | — | — | — | — | — | — |
| mobilenetv3-large | planned | — | — | — | — | — | — | — | — | — |

## Historical comparison

Historical CIFAR-10 scores are calculated from the preserved original confusion matrices. The old CIFAR loader used native 32x32 inputs and normalization; these reruns use the transfer pipeline's 224x224 resize and ToTensor-only policy, so differences are protocol-related as well as reflecting fresh initialization.

| Model | Metric | Historical | Rerun | Difference |
|---|---|---:|---:|---:|
| efficientnet-b0 | accuracy | 0.734500 | — | — |
| efficientnet-b0 | macro_precision | 0.735208 | — | — |
| efficientnet-b0 | macro_recall | 0.734500 | — | — |
| efficientnet-b0 | macro_f1 | 0.733863 | — | — |
| shufflenetv2-1.5x | accuracy | 0.708600 | — | — |
| shufflenetv2-1.5x | macro_precision | 0.710967 | — | — |
| shufflenetv2-1.5x | macro_recall | 0.708600 | — | — |
| shufflenetv2-1.5x | macro_f1 | 0.709185 | — | — |
| mobilenetv3-large | accuracy | 0.555600 | — | — |
| mobilenetv3-large | macro_precision | 0.558675 | — | — |
| mobilenetv3-large | macro_recall | 0.555600 | — | — |
| mobilenetv3-large | macro_f1 | 0.554000 | — | — |

These are single runs, not repeated trials sufficient for a statistical comparison. Historical checkpoints were recorded for provenance only and were not loaded.
