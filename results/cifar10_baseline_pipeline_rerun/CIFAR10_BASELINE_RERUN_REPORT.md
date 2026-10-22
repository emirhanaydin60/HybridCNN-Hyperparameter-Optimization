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
| efficientnet-b0 | completed | 0.909300 | 0.909186 | 0.909300 | 0.909130 | 0.909300 | 0.587681 | 50 | 4020358 | 10111.17 |
| shufflenetv2-1.5x | completed | 0.871200 | 0.870846 | 0.871200 | 0.870982 | 0.871200 | 0.773130 | 39 | 2488874 | 4651.64 |
| mobilenetv3-large | completed | 0.894300 | 0.894009 | 0.894300 | 0.894059 | 0.894300 | 0.723171 | 50 | 4214842 | 6434.49 |

## Historical comparison

Historical CIFAR-10 scores are calculated from the preserved original confusion matrices. The old CIFAR loader used native 32x32 inputs and normalization; these reruns use the transfer pipeline's 224x224 resize and ToTensor-only policy, so differences are protocol-related as well as reflecting fresh initialization.

| Model | Metric | Historical | Rerun | Difference |
|---|---|---:|---:|---:|
| efficientnet-b0 | accuracy | 0.734500 | 0.909300 | +0.174800 |
| efficientnet-b0 | macro_precision | 0.735208 | 0.909186 | +0.173978 |
| efficientnet-b0 | macro_recall | 0.734500 | 0.909300 | +0.174800 |
| efficientnet-b0 | macro_f1 | 0.733863 | 0.909130 | +0.175267 |
| shufflenetv2-1.5x | accuracy | 0.708600 | 0.871200 | +0.162600 |
| shufflenetv2-1.5x | macro_precision | 0.710967 | 0.870846 | +0.159879 |
| shufflenetv2-1.5x | macro_recall | 0.708600 | 0.871200 | +0.162600 |
| shufflenetv2-1.5x | macro_f1 | 0.709185 | 0.870982 | +0.161798 |
| mobilenetv3-large | accuracy | 0.555600 | 0.894300 | +0.338700 |
| mobilenetv3-large | macro_precision | 0.558675 | 0.894009 | +0.335334 |
| mobilenetv3-large | macro_recall | 0.555600 | 0.894300 | +0.338700 |
| mobilenetv3-large | macro_f1 | 0.554000 | 0.894059 | +0.340060 |

These are single runs, not repeated trials sufficient for a statistical comparison. Historical checkpoints were recorded for provenance only and were not loaded.
