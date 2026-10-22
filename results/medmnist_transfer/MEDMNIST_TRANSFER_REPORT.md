# MedMNIST Transfer Evaluation

## Research objective

Evaluate whether three CIFAR-10-optimized HybridCNN configurations transfer to DermaMNIST and OrganSMNIST, alongside three randomly initialized torchvision CNN baselines. This is not evidence of state-of-the-art superiority.

## Datasets and official splits

| Dataset | MedMNIST version | Split counts (train/val/test) | Classes | Channels |
|---|---|---:|---:|---:|
| DermaMNIST | 3.0.2 | 7007/1003/2005 | 7 | 3 |
| OrganSMNIST | 3.0.2 | 13932/2452/8827 | 11 | 1 |

## Models, input sizes, and preprocessing

| Model | Input | Training configuration |
|---|---:|---|
| HybridCNN_GWO_Run1 | 32x32x3 | Adam, lr=0.0012351628680148583, batch=32, no augmentation, from scratch |
| HybridCNN_GWO_Run3 | 32x32x3 | Adam, lr=0.0014598678014005684, batch=32, no augmentation, from scratch |
| HybridCNN_WOA_Run3 | 32x32x3 | Adam, lr=0.0014188144847256917, batch=32, no augmentation, from scratch |
| efficientnet-b0 | 224x224x3 | Adam, lr=0.001, batch=32, no augmentation, from scratch |
| mobilenetv3-large | 224x224x3 | Adam, lr=0.001, batch=32, no augmentation, from scratch |
| shufflenetv2-1.5x | 224x224x3 | Adam, lr=0.001, batch=32, no augmentation, from scratch |

The three torchvision models use their default torchvision architectures and no pretrained weights. All models use `ToTensor` with no normalization. OrganSMNIST is converted to three channels before resizing. HybridCNN inputs are resized to 32x32; baseline inputs are resized to 224x224. This resolution difference is intentional and does not create additional image information.

HybridCNN architectural hyperparameters, learning rate, and batch size come from the indicated CIFAR-10 run summaries. Source checkpoints are hashed for provenance but are not loaded. The baseline settings are Adam, lr=0.001, batch size 32, weight decay 0, and no augmentation.

## Training protocol

Maximum 50 epochs; validation macro-F1 selects checkpoints, drives ReduceLROnPlateau (factor 0.5, patience 4), and controls early stopping (patience 10). Test data is evaluated only after loading the best validation checkpoint. No AMP or gradient accumulation is used.

## Test results

| Dataset | Model | Status | Accuracy | Macro precision | Macro recall | Macro-F1 | Balanced accuracy | Loss | Best epoch | Params | Train seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DermaMNIST | HybridCNN_GWO_Run1 | completed | 0.7521 | 0.5437 | 0.5356 | 0.5303 | 0.5356 | 0.6862 | 35 | 7428647 | 1760.7 |
| DermaMNIST | HybridCNN_GWO_Run3 | completed | 0.7666 | 0.5884 | 0.5040 | 0.5327 | 0.5040 | 0.6310 | 34 | 3238471 | 918.5 |
| DermaMNIST | HybridCNN_WOA_Run3 | completed | 0.7546 | 0.6160 | 0.5423 | 0.5491 | 0.5423 | 0.7150 | 43 | 7411271 | 1840.1 |
| DermaMNIST | efficientnet-b0 | completed | 0.7636 | 0.5429 | 0.5177 | 0.5243 | 0.5177 | 0.9061 | 28 | 4016515 | 1125.9 |
| DermaMNIST | mobilenetv3-large | completed | 0.6953 | 0.4337 | 0.4288 | 0.4190 | 0.4288 | 0.7976 | 18 | 4210999 | 549.1 |
| DermaMNIST | shufflenetv2-1.5x | completed | 0.7312 | 0.5129 | 0.4484 | 0.4608 | 0.4484 | 0.7620 | 11 | 2485799 | 321.0 |
| OrganSMNIST | HybridCNN_GWO_Run1 | completed | 0.8091 | 0.7728 | 0.7702 | 0.7706 | 0.7702 | 1.1306 | 49 | 7429675 | 3747.8 |
| OrganSMNIST | HybridCNN_GWO_Run3 | completed | 0.8011 | 0.7647 | 0.7581 | 0.7606 | 0.7581 | 1.1070 | 37 | 3239499 | 2079.1 |
| OrganSMNIST | HybridCNN_WOA_Run3 | completed | 0.8106 | 0.7814 | 0.7745 | 0.7770 | 0.7745 | 0.9892 | 32 | 7411787 | 3203.2 |
| OrganSMNIST | efficientnet-b0 | completed | 0.8190 | 0.7866 | 0.7777 | 0.7797 | 0.7777 | 1.0660 | 32 | 4021639 | 2771.8 |
| OrganSMNIST | mobilenetv3-large | completed | 0.8151 | 0.7745 | 0.7725 | 0.7731 | 0.7725 | 1.1907 | 43 | 4216123 | 2232.0 |
| OrganSMNIST | shufflenetv2-1.5x | completed | 0.7902 | 0.7552 | 0.7506 | 0.7509 | 0.7506 | 0.8483 | 12 | 2489899 | 713.4 |

## Confusion matrices and run status

- DermaMNIST / HybridCNN_GWO_Run1: `results\medmnist_transfer\DermaMNIST\HybridCNN_GWO_Run1\attempt_003\confusion_matrix.json` (completed)
- DermaMNIST / HybridCNN_GWO_Run3: `results\medmnist_transfer\DermaMNIST\HybridCNN_GWO_Run3\attempt_003\confusion_matrix.json` (completed)
- DermaMNIST / HybridCNN_WOA_Run3: `results\medmnist_transfer\DermaMNIST\HybridCNN_WOA_Run3\attempt_003\confusion_matrix.json` (completed)
- DermaMNIST / efficientnet-b0: `results\medmnist_transfer\DermaMNIST\efficientnet-b0\attempt_003\confusion_matrix.json` (completed)
- DermaMNIST / mobilenetv3-large: `results\medmnist_transfer\DermaMNIST\mobilenetv3-large\attempt_003\confusion_matrix.json` (completed)
- DermaMNIST / shufflenetv2-1.5x: `results\medmnist_transfer\DermaMNIST\shufflenetv2-1.5x\attempt_003\confusion_matrix.json` (completed)
- OrganSMNIST / HybridCNN_GWO_Run1: `results\medmnist_transfer\OrganSMNIST\HybridCNN_GWO_Run1\attempt_003\confusion_matrix.json` (completed)
- OrganSMNIST / HybridCNN_GWO_Run3: `results\medmnist_transfer\OrganSMNIST\HybridCNN_GWO_Run3\attempt_003\confusion_matrix.json` (completed)
- OrganSMNIST / HybridCNN_WOA_Run3: `results\medmnist_transfer\OrganSMNIST\HybridCNN_WOA_Run3\attempt_003\confusion_matrix.json` (completed)
- OrganSMNIST / efficientnet-b0: `results\medmnist_transfer\OrganSMNIST\efficientnet-b0\attempt_004\confusion_matrix.json` (completed)
- OrganSMNIST / mobilenetv3-large: `results\medmnist_transfer\OrganSMNIST\mobilenetv3-large\attempt_004\confusion_matrix.json` (completed)
- OrganSMNIST / shufflenetv2-1.5x: `results\medmnist_transfer\OrganSMNIST\shufflenetv2-1.5x\attempt_004\confusion_matrix.json` (completed)

## Interpretation and limitations

This study assesses transfer of CIFAR-10-selected HybridCNN hyperparameters to two medical image tasks. The different input resolutions, small 28x28 source images, and training-from-scratch baseline protocol limit direct architectural comparisons. Results must not be presented as proof of general or state-of-the-art superiority.
