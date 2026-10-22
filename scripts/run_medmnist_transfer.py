"""Run the MedMNIST transfer comparison without modifying the CIFAR-10 runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import HybridCNN
from model_factory import create_model
from utils import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "medmnist_transfer"
DATA_ROOT = PROJECT_ROOT / "data" / "medmnist"
MODEL_NAMES = (
    "HybridCNN_GWO_Run1",
    "HybridCNN_GWO_Run3",
    "HybridCNN_WOA_Run3",
    "efficientnet-b0",
    "mobilenetv3-large",
    "shufflenetv2-1.5x",
)
DATASET_NAMES = ("DermaMNIST", "OrganSMNIST")
SOURCE_RUNS = {
    "HybridCNN_GWO_Run1": ("GWO", "run_01"),
    "HybridCNN_GWO_Run3": ("GWO", "run_03"),
    "HybridCNN_WOA_Run3": ("WOA", "run_03"),
}
BASELINE_SETTINGS = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": 32,
    "augmentation": "none",
}
TRAINING_SETTINGS = {
    "max_epochs": 50,
    "selection_metric": "validation_macro_f1",
    "early_stopping_patience": 10,
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "monitor": "validation_macro_f1",
        "mode": "max",
        "factor": 0.5,
        "patience": 4,
    },
    "seed": 42,
    "mixed_precision": False,
    "gradient_accumulation_steps": 1,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_configurations() -> dict[str, dict[str, Any]]:
    configurations = {}
    for model_name, (algorithm, run_name) in SOURCE_RUNS.items():
        run_dir = RESULTS_ROOT.parent / "main_experiment" / algorithm / run_name
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "best_model.pth"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing source summary or checkpoint for {model_name}: {run_dir}"
            )

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("algorithm", "").lower() != algorithm.lower():
            raise ValueError(f"Unexpected optimizer in {summary_path}")
        if summary.get("dataset", "").lower() != "cifar10":
            raise ValueError(f"Unexpected source dataset in {summary_path}")
        hyperparameters = summary.get("best_hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise ValueError(f"Missing best_hyperparameters in {summary_path}")
        required = (
            "shared_conv_kernel_size",
            "base_filters",
            "dilation",
            "final_neurons",
            "dropout",
            "learning_rate",
            "batch_size",
            "se_ratio",
        )
        missing = [key for key in required if key not in hyperparameters]
        if missing:
            raise ValueError(f"Missing {missing} in {summary_path}")

        source_model = HybridCNN(
            in_channels=3,
            img_size=32,
            shared_conv_kernel_size=hyperparameters["shared_conv_kernel_size"],
            base_filters=hyperparameters["base_filters"],
            dilation=hyperparameters["dilation"],
            final_neurons=hyperparameters["final_neurons"],
            dropout=hyperparameters["dropout"],
            se_ratio=hyperparameters["se_ratio"],
            num_classes=10,
        )
        try:
            checkpoint_state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        expected_state = source_model.state_dict()
        if set(checkpoint_state) != set(expected_state) or any(
            checkpoint_state[name].shape != expected_state[name].shape
            for name in expected_state
        ):
            raise ValueError(
                f"Source checkpoint tensors do not match {model_name} hyperparameters"
            )
        configurations[model_name] = {
            "hyperparameters": hyperparameters,
            "source_summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "source_summary_sha256": file_sha256(summary_path),
            "source_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "source_checkpoint_sha256": file_sha256(checkpoint_path),
            "source_checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "source_run_seed": summary.get("run_seed"),
            "source_parameter_count": sum(
                parameter.numel() for parameter in source_model.parameters()
            ),
            "historical_test_metrics": {
                "accuracy": summary.get("best_test_accuracy"),
                "macro_precision": summary.get("precision"),
                "macro_recall": summary.get("recall"),
                "macro_f1": summary.get("f1"),
                "best_validation_accuracy": summary.get("best_validation_accuracy"),
            },
            "checkpoint_tensor_shapes_match": True,
            "source_weights_loaded": False,
        }
    return configurations


def class_names_from_info(info: dict[str, Any], num_classes: int) -> list[str]:
    labels = info.get("label", {})
    names = [str(labels.get(str(index), index)) for index in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(names)}")
    return names


def make_transform(dataset_name: str, resolution: int):
    operations = []
    if dataset_name == "OrganSMNIST":
        operations.append(transforms.Grayscale(num_output_channels=3))
    elif dataset_name != "DermaMNIST":
        raise ValueError(f"Unsupported MedMNIST dataset: {dataset_name}")
    operations.extend(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    return transforms.Compose(operations)


def load_medmnist_splits(dataset_name: str, resolution: int):
    import medmnist

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_key = dataset_name.lower()
    info = medmnist.INFO[dataset_key]
    dataset_class = getattr(medmnist, info["python_class"])
    as_rgb = dataset_name == "DermaMNIST"
    transform = make_transform(dataset_name, resolution)
    datasets = {
        split: dataset_class(
            root=str(DATA_ROOT),
            split=split,
            download=True,
            as_rgb=as_rgb,
            transform=transform,
        )
        for split in ("train", "val", "test")
    }
    class_indices = sorted(int(index) for index in info["label"])
    if class_indices != list(range(len(class_indices))):
        raise ValueError(f"Unexpected class label mapping for {dataset_name}")
    num_classes = len(class_indices)
    count_mismatches = {}
    for split_name, dataset in datasets.items():
        actual = len(dataset)
        expected = int(info["n_samples"][split_name])
        if actual != expected:
            count_mismatches[split_name] = {"expected": expected, "actual": actual}
        labels = np.asarray(dataset.labels).reshape(-1)
        if len(labels) != actual:
            raise ValueError(f"Label count mismatch in {dataset_name} {split_name}")
        if len(labels) == 0 or labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"Invalid class index in {dataset_name} {split_name}")

    return datasets, info, num_classes, count_mismatches


def create_experiment_model(model_name: str, num_classes: int, source: dict[str, Any] | None):
    if source is None:
        return create_model(model_name, num_classes=num_classes, device="cpu", pretrained=False)

    hp = source["hyperparameters"]
    model = HybridCNN(
        in_channels=3,
        img_size=32,
        shared_conv_kernel_size=hp["shared_conv_kernel_size"],
        base_filters=hp["base_filters"],
        dilation=hp["dilation"],
        final_neurons=hp["final_neurons"],
        dropout=hp["dropout"],
        se_ratio=hp["se_ratio"],
        num_classes=num_classes,
    )
    return model


def experiment_training_config(source: dict[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return copy.deepcopy(BASELINE_SETTINGS)
    hp = source["hyperparameters"]
    return {
        "optimizer": "Adam",
        "learning_rate": float(hp["learning_rate"]),
        "weight_decay": 0.0,
        "batch_size": int(hp["batch_size"]),
        "augmentation": "none",
        "hyperparameters_source": source["source_summary"],
    }


def experiment_resolution(model_name: str) -> int:
    return 32 if model_name in SOURCE_RUNS else 224


def make_loaders(
    datasets: dict[str, Any],
    seed: int,
    source: dict[str, Any] | None,
):
    batch_size = experiment_training_config(source)["batch_size"]
    loaders = {}
    for offset, split_name in enumerate(("train", "val", "test")):
        loaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(seed + offset),
        )
    return loaders


def create_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"medmnist.{log_path.parent.name}.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    actual_count = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros_like(true_positive),
        where=actual_count != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    total = int(confusion.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
    }


def evaluate(model, loader, device, num_classes: int) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.reshape(-1).long().to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            count += batch_size
            indices = labels * num_classes + predictions
            confusion += (
                torch.bincount(indices, minlength=num_classes * num_classes)
                .reshape(num_classes, num_classes)
                .cpu()
                .numpy()
            )
    results = confusion_metrics(confusion)
    results["loss"] = total_loss / count if count else 0.0
    results["confusion_matrix"] = confusion.tolist()
    return results


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def experiment_configs(source_configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    configs = {}
    for dataset_name in DATASET_NAMES:
        for model_name in MODEL_NAMES:
            source = source_configs.get(model_name)
            resolution = experiment_resolution(model_name)
            configs[f"{dataset_name}/{model_name}"] = {
                "experiment_id": f"{dataset_name}-{model_name}",
                "dataset": dataset_name,
                "model": model_name,
                "num_classes": 7 if dataset_name == "DermaMNIST" else 11,
                "input_size": [3, resolution, resolution],
                "resize": {
                    "size": [resolution, resolution],
                    "interpolation": "bilinear",
                },
                "grayscale_to_rgb": dataset_name == "OrganSMNIST",
                "normalization": "none; ToTensor scales uint8 [0,255] to float32 [0,1]",
                "training": experiment_training_config(source),
                "protocol": copy.deepcopy(TRAINING_SETTINGS),
                "source_run": source,
                "pretrained": False,
            }
    return configs


def make_attempt_dirs(configs: dict[str, dict[str, Any]]) -> dict[str, Path]:
    attempts = {}
    for key, config in configs.items():
        model_dir = RESULTS_ROOT / config["dataset"] / config["model"]
        model_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(model_dir.glob("attempt_*"))
        if existing:
            latest_status = None
            latest_state = existing[-1] / "experiment.json"
            if latest_state.is_file():
                with latest_state.open("r", encoding="utf-8") as handle:
                    previous_state = json.load(handle)
                latest_status = previous_state.get("status")
            if latest_status == "completed":
                attempts[key] = existing[-1]
                continue
            if latest_status in ("planned", "running"):
                previous_state.update(
                    {
                        "status": "incomplete",
                        "reason": "A previous invocation ended before completion.",
                    }
                )
                write_json(latest_state, previous_state)
        attempt = model_dir / f"attempt_{len(existing) + 1:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        save_yaml(attempt / "config.yaml", config)
        write_json(
            attempt / "experiment.json",
            {"experiment_id": config["experiment_id"], "status": "planned"},
        )
        attempts[key] = attempt
    return attempts


def write_results(results_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for entry in entries.values():
        summary = {
            "experiment_id": entry["config"]["experiment_id"],
            "dataset": entry["config"]["dataset"],
            "model": entry["config"]["model"],
            "status": entry["status"],
            "experiment_dir": str(entry["dir"].relative_to(PROJECT_ROOT)),
        }
        if "metrics" in entry:
            summary.update(entry["metrics"])
        rows.append(summary)
    write_json(results_path, {"experiments": rows})


def smoke_test(
    dataset_name: str,
    model_name: str,
    datasets: dict[str, Any],
    num_classes: int,
    source: dict[str, Any] | None,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    set_global_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = create_experiment_model(model_name, num_classes, source).to(device)
    loaders = make_loaders(datasets, seed, source)
    images, labels = next(iter(loaders["train"]))
    labels = labels.reshape(-1).long()
    if images.ndim != 4 or list(images.shape[1:]) != [
        3,
        experiment_resolution(model_name),
        experiment_resolution(model_name),
    ]:
        raise ValueError(f"Unexpected input tensor shape: {list(images.shape)}")
    if images.dtype != torch.float32 or not torch.isfinite(images).all():
        raise ValueError(f"Invalid input tensor dtype or values: {images.dtype}")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise ValueError("Tensor values are outside the configured [0,1] range")
    if labels.dtype != torch.int64 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("Invalid label dtype or class index")

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment_training_config(source)["learning_rate"],
        weight_decay=0.0,
    )
    optimizer.zero_grad()
    logits = model(images.to(device))
    loss = nn.CrossEntropyLoss()(logits, labels.to(device))
    if tuple(logits.shape) != (len(labels), num_classes):
        raise ValueError(f"Unexpected model output shape: {list(logits.shape)}")
    loss.backward()
    optimizer.step()

    memory = {
        "gpu_allocated_bytes": (
            int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    result = {
        "status": "passed",
        "dataset": dataset_name,
        "model": model_name,
        "input_batch_shape": list(images.shape),
        "input_dtype": str(images.dtype),
        "input_min": float(images.min()),
        "input_max": float(images.max()),
        "label_dtype": str(labels.dtype),
        "label_indices_in_batch": sorted(set(labels.tolist())),
        "forward_output_shape": list(logits.shape),
        "smoke_loss": float(loss.item()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **memory,
    }
    del model, loaders, images, labels, logits, loss, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def train_experiment(
    config: dict[str, Any],
    experiment_dir: Path,
    datasets: dict[str, Any],
    num_classes: int,
    source: dict[str, Any] | None,
    device: torch.device,
    epochs: int,
) -> dict[str, Any]:
    logger = create_logger(experiment_dir / "run.log")
    training_config = config["training"]
    seed = int(config["protocol"]["seed"])
    set_global_seed(seed)
    loaders = make_loaders(datasets, seed, source)
    model = create_experiment_model(config["model"], num_classes, source).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = {"epochs": []}
    checkpoint_path = experiment_dir / "best_checkpoint.pth"
    started = time.perf_counter()
    state = {"experiment_id": config["experiment_id"], "status": "running"}
    write_json(experiment_dir / "experiment.json", state)

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.reshape(-1).long().to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_count = labels.size(0)
            train_loss_sum += float(loss.item()) * batch_count
            train_count += batch_count
            indices = labels * num_classes + logits.detach().argmax(dim=1)
            train_confusion += (
                torch.bincount(indices, minlength=num_classes * num_classes)
                .reshape(num_classes, num_classes)
                .cpu()
                .numpy()
            )

        train_metrics = confusion_metrics(train_confusion)
        validation = evaluate(model, loaders["val"], device, num_classes)
        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_precision": validation["macro_precision"],
            "validation_macro_recall": validation["macro_recall"],
            "validation_macro_f1": validation["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history["epochs"].append(record)
        logger.info(
            "epoch=%d train_loss=%.6f train_acc=%.6f train_macro_f1=%.6f "
            "val_loss=%.6f val_acc=%.6f val_macro_f1=%.6f",
            epoch,
            record["train_loss"],
            record["train_accuracy"],
            record["train_macro_f1"],
            record["validation_loss"],
            record["validation_accuracy"],
            record["validation_macro_f1"],
        )
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step(validation["macro_f1"])
        write_json(experiment_dir / "history.json", history)
        if stale_epochs >= int(config["protocol"]["early_stopping_patience"]):
            logger.info("early_stopping epoch=%d", epoch)
            break

    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError("Training did not produce a validation-selected checkpoint")
    training_seconds = time.perf_counter() - started
    test_evaluation_started = time.perf_counter()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    test_metrics = evaluate(model, loaders["test"], device, num_classes)
    test_evaluation_seconds = time.perf_counter() - test_evaluation_started
    confusion = np.asarray(test_metrics.pop("confusion_matrix"), dtype=np.int64)
    if int(confusion.sum()) != len(datasets["test"]):
        raise ValueError("Confusion matrix total does not match the official test split")
    if not np.isclose(
        confusion_metrics(confusion)["accuracy"], test_metrics["accuracy"], atol=1e-12
    ):
        raise ValueError("Confusion matrix and test accuracy do not match")

    epoch_seconds = [row["epoch_seconds"] for row in history["epochs"]]
    metrics = {
        **test_metrics,
        "best_validation_macro_f1": float(best_f1),
        "best_epoch": best_epoch,
        "epochs_completed": len(history["epochs"]),
        "total_training_seconds": training_seconds,
        "test_evaluation_seconds": test_evaluation_seconds,
        "epoch_seconds": epoch_seconds,
        "parameter_count": parameter_count,
        "trainable_parameters": parameter_count,
        "input_resolution": config["input_size"][1],
        "batch_size": training_config["batch_size"],
        "gradient_accumulation_steps": config["protocol"][
            "gradient_accumulation_steps"
        ],
        "preprocessing": config["normalization"],
        "test_split_size": len(datasets["test"]),
    }
    write_json(experiment_dir / "metrics.json", metrics)
    write_json(
        experiment_dir / "confusion_matrix.json",
        {
            "class_indices": list(range(num_classes)),
            "matrix": confusion.tolist(),
        },
    )
    write_json(
        experiment_dir / "experiment.json",
        {
            "experiment_id": config["experiment_id"],
            "status": "completed",
            "best_epoch": best_epoch,
        },
    )
    logger.info(
        "completed best_epoch=%d test_accuracy=%.6f test_macro_f1=%.6f",
        best_epoch,
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    return metrics


def render_report(
    report_path: Path,
    dataset_metadata: dict[str, dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    model_configs: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# MedMNIST Transfer Evaluation",
        "",
        "## Research objective",
        "",
        "Evaluate whether three CIFAR-10-optimized HybridCNN configurations transfer "
        "to DermaMNIST and OrganSMNIST, alongside three randomly initialized "
        "torchvision CNN baselines. This is not evidence of state-of-the-art "
        "superiority.",
        "",
        "## Datasets and official splits",
        "",
        "| Dataset | MedMNIST version | Split counts (train/val/test) | Classes | Channels |",
        "|---|---|---:|---:|---:|",
    ]
    for name, metadata in dataset_metadata.items():
        lines.append(
            f"| {name} | {metadata['medmnist_version']} | "
            f"{metadata['split_counts']['train']}/"
            f"{metadata['split_counts']['val']}/"
            f"{metadata['split_counts']['test']} | {metadata['num_classes']} | "
            f"{metadata['source_channels']} |"
        )
    lines.extend(
        [
            "",
            "## Models, input sizes, and preprocessing",
            "",
            "| Model | Input | Training configuration |",
            "|---|---:|---|",
        ]
    )
    for model_name in MODEL_NAMES:
        configs = [
            config
            for config in model_configs.values()
            if config["model"] == model_name
        ]
        example = configs[0]
        settings = example["training"]
        lines.append(
            f"| {model_name} | {example['input_size'][1]}x"
            f"{example['input_size'][2]}x3 | Adam, lr={settings['learning_rate']}, "
            f"batch={settings['batch_size']}, no augmentation, from scratch |"
        )
    lines.extend(
        [
            "",
            "The three torchvision models use their default torchvision "
            "architectures and no pretrained weights. All models use `ToTensor` "
            "with no normalization. OrganSMNIST is converted to three channels "
            "before resizing. HybridCNN inputs are resized to 32x32; baseline "
            "inputs are resized to 224x224. This resolution difference is "
            "intentional and does not create additional image information.",
            "",
            "HybridCNN architectural hyperparameters, learning rate, and batch "
            "size come from the indicated CIFAR-10 run summaries. Source "
            "checkpoints are hashed for provenance but are not loaded. The "
            "baseline settings are Adam, lr=0.001, batch size 32, weight decay "
            "0, and no augmentation.",
            "",
            "## Training protocol",
            "",
            "Maximum 50 epochs; validation macro-F1 selects checkpoints, "
            "drives ReduceLROnPlateau (factor 0.5, patience 4), and controls "
            "early stopping (patience 10). Test data is evaluated only after "
            "loading the best validation checkpoint. No AMP or gradient "
            "accumulation is used.",
            "",
            "## Test results",
            "",
            "| Dataset | Model | Status | Accuracy | Macro precision | Macro recall | "
            "Macro-F1 | Balanced accuracy | Loss | Best epoch | Params | Train seconds |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in entries.values():
        config = entry["config"]
        metrics = entry.get("metrics", {})
        if not metrics:
            lines.append(
                f"| {config['dataset']} | {config['model']} | {entry['status']} | "
                "— | — | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {config['dataset']} | {config['model']} | {entry['status']} | "
            f"{metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | "
            f"{metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['loss']:.4f} | "
            f"{metrics['best_epoch']} | {metrics['parameter_count']} | "
            f"{metrics['total_training_seconds']:.1f} |"
        )
    lines.extend(["", "## Confusion matrices and run status", ""])
    for entry in entries.values():
        config = entry["config"]
        matrix_path = entry["dir"] / "confusion_matrix.json"
        if matrix_path.is_file():
            lines.append(
                f"- {config['dataset']} / {config['model']}: "
                f"`{matrix_path.relative_to(PROJECT_ROOT)}` ({entry['status']})"
            )
        else:
            state = entry["status"]
            lines.append(
                f"- {config['dataset']} / {config['model']}: no test confusion "
                f"matrix; status `{state}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "This study assesses transfer of CIFAR-10-selected HybridCNN "
            "hyperparameters to two medical image tasks. The different input "
            "resolutions, small 28x28 source images, and training-from-scratch "
            "baseline protocol limit direct architectural comparisons. Results "
            "must not be presented as proof of general or state-of-the-art "
            "superiority.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> int:
    import medmnist

    if args.epochs < 1 or args.epochs > TRAINING_SETTINGS["max_epochs"]:
        raise ValueError("--epochs must be between 1 and 50")
    source_configs = source_configurations()
    configs = experiment_configs(source_configs)
    attempts = make_attempt_dirs(configs)
    entries = {
        key: {
            "config": config,
            "dir": attempts[key],
            "status": (
                "completed"
                if (attempts[key] / "metrics.json").is_file()
                else "planned"
            ),
        }
        for key, config in configs.items()
    }
    for key, entry in entries.items():
        if entry["status"] == "completed":
            with (entry["dir"] / "metrics.json").open("r", encoding="utf-8") as handle:
                entry["metrics"] = json.load(handle)

    results_path = RESULTS_ROOT / "results.json"
    write_results(results_path, entries)
    dataset_metadata = {}
    dataset_errors = {}
    for dataset_name in DATASET_NAMES:
        try:
            datasets, info, num_classes, count_mismatches = load_medmnist_splits(
                dataset_name, 32
            )
            dataset_metadata[dataset_name] = {
                "medmnist_version": medmnist.__version__,
                "source_url": info["url"],
                "source_channels": int(info["n_channels"]),
                "num_classes": num_classes,
                "class_names": class_names_from_info(info, num_classes),
                "split_counts": {split: len(data) for split, data in datasets.items()},
                "expected_split_counts": info["n_samples"],
                "split_count_mismatches": count_mismatches,
                "source_resolution": 28,
                "split_source": "official MedMNIST API",
            }
        except Exception as error:
            dataset_errors[dataset_name] = f"{type(error).__name__}: {error}"
    write_json(
        RESULTS_ROOT / "study.json",
        {
            "study": "MedMNIST transfer evaluation",
            "medmnist_version": medmnist.__version__,
            "datasets": dataset_metadata,
            "models": list(MODEL_NAMES),
            "training_protocol": TRAINING_SETTINGS,
            "source_runs": source_configs,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        },
    )
    render_report(
        RESULTS_ROOT / "MEDMNIST_TRANSFER_REPORT.md",
        dataset_metadata,
        entries,
        configs,
    )
    if dataset_errors:
        for key, entry in entries.items():
            dataset_name = entry["config"]["dataset"]
            if dataset_name in dataset_errors:
                entry["status"] = "failed"
                write_json(
                    entry["dir"] / "experiment.json",
                    {
                        "experiment_id": entry["config"]["experiment_id"],
                        "status": "failed",
                        "phase": "dataset_setup",
                        "error": dataset_errors[dataset_name],
                    },
                )
        write_results(results_path, entries)
        render_report(
            RESULTS_ROOT / "MEDMNIST_TRANSFER_REPORT.md",
            dataset_metadata,
            entries,
            configs,
        )
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preflight = []
    failures = []
    for dataset_name in DATASET_NAMES:
        for model_name in MODEL_NAMES:
            resolution = experiment_resolution(model_name)
            try:
                datasets, _, num_classes, _ = load_medmnist_splits(
                    dataset_name, resolution
                )
                result = smoke_test(
                    dataset_name,
                    model_name,
                    datasets,
                    num_classes,
                    source_configs.get(model_name),
                    device,
                    TRAINING_SETTINGS["seed"],
                )
                preflight.append(result)
                entries[f"{dataset_name}/{model_name}"]["smoke"] = "passed"
            except Exception as error:
                failure = {
                    "status": "failed",
                    "dataset": dataset_name,
                    "model": model_name,
                    "error": f"{type(error).__name__}: {error}",
                }
                preflight.append(failure)
                failures.append(failure)
                entries[f"{dataset_name}/{model_name}"]["smoke"] = "failed"
                logger = create_logger(
                    attempts[f"{dataset_name}/{model_name}"] / "run.log"
                )
                logger.exception("preflight_failed")
    write_json(RESULTS_ROOT / "preflight.json", {"checks": preflight})
    write_results(results_path, entries)
    if failures:
        for failure in failures:
            key = f"{failure['dataset']}/{failure['model']}"
            entries[key]["status"] = "failed"
            write_json(
                entries[key]["dir"] / "experiment.json",
                {
                    "experiment_id": entries[key]["config"]["experiment_id"],
                    "status": "failed",
                    "phase": "preflight",
                    "error": failure["error"],
                },
            )
        write_results(results_path, entries)
        render_report(
            RESULTS_ROOT / "MEDMNIST_TRANSFER_REPORT.md",
            dataset_metadata,
            entries,
            configs,
        )
        return 1

    if args.smoke_only:
        render_report(
            RESULTS_ROOT / "MEDMNIST_TRANSFER_REPORT.md",
            dataset_metadata,
            entries,
            configs,
        )
        return 0

    for key, entry in entries.items():
        if entry["status"] == "completed":
            continue
        dataset_name, model_name = key.split("/", 1)
        entry["status"] = "running"
        write_json(
            entry["dir"] / "experiment.json",
            {
                "experiment_id": entry["config"]["experiment_id"],
                "status": "running",
            },
        )
        write_results(results_path, entries)
        render_report(
            RESULTS_ROOT / "MEDMNIST_TRANSFER_REPORT.md",
            dataset_metadata,
            entries,
            configs,
        )
        try:
            resolution = experiment_resolution(model_name)
            datasets, _, num_classes, _ = load_medmnist_splits(
                dataset_name, resolution
            )
            entry["metrics"] = train_experiment(
                entry["config"],
                entry["dir"],
                datasets,
                num_classes,
                source_configs.get(model_name),
                device,
                args.epochs,
            )
            entry["status"] = "completed"
        except Exception as error:
            entry["status"] = "failed"
            write_json(
                entry["dir"] / "experiment.json",
                {
                    "experiment_id": entry["config"]["experiment_id"],
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            logger = create_logger(entry["dir"] / "run.log")
            logger.exception("experiment_failed")
        write_results(results_path, entries)
        render_report(
            RESULTS_ROOT / "MEDMNIST_TRANSFER_REPORT.md",
            dataset_metadata,
            entries,
            configs,
        )
    return 0 if all(entry["status"] == "completed" for entry in entries.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="download/verify official splits and run the 12 one-batch preflight checks",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_SETTINGS["max_epochs"],
        help="maximum epochs per experiment (default: 50)",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
