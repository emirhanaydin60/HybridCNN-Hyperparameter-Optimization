"""Run the six Brain Tumor experiments with an independent test protocol."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_factory import create_model
from model import HybridCNN


OUTPUT_ROOT = ROOT / "results" / "brain_tumor_completion"
DATA_ROOT = Path(r"C:\Users\emirh\Desktop\Projects\datasets\BT")
MAIN_RESULTS = ROOT / "results" / "main_experiment"
MODELS = [
    "HybridCNN_GWO_Run1",
    "HybridCNN_GWO_Run3",
    "HybridCNN_WOA_Run3",
    "efficientnet-b0",
    "mobilenetv3-large",
    "shufflenetv2-1.5x",
]
HYBRID_SUMMARIES = {
    "HybridCNN_GWO_Run1": MAIN_RESULTS / "GWO" / "run_01" / "summary.json",
    "HybridCNN_GWO_Run3": MAIN_RESULTS / "GWO" / "run_03" / "summary.json",
    "HybridCNN_WOA_Run3": MAIN_RESULTS / "WOA" / "run_03" / "summary.json",
}
SEED = 42
EPOCHS = 50
PATIENCE = 10
MICRO_BATCH_SIZE = 16
EFFECTIVE_BATCH_SIZE = 32


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(str(path))
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def metric_summary(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )
    total = matrix.sum()
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }


def split_train(dataset: ImageFolder) -> tuple[Subset, Subset]:
    rng = np.random.default_rng(SEED)
    labels = np.asarray(dataset.targets)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * 0.10)))
        validation_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def make_datasets() -> tuple[Subset, Subset, ImageFolder, int, dict[str, int]]:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_root = DATA_ROOT / "TRAIN"
    test_root = DATA_ROOT / "TEST"
    if not train_root.exists():
        train_root = DATA_ROOT / "train"
    if not test_root.exists():
        test_root = DATA_ROOT / "test"
    full_train = ImageFolder(train_root, transform=transform)
    test = ImageFolder(test_root, transform=transform)
    if full_train.class_to_idx != test.class_to_idx:
        raise RuntimeError("TRAIN and TEST class mappings differ.")
    train, validation = split_train(full_train)
    return train, validation, test, len(full_train.classes), full_train.class_to_idx


def load_hybrid_config(model_name: str) -> dict[str, Any]:
    summary = json.loads(HYBRID_SUMMARIES[model_name].read_text(encoding="utf-8"))
    return summary["best_hyperparameters"]


def build_model(model_name: str, class_count: int) -> tuple[nn.Module, float, dict[str, Any]]:
    if model_name in HYBRID_SUMMARIES:
        config = load_hybrid_config(model_name)
        model = HybridCNN(
            in_channels=3,
            img_size=224,
            shared_conv_kernel_size=config["shared_conv_kernel_size"],
            base_filters=config["base_filters"],
            dilation=config["dilation"],
            final_neurons=config["final_neurons"],
            dropout=config["dropout"],
            se_ratio=config["se_ratio"],
            num_classes=class_count,
        )
        return model, float(config["learning_rate"]), config
    config = {"learning_rate": 0.001, "pretrained": False}
    return create_model(model_name, num_classes=class_count, device="cpu", pretrained=False), 0.001, config


def autocast(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_count: int, amp_enabled: bool) -> tuple[dict[str, float], float, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=amp_enabled)
            targets = targets.to(device, non_blocking=amp_enabled)
            with autocast(amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            for target, prediction in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                matrix[target, prediction] += 1
    return metric_summary(matrix), total_loss / max(total, 1), matrix


def run_one(model_name: str) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir / "run.log")
    train_dataset, validation_dataset, test_dataset, class_count, class_to_idx = make_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    accumulation_steps = math.ceil(EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE)
    model, learning_rate, model_config = build_model(model_name, class_count)
    model = model.to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=amp_enabled)
    validation_loader = DataLoader(validation_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=amp_enabled)
    test_loader = DataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=amp_enabled)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss()
    grad_scaler = scaler(amp_enabled)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }
    checkpoint_path = output_dir / "best_model.pth"
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    logger.info(
        "planned dataset=BrainTumor model=%s seed=%d train=%d validation=%d test=%d micro_batch=%d effective_batch=%d accumulation=%d amp=%s lr=%s",
        model_name,
        SEED,
        len(train_dataset),
        len(validation_dataset),
        len(test_dataset),
        MICRO_BATCH_SIZE,
        EFFECTIVE_BATCH_SIZE,
        accumulation_steps,
        amp_enabled,
        learning_rate,
    )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=amp_enabled)
            targets = targets.to(device, non_blocking=amp_enabled)
            with autocast(amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets) / accumulation_steps
            grad_scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
            count = targets.size(0)
            train_loss += loss.item() * accumulation_steps * count
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += count
        val_metrics, val_loss, _ = evaluate(model, validation_loader, device, class_count, amp_enabled)
        train_loss /= max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["f1_macro"])
        logger.info(
            "epoch=%d train_loss=%.6f train_acc=%.6f val_loss=%.6f val_acc=%.6f val_macro_f1=%.6f",
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
        )
        scheduler.step(val_metrics["f1_macro"])
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
        else:
            stale += 1
        if stale >= PATIENCE:
            logger.info("early_stopping monitor=val_macro_f1 epoch=%d", epoch)
            break
    model.load_state_dict(best_state)
    test_metrics, test_loss, test_matrix = evaluate(model, test_loader, device, class_count, amp_enabled)
    elapsed = time.perf_counter() - start
    config = {
        "dataset": "BrainTumor",
        "model": model_name,
        "seed": SEED,
        "epochs_requested": EPOCHS,
        "patience": PATIENCE,
        "checkpoint_monitor": "val_macro_f1",
        "micro_batch_size": MICRO_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": accumulation_steps,
        "mixed_precision": amp_enabled,
        "device": str(device),
        "learning_rate": learning_rate,
        "image_size": 224,
        "validation_protocol": "stratified 10% split from TRAIN with fixed seed 42",
        "test_protocol": "untouched TEST evaluated once after checkpoint selection",
        "class_to_idx": class_to_idx,
        "model_configuration": model_config,
    }
    write_json(output_dir / "config_used.json", config)
    write_json(output_dir / "training_history.json", history)
    write_json(output_dir / "confusion_matrix.json", {"matrix": test_matrix.tolist(), "class_to_idx": class_to_idx})
    result = {
        "dataset": "BrainTumor",
        "model": model_name,
        "training_status": "COMPLETED",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "best_validation_accuracy": max(history["val_accuracy"]),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_loss": test_loss,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_time_seconds": elapsed,
        "best_model_path": relative_path(checkpoint_path),
        "confusion_matrix_path": relative_path(output_dir / "confusion_matrix.json"),
        "source_files": [relative_path(output_dir / name) for name in ("run.log", "config_used.json", "training_history.json", "confusion_matrix.json")],
        "configuration": config,
        "status": "SUCCESS",
    }
    write_json(output_dir / "result.json", result)
    logger.info(
        "completed best_epoch=%d best_val_macro_f1=%.6f test_accuracy=%.6f test_macro_f1=%.6f time_seconds=%.1f",
        best_epoch,
        best_val_f1,
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
        elapsed,
    )
    return result


def write_master(rows: list[dict[str, Any]]) -> None:
    write_json(OUTPUT_ROOT / "master_results.json", {"rows": rows})
    fields = [
        "dataset",
        "model",
        "training_status",
        "best_epoch",
        "best_validation_macro_f1",
        "best_validation_accuracy",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "total_parameters",
        "trainable_parameters",
        "training_time_seconds",
        "best_model_path",
        "confusion_matrix_path",
        "status",
    ]
    with (OUTPUT_ROOT / "master_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    write_json(
        OUTPUT_ROOT / "final_summary.json",
        {
            "dataset": "BrainTumor",
            "model_count": len(rows),
            "completed": sum(row.get("status") == "SUCCESS" for row in rows),
            "failed": sum(row.get("status") == "FAILED" for row in rows),
            "validation_protocol": "TRAIN 90/10 stratified split; TEST untouched until final evaluation",
            "checkpoint_monitor": "val_macro_f1",
        },
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        output_dir = OUTPUT_ROOT / model_name
        planned = {
            "dataset": "BrainTumor",
            "model": model_name,
            "status": "PLANNED",
            "training_status": "PLANNED",
        }
        write_json(output_dir / "status.json", planned)
        try:
            rows.append(run_one(model_name))
        except Exception as error:
            result = {
                "dataset": "BrainTumor",
                "model": model_name,
                "training_status": "FAILED",
                "status": "FAILED",
                "error": str(error),
                "configuration": {
                    "micro_batch_size": MICRO_BATCH_SIZE,
                    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                    "mixed_precision": torch.cuda.is_available(),
                    "validation_protocol": "TRAIN-only stratified split",
                },
            }
            write_json(output_dir / "result.json", result)
            rows.append(result)
    write_master(rows)


if __name__ == "__main__":
    main()
