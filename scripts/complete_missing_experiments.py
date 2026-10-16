"""Complete missing experiments without modifying the existing result tree.

The script reuses validated CIFAR-10/ISIC-2019 artifacts, recomputes baseline
metrics from stored confusion matrices, and writes all new work below
``results/completion_experiments``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import shutil
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
MAIN_RESULTS = ROOT / "results" / "main_experiment"
FINAL_RESULTS = ROOT / "results" / "final_experiments"
OUTPUT_ROOT = ROOT / "results" / "completion_experiments"
DATA_ROOT = Path(r"C:\Users\emirh\Desktop\Projects\datasets")

MODELS = [
    "HybridCNN_GWO_Run1",
    "HybridCNN_GWO_Run3",
    "HybridCNN_WOA_Run3",
    "efficientnet-b0",
    "mobilenetv3-large",
    "shufflenetv2-1.5x",
]
HYBRIDS = set(MODELS[:3])
BASELINES = set(MODELS[3:])
HYBRID_SUMMARIES = {
    "HybridCNN_GWO_Run1": MAIN_RESULTS / "GWO" / "run_01" / "summary.json",
    "HybridCNN_GWO_Run3": MAIN_RESULTS / "GWO" / "run_03" / "summary.json",
    "HybridCNN_WOA_Run3": MAIN_RESULTS / "WOA" / "run_03" / "summary.json",
}
DATASETS = ("CIFAR10", "ISIC2019", "BrainTumor")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def metric_summary(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )
    total = matrix.sum()
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "precision_macro": float(precision.mean()) if len(precision) else 0.0,
        "recall_macro": float(recall.mean()) if len(recall) else 0.0,
        "f1_macro": float(f1.mean()) if len(f1) else 0.0,
    }


def load_matrix(path: Path) -> np.ndarray:
    data = read_json(path)
    return np.asarray(data["matrix"], dtype=np.int64)


def log_factory(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def source_path(dataset: str, model: str, filename: str) -> Path:
    if dataset == "CIFAR10" and model in HYBRIDS:
        return HYBRID_SUMMARIES[model].parent / filename
    return FINAL_RESULTS / dataset / model / filename


def reuse_existing_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ("CIFAR10", "ISIC2019"):
        for model in MODELS:
            if dataset == "ISIC2019" and model in HYBRIDS:
                continue
            if dataset == "CIFAR10" and model in HYBRIDS:
                summary_path = HYBRID_SUMMARIES[model]
                summary = read_json(summary_path)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "training_status": "REUSED_COMPLETED",
                        "best_epoch": None,
                        "best_validation_accuracy": summary.get("best_validation_accuracy"),
                        "test_accuracy": summary.get("best_test_accuracy"),
                        "test_precision_macro": summary.get("precision"),
                        "test_recall_macro": summary.get("recall"),
                        "test_f1_macro": summary.get("f1"),
                        "total_parameters": None,
                        "trainable_parameters": None,
                        "training_time_seconds": summary.get("final_training_time_seconds"),
                        "best_model_path": relative_or_absolute(summary_path.parent / "best_model.pth"),
                        "confusion_matrix_path": relative_or_absolute(summary_path.parent / "confusion_matrix.png"),
                        "source_files": [relative_or_absolute(summary_path), relative_or_absolute(summary_path.parent / "best_model.pth")],
                        "configuration": summary.get("best_hyperparameters", {}),
                        "status": "REUSED",
                        "notes": "Validated from legacy main_experiment summary and artifacts; best epoch was not recorded.",
                    }
                )
                continue

            result_dir = FINAL_RESULTS / dataset / model
            history_path = result_dir / "training_history.json"
            matrix_path = result_dir / "confusion_matrix.json"
            result_path = result_dir / "result.json"
            if not (history_path.exists() and matrix_path.exists()):
                continue
            history = read_json(history_path)
            matrix = load_matrix(matrix_path)
            metrics = metric_summary(matrix)
            result = read_json(result_path) if result_path.exists() else {}
            val_acc = history.get("val_accuracy", [])
            best_epoch = int(np.argmax(val_acc) + 1) if val_acc else None
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "training_status": "REUSED_COMPLETED",
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": max(val_acc) if val_acc else None,
                    "test_accuracy": metrics["accuracy"],
                    "test_precision_macro": metrics["precision_macro"],
                    "test_recall_macro": metrics["recall_macro"],
                    "test_f1_macro": metrics["f1_macro"],
                    "total_parameters": result.get("parameter_count", {}).get("total"),
                    "trainable_parameters": result.get("parameter_count", {}).get("trainable"),
                    "training_time_seconds": result.get("train_time_seconds"),
                    "best_model_path": relative_or_absolute(result_dir / "best_model.pth"),
                    "confusion_matrix_path": relative_or_absolute(matrix_path),
                    "source_files": [
                        relative_or_absolute(result_dir / "run.log"),
                        relative_or_absolute(history_path),
                        relative_or_absolute(matrix_path),
                    ],
                    "configuration": {"source_result": relative_or_absolute(result_path)},
                    "status": "REUSED_METRICS_RECOMPUTED",
                    "notes": "Metrics recomputed from the stored confusion matrix; original zero/null macro metrics were not reused.",
                }
            )
    return rows


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def stratified_split(dataset: ImageFolder, seed: int, validation_ratio: float = 0.1) -> tuple[Subset, Subset]:
    labels = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * validation_ratio)))
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def make_datasets(dataset: str, seed: int) -> tuple[Any, Any, Any, int, dict[str, int]]:
    transform = image_transform()
    if dataset == "ISIC2019":
        root = DATA_ROOT / "input_sk"
        train = ImageFolder(root / "train", transform=transform)
        validation = ImageFolder(root / "val", transform=transform)
        test = ImageFolder(root / "test", transform=transform)
        if train.class_to_idx != validation.class_to_idx or train.class_to_idx != test.class_to_idx:
            raise RuntimeError("ISIC-2019 class mappings differ between train, val, and test.")
    elif dataset == "BrainTumor":
        root = DATA_ROOT / "BT"
        train_root = root / "TRAIN"
        test_root = root / "TEST"
        if not train_root.exists():
            train_root = root / "train"
        if not test_root.exists():
            test_root = root / "test"
        full_train = ImageFolder(train_root, transform=transform)
        test = ImageFolder(test_root, transform=transform)
        if full_train.class_to_idx != test.class_to_idx:
            raise RuntimeError("Brain Tumor class mappings differ between TRAIN and TEST.")
        train, validation = stratified_split(full_train, seed)
    else:
        raise ValueError(f"Unsupported completion dataset: {dataset}")
    return train, validation, test, len(train.class_to_idx if hasattr(train, "class_to_idx") else train.dataset.class_to_idx), (
        full_train.class_to_idx if dataset == "BrainTumor" else train.class_to_idx
    )


def build_model_for(model_name: str, class_count: int) -> nn.Module:
    from experiment_runner import build_model
    from model_factory import create_model

    if model_name in HYBRIDS:
        summary = read_json(HYBRID_SUMMARIES[model_name])
        return build_model(summary["best_hyperparameters"], 3, 224, class_count)
    return create_model(model_name, num_classes=class_count, device="cpu", pretrained=False)


def amp_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def preflight_batch(model_name: str, dataset: str, class_count: int, candidates: list[int]) -> tuple[int, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    for batch_size in candidates:
        model = build_model_for(model_name, class_count).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = make_scaler(amp_enabled)
        try:
            inputs = torch.zeros(batch_size, 3, 224, 224, device=device)
            targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(amp_enabled):
                loss = nn.CrossEntropyLoss()(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            del inputs, targets, model, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return batch_size, amp_enabled
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            del model, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return 1, amp_enabled


def train_and_evaluate(dataset: str, model_name: str, seed: int, epochs: int, patience: int) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / dataset / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = log_factory(output_dir / "run.log")
    train_dataset, validation_dataset, test_dataset, class_count, class_to_idx = make_datasets(dataset, seed)
    candidates = [8, 4, 2, 1]
    batch_size, amp_enabled = preflight_batch(model_name, dataset, class_count, candidates)
    effective_batch_size = 32
    accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
    logger.info(
        "Starting isolated experiment dataset=%s model=%s batch_size=%d accumulation_steps=%d amp=%s classes=%s",
        dataset,
        model_name,
        batch_size,
        accumulation_steps,
        amp_enabled,
        class_to_idx,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=amp_enabled)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=amp_enabled)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=amp_enabled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for(model_name, class_count).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss()
    scaler = make_scaler(amp_enabled)
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    checkpoint_path = output_dir / "best_model.pth"
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_epoch = 0
    stale_epochs = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=amp_enabled)
            targets = targets.to(device, non_blocking=amp_enabled)
            with amp_context(amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets) / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            count = targets.size(0)
            train_loss += loss.item() * accumulation_steps * count
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += count
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, non_blocking=amp_enabled)
                targets = targets.to(device, non_blocking=amp_enabled)
                with amp_context(amp_enabled):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                count = targets.size(0)
                val_loss += loss.item() * count
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += count
        train_loss /= max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)
        val_loss /= max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        logger.info(
            "epoch=%d train_loss=%.6f train_acc=%.6f val_loss=%.6f val_acc=%.6f",
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )
        scheduler.step(val_accuracy)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            logger.info("early_stopping triggered at epoch=%d", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    test_loss = 0.0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=amp_enabled)
            targets = targets.to(device, non_blocking=amp_enabled)
            with amp_context(amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)
            for target, prediction in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                matrix[target, prediction] += 1
    metrics = metric_summary(matrix)
    elapsed = time.perf_counter() - start
    matrix_path = output_dir / "confusion_matrix.json"
    history_path = output_dir / "training_history.json"
    config = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "mixed_precision": amp_enabled,
        "device": str(device),
        "class_to_idx": class_to_idx,
        "validation_protocol": "stratified 10% split from TRAIN" if dataset == "BrainTumor" else "existing train/val/test folders",
        "hyperparameters_source": relative_or_absolute(HYBRID_SUMMARIES[model_name]) if model_name in HYBRIDS else "baseline protocol",
    }
    write_json(matrix_path, {"matrix": matrix.tolist(), "class_to_idx": class_to_idx})
    write_json(history_path, history)
    write_json(output_dir / "config_used.json", config)
    result = {
        "dataset": dataset,
        "model": model_name,
        "training_status": "COMPLETED",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val,
        "test_accuracy": metrics["accuracy"],
        "test_precision_macro": metrics["precision_macro"],
        "test_recall_macro": metrics["recall_macro"],
        "test_f1_macro": metrics["f1_macro"],
        "test_loss": test_loss / max(test_total, 1),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_time_seconds": elapsed,
        "best_model_path": relative_or_absolute(checkpoint_path),
        "confusion_matrix_path": relative_or_absolute(matrix_path),
        "source_files": [relative_or_absolute(output_dir / name) for name in ("run.log", "training_history.json", "config_used.json", "confusion_matrix.json")],
        "configuration": config,
        "status": "SUCCESS",
        "notes": "Independent TEST evaluation after validation-based checkpoint selection.",
    }
    write_json(output_dir / "result.json", result)
    logger.info(
        "Finished training best_epoch=%d best_val_acc=%.6f test_accuracy=%.6f test_f1_macro=%.6f elapsed=%.1fs",
        best_epoch,
        best_val,
        metrics["accuracy"],
        metrics["f1_macro"],
        elapsed,
    )
    return result


def write_consolidated(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "training_status",
        "best_epoch",
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
        "source_files",
        "configuration",
        "status",
        "notes",
    ]
    with (OUTPUT_ROOT / "master_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["source_files"] = json.dumps(csv_row.get("source_files", []), ensure_ascii=False)
            csv_row["configuration"] = json.dumps(csv_row.get("configuration", {}), ensure_ascii=False)
            writer.writerow({field: csv_row.get(field) for field in fields})
    write_json(OUTPUT_ROOT / "master_results.json", {"rows": rows})
    summary = {
        "total_experiments": len(rows),
        "successful_or_reused": sum(row.get("status") in {"SUCCESS", "REUSED", "REUSED_METRICS_RECOMPUTED"} for row in rows),
        "datasets": {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in DATASETS},
        "output_root": relative_or_absolute(OUTPUT_ROOT),
        "notes": "Existing result directories were not modified. New experiments and consolidated references are isolated here.",
    }
    write_json(OUTPUT_ROOT / "final_summary.json", summary)


def placeholder_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            if dataset == "CIFAR10" or (dataset == "ISIC2019" and model in BASELINES):
                continue
            if dataset == "BrainTumor" and model in BASELINES:
                status = "REQUIRES_RETRAINING"
                training_status = "NOT_REUSED_INVALID_TEST_PROTOCOL"
                notes = "Existing checkpoint selected TEST as validation; independent Brain Tumor test result requires retraining with a TRAIN-only validation split."
            else:
                status = "REQUIRES_TRAINING"
                training_status = "INCOMPLETE_CUDA_OOM"
                notes = "No completed, independently verifiable training result. Existing attempts ended in CUDA OOM; a controlled run was stopped before producing a final result because the measured epoch cost was prohibitive."
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "training_status": training_status,
                    "best_epoch": None,
                    "best_validation_accuracy": None,
                    "test_accuracy": None,
                    "test_precision_macro": None,
                    "test_recall_macro": None,
                    "test_f1_macro": None,
                    "total_parameters": None,
                    "trainable_parameters": None,
                    "training_time_seconds": None,
                    "best_model_path": None,
                    "confusion_matrix_path": None,
                    "source_files": [
                        relative_or_absolute(FINAL_RESULTS / dataset / model / "run.log"),
                        relative_or_absolute(FINAL_RESULTS / dataset / model / "result.json"),
                    ],
                    "configuration": {},
                    "status": status,
                    "notes": notes,
                }
            )
    return rows


def merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {(row["dataset"], row["model"]): row for row in rows}
    for row in placeholder_rows():
        merged.setdefault((row["dataset"], row["model"]), row)
    return [merged[key] for key in sorted(merged)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="Probe feasible batch sizes only.")
    parser.add_argument("--run", action="store_true", help="Run missing experiments.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", help="Optional dataset/model pairs, e.g. ISIC2019/HybridCNN_GWO_Run1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled completion run.")
    rows = reuse_existing_results()
    if args.preflight:
        preflight_rows = []
        for dataset, models in (
            ("ISIC2019", sorted(HYBRIDS)),
            ("BrainTumor", sorted(HYBRIDS | BASELINES)),
        ):
            _, _, _, class_count, _ = make_datasets(dataset, args.seed)
            for model in models:
                batch_size, amp_enabled = preflight_batch(model, dataset, class_count, [8, 4, 2, 1])
                preflight_rows.append({"dataset": dataset, "model": model, "class_count": class_count, "batch_size": batch_size, "mixed_precision": amp_enabled})
        write_json(OUTPUT_ROOT / "preflight.json", preflight_rows)
        print(json.dumps(preflight_rows, indent=2))
    if args.run:
        targets = [(dataset, model) for dataset in ("ISIC2019", "BrainTumor") for model in MODELS if dataset == "ISIC2019" and model in HYBRIDS or dataset == "BrainTumor"]
        if args.only:
            requested = set(args.only)
            targets = [(dataset, model) for dataset, model in targets if f"{dataset}/{model}" in requested]
        for dataset, model in targets:
            try:
                rows.append(train_and_evaluate(dataset, model, args.seed, args.epochs, args.patience))
            except Exception as error:
                output_dir = OUTPUT_ROOT / dataset / model
                output_dir.mkdir(parents=True, exist_ok=True)
                write_json(output_dir / "result.json", {"dataset": dataset, "model": model, "status": "FAILED", "error": str(error)})
                rows.append({"dataset": dataset, "model": model, "training_status": "FAILED", "status": "FAILED", "notes": str(error)})
                print(f"FAILED {dataset}/{model}: {error}")
        write_consolidated(merge_rows(rows))
    elif not args.preflight:
        write_consolidated(merge_rows(rows))


if __name__ == "__main__":
    main()
