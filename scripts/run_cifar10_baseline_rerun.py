"""Rerun the three torchvision baseline models on CIFAR-10."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import random_split
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_medmnist_transfer as pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "cifar10_baseline_pipeline_rerun"
DATA_ROOT = PROJECT_ROOT / "data"
HISTORICAL_ROOT = PROJECT_ROOT / "results" / "final_experiments"
SEED = 42
TRAIN_SIZE = 45_000
VALIDATION_SIZE = 5_000
TEST_SIZE = 10_000
INPUT_RESOLUTION = 224
MODEL_NAMES = (
    "efficientnet-b0",
    "shufflenetv2-1.5x",
    "mobilenetv3-large",
)
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cifar10():
    transform = transforms.Compose(
        [
            transforms.Resize(
                (INPUT_RESOLUTION, INPUT_RESOLUTION),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    official_train = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=transform,
    )
    official_test = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=transform,
    )
    if tuple(official_train.classes) != CLASS_NAMES:
        raise ValueError(f"Unexpected CIFAR-10 class mapping: {official_train.classes}")
    if tuple(official_test.classes) != CLASS_NAMES:
        raise ValueError(f"Unexpected CIFAR-10 test mapping: {official_test.classes}")
    if len(official_train) != 50_000 or len(official_test) != TEST_SIZE:
        raise ValueError(
            f"Unexpected CIFAR-10 split sizes: {len(official_train)} train, "
            f"{len(official_test)} test"
        )
    for split_name, target_values in (
        ("train", official_train.targets),
        ("test", official_test.targets),
    ):
        if sorted(set(int(label) for label in target_values)) != list(range(10)):
            raise ValueError(f"Invalid CIFAR-10 label indices in {split_name}")

    generator = torch.Generator().manual_seed(SEED)
    train_subset, validation_subset = random_split(
        official_train,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=generator,
    )
    train_indices = [int(index) for index in train_subset.indices]
    validation_indices = [int(index) for index in validation_subset.indices]
    if set(train_indices).intersection(validation_indices):
        raise ValueError("CIFAR-10 train and validation indices overlap")
    if len(set(train_indices + validation_indices)) != len(official_train):
        raise ValueError("Train and validation split does not cover official train")
    data_sets = {
        "train": train_subset,
        "val": validation_subset,
        "test": official_test,
    }
    split_manifest = {
        "source": "torchvision.datasets.CIFAR10 official train/test splits",
        "torchvision_version": torchvision.__version__,
        "train_official_size": len(official_train),
        "train_size": len(train_subset),
        "validation_size": len(validation_subset),
        "test_size": len(official_test),
        "split_seed": SEED,
        "classes": list(CLASS_NAMES),
        "class_indices": list(range(10)),
        "transform": (
            "Resize 224x224 with bilinear interpolation, then ToTensor; "
            "no normalization or augmentation"
        ),
    }
    indices = {
        "seed": SEED,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    return data_sets, split_manifest, indices


def load_historical_results() -> dict[str, dict[str, Any]]:
    csv_path = HISTORICAL_ROOT / "master_results.csv"
    previous_rows = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") == "CIFAR10" and row.get("model") in MODEL_NAMES:
                previous_rows[row["model"]] = row

    historical = {}
    for model_name in MODEL_NAMES:
        if model_name not in previous_rows:
            raise ValueError(f"Missing historical CIFAR-10 row for {model_name}")
        row = previous_rows[model_name]
        if row.get("status") != "SUCCESS":
            raise ValueError(f"Historical run for {model_name} is not successful")
        confusion_path = (
            HISTORICAL_ROOT
            / "CIFAR10"
            / model_name
            / "confusion_matrix.json"
        )
        checkpoint_path = (
            HISTORICAL_ROOT / "CIFAR10" / model_name / "best_model.pth"
        )
        with confusion_path.open("r", encoding="utf-8") as handle:
            confusion_payload = json.load(handle)
        confusion = np.asarray(confusion_payload["matrix"], dtype=np.int64)
        if confusion.shape != (10, 10) or int(confusion.sum()) != TEST_SIZE:
            raise ValueError(f"Invalid historical confusion matrix for {model_name}")
        metrics = pipeline.confusion_metrics(confusion)
        if not np.isclose(
            metrics["accuracy"], float(row["test_accuracy"]), rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Historical accuracy mismatch for {model_name}")
        historical[model_name] = {
            "status": row["status"],
            "seed": int(row["seed"]),
            "best_epoch": int(row["best_epoch"]),
            "trainable_parameters": int(row["trainable_parameters"]),
            "test_accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "training_time_seconds": float(row["training_time_seconds"]),
            "master_results_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "master_results_csv_sha256": sha256_file(csv_path),
            "confusion_matrix_path": str(confusion_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_sha256": sha256_file(confusion_path),
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_checkpoint_loaded": False,
        }
    return historical


def build_configs(historical: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    configs = {}
    for model_name in MODEL_NAMES:
        configs[model_name] = {
            "experiment_id": f"CIFAR10-{model_name}-baseline-rerun",
            "dataset": "CIFAR10",
            "model": model_name,
            "num_classes": 10,
            "class_names": list(CLASS_NAMES),
            "input_size": [3, INPUT_RESOLUTION, INPUT_RESOLUTION],
            "resize": {
                "size": [INPUT_RESOLUTION, INPUT_RESOLUTION],
                "interpolation": "bilinear",
            },
            "grayscale_to_rgb": False,
            "normalization": (
                "none; bilinear Resize then ToTensor scales uint8 [0,255] "
                "to float32 [0,1]"
            ),
            "training": pipeline.experiment_training_config(None),
            "protocol": copy.deepcopy(pipeline.TRAINING_SETTINGS),
            "pretrained": False,
            "initialization": "random; historical checkpoint used for provenance only",
            "historical_run": historical[model_name],
            "architecture": {
                "factory": "torchvision via model_factory.create_model",
                "num_classes": 10,
                "pretrained": False,
            },
        }
    return configs


def update_summaries(entries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for entry in entries.values():
        config = entry["config"]
        metrics = entry.get("metrics", {})
        old = config["historical_run"]
        row = {
            "configuration": config["model"],
            "status": entry["status"],
            "seed": SEED,
            "input_resolution": INPUT_RESOLUTION,
            "accuracy": metrics.get("accuracy"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "macro_f1": metrics.get("macro_f1"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "test_loss": metrics.get("loss"),
            "best_validation_macro_f1": metrics.get("best_validation_macro_f1"),
            "best_epoch": metrics.get("best_epoch"),
            "trainable_parameters": metrics.get("trainable_parameters"),
            "total_training_seconds": metrics.get("total_training_seconds"),
            "test_evaluation_seconds": metrics.get("test_evaluation_seconds"),
            "historical_accuracy": old["test_accuracy"],
            "historical_macro_precision": old["macro_precision"],
            "historical_macro_recall": old["macro_recall"],
            "historical_macro_f1": old["macro_f1"],
            "run_directory": str(entry["dir"].relative_to(PROJECT_ROOT)),
            "error": entry.get("error"),
        }
        rows.append(row)
    pipeline.write_json(OUTPUT_ROOT / "summary.json", {"runs": rows})
    with (OUTPUT_ROOT / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pipeline.write_json(OUTPUT_ROOT / "results.json", {"experiments": rows})


def validate_completed_run(entry: dict[str, Any]) -> None:
    run_dir = entry["dir"]
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (run_dir / "history.json").open("r", encoding="utf-8") as handle:
        history = json.load(handle)["epochs"]
    with (run_dir / "confusion_matrix.json").open(
        "r", encoding="utf-8"
    ) as handle:
        matrix_payload = json.load(handle)
    matrix = np.asarray(matrix_payload["matrix"], dtype=np.int64)
    if matrix.shape != (10, 10) or int(matrix.sum()) != TEST_SIZE:
        raise ValueError(f"Invalid CIFAR-10 confusion matrix for {entry['config']['model']}")
    if matrix_payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Exported confusion matrix has incorrect class ordering")
    recomputed = pipeline.confusion_metrics(matrix)
    for metric_name in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    ):
        if not np.isclose(
            recomputed[metric_name], metrics[metric_name], rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"{metric_name} disagrees with confusion matrix for "
                f"{entry['config']['model']}"
            )
    if not np.isclose(
        metrics["macro_recall"], metrics["balanced_accuracy"], atol=1e-12
    ):
        raise ValueError("CIFAR-10 balanced accuracy must equal macro recall")
    best_record = max(history, key=lambda item: item["validation_macro_f1"])
    if best_record["epoch"] != metrics["best_epoch"]:
        raise ValueError("Best checkpoint does not match highest validation Macro-F1")
    if not np.isclose(
        best_record["validation_macro_f1"],
        metrics["best_validation_macro_f1"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Best validation Macro-F1 disagrees with training history")
    expected_parameters = entry["config"]["historical_run"]["trainable_parameters"]
    if metrics["parameter_count"] != expected_parameters:
        raise ValueError("Instantiated baseline parameter count changed")
    entry["metrics"] = metrics


def render_report(entries: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# CIFAR-10 Baseline Pipeline Rerun",
        "",
        "Fresh-scratch reruns of EfficientNet-B0, ShuffleNetV2-1.5x, and "
        "MobileNetV3-Large using the shared MedMNIST transfer trainer.",
        "",
        "## Protocol",
        "",
        "- Official CIFAR-10 train/test splits; train/validation split is 45,000/5,000 "
        "with seed 42, and the official 10,000-image test split is kept separate.",
        "- Baseline input follows the shared transfer pipeline at 224x224 RGB, "
        "bilinear resize, then ToTensor only. No normalization, augmentation, or "
        "pretrained weights.",
        "- Adam, learning rate 0.001, batch 32, maximum 50 epochs; checkpoint and "
        "early stopping use validation Macro-F1. ReduceLROnPlateau monitors "
        "validation Macro-F1 (factor 0.5, patience 4); early stopping patience 10.",
        "- Seed 42 is set for Python, NumPy, PyTorch, and CUDA when available. "
        "cuDNN deterministic mode is enabled and benchmark mode disabled.",
        "",
        "## Results",
        "",
        "| Model | Status | Accuracy | Macro precision | Macro recall | Macro-F1 | "
        "Balanced accuracy | Test loss | Best epoch | Parameters | Train seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        entry = entries[model_name]
        metrics = entry.get("metrics", {})
        if not metrics:
            lines.append(f"| {model_name} | {entry['status']} | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {model_name} | {entry['status']} | {metrics['accuracy']:.6f} | "
            f"{metrics['macro_precision']:.6f} | {metrics['macro_recall']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['balanced_accuracy']:.6f} | "
            f"{metrics['loss']:.6f} | {metrics['best_epoch']} | "
            f"{metrics['trainable_parameters']} | "
            f"{metrics['total_training_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Historical comparison",
            "",
            "Historical CIFAR-10 scores are calculated from the preserved original "
            "confusion matrices. The old CIFAR loader used native 32x32 inputs and "
            "normalization; these reruns use the transfer pipeline's 224x224 resize "
            "and ToTensor-only policy, so differences are protocol-related as well "
            "as reflecting fresh initialization.",
            "",
            "| Model | Metric | Historical | Rerun | Difference |",
            "|---|---|---:|---:|---:|",
        ]
    )
    metrics_to_compare = (
        ("accuracy", "test_accuracy"),
        ("macro_precision", "macro_precision"),
        ("macro_recall", "macro_recall"),
        ("macro_f1", "macro_f1"),
    )
    for model_name in MODEL_NAMES:
        entry = entries[model_name]
        historical = entry["config"]["historical_run"]
        current = entry.get("metrics", {})
        for display_name, historical_key in metrics_to_compare:
            old_value = historical[historical_key]
            current_value = current.get(display_name)
            if current_value is None:
                lines.append(f"| {model_name} | {display_name} | {old_value:.6f} | — | — |")
            else:
                lines.append(
                    f"| {model_name} | {display_name} | {old_value:.6f} | "
                    f"{current_value:.6f} | {current_value - old_value:+.6f} |"
                )
    lines.extend(
        [
            "",
            "These are single runs, not repeated trials sufficient for a statistical "
            "comparison. Historical checkpoints were recorded for provenance only "
            "and were not loaded.",
            "",
        ]
    )
    (OUTPUT_ROOT / "CIFAR10_BASELINE_RERUN_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run() -> int:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing baseline rerun outputs in {OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    historical = load_historical_results()
    configs = build_configs(historical)
    data_sets, split_manifest, split_indices = load_cifar10()
    pipeline.write_json(OUTPUT_ROOT / "split_manifest.json", split_manifest)
    pipeline.write_json(OUTPUT_ROOT / "split_indices.json", split_indices)

    pipeline.set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = {
        "study": "CIFAR-10 baseline pipeline rerun",
        "dataset": "CIFAR10",
        "official_train_size": 50_000,
        "official_test_size": TEST_SIZE,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "training_protocol": pipeline.TRAINING_SETTINGS,
        "historical_runs": historical,
        "source_checkpoints_loaded": False,
    }
    pipeline.write_json(OUTPUT_ROOT / "study.json", study)

    entries = {}
    for model_name in MODEL_NAMES:
        run_dir = OUTPUT_ROOT / "runs" / model_name
        run_dir.mkdir(parents=True, exist_ok=False)
        config = configs[model_name]
        pipeline.save_yaml(run_dir / "config.yaml", config)
        pipeline.write_json(
            run_dir / "experiment.json",
            {"experiment_id": config["experiment_id"], "status": "planned"},
        )
        entries[model_name] = {
            "config": config,
            "dir": run_dir,
            "status": "planned",
            "metrics": {},
        }
    update_summaries(entries)
    render_report(entries)

    preflight = []
    for model_name in MODEL_NAMES:
        try:
            result = pipeline.smoke_test(
                "CIFAR10",
                model_name,
                data_sets,
                10,
                None,
                device,
                SEED,
            )
            expected_parameters = historical[model_name]["trainable_parameters"]
            if result["parameter_count"] != expected_parameters:
                raise ValueError(
                    f"{model_name} parameter count {result['parameter_count']} != "
                    f"historical architecture count {expected_parameters}"
                )
            result["class_names"] = list(CLASS_NAMES)
            preflight.append(result)
        except Exception as error:
            preflight.append(
                {
                    "status": "failed",
                    "dataset": "CIFAR10",
                    "model": model_name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    pipeline.write_json(OUTPUT_ROOT / "preflight.json", {"checks": preflight})
    failed = [item for item in preflight if item["status"] != "passed"]
    if failed:
        for item in failed:
            entry = entries[item["model"]]
            entry["status"] = "failed"
            entry["error"] = item["error"]
            pipeline.write_json(
                entry["dir"] / "experiment.json",
                {
                    "experiment_id": entry["config"]["experiment_id"],
                    "status": "failed",
                    "phase": "preflight",
                    "error": item["error"],
                },
            )
        update_summaries(entries)
        render_report(entries)
        return 1

    for model_name in MODEL_NAMES:
        entry = entries[model_name]
        config = entry["config"]
        run_dir = entry["dir"]
        entry["status"] = "running"
        pipeline.write_json(
            run_dir / "experiment.json",
            {"experiment_id": config["experiment_id"], "status": "running"},
        )
        update_summaries(entries)
        started = time.perf_counter()
        try:
            metrics = pipeline.train_experiment(
                config,
                run_dir,
                data_sets,
                10,
                None,
                device,
                epochs=50,
            )
            metrics["trainable_parameters"] = metrics["parameter_count"]
            entry["metrics"] = metrics
            matrix_path = run_dir / "confusion_matrix.json"
            with matrix_path.open("r", encoding="utf-8") as handle:
                matrix_payload = json.load(handle)
            matrix_payload["class_names"] = list(CLASS_NAMES)
            pipeline.write_json(matrix_path, matrix_payload)
            validate_completed_run(entry)

            historical_run = historical[model_name]
            pipeline.write_json(
                run_dir / "run_metadata.json",
                {
                    "seed": SEED,
                    "status": "completed",
                    "device": str(device),
                    "cuda_device_name": study["cuda_device_name"],
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "torchvision_version": torchvision.__version__,
                    "numpy_version": np.__version__,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "deterministic_algorithms": (
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "source_checkpoint_loaded": False,
                    "checkpoint_sha256": sha256_file(run_dir / "best_checkpoint.pth"),
                    "wall_seconds_including_test_evaluation": (
                        time.perf_counter() - started
                    ),
                    "historical_run_provenance": historical_run,
                },
            )
            entry["status"] = "completed"
            pipeline.write_json(
                run_dir / "experiment.json",
                {
                    "experiment_id": config["experiment_id"],
                    "status": "completed",
                    "best_epoch": metrics["best_epoch"],
                },
            )
        except Exception as error:
            entry["status"] = "failed"
            entry["error"] = f"{type(error).__name__}: {error}"
            pipeline.write_json(
                run_dir / "experiment.json",
                {
                    "experiment_id": config["experiment_id"],
                    "status": "failed",
                    "error": entry["error"],
                },
            )
        update_summaries(entries)
        render_report(entries)

    return int(any(entry["status"] != "completed" for entry in entries.values()))


if __name__ == "__main__":
    raise SystemExit(run())
