"""Rerun the selected HybridCNN configurations on CIFAR-10."""

from __future__ import annotations

import csv
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
OUTPUT_ROOT = PROJECT_ROOT / "results" / "cifar10_pipeline_rerun"
DATA_ROOT = PROJECT_ROOT / "data"
SEED = 42
TRAIN_SIZE = 45_000
VALIDATION_SIZE = 5_000
TEST_SIZE = 10_000
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
RUN_NAMES = (
    "HybridCNN_GWO_Run1",
    "HybridCNN_GWO_Run3",
    "HybridCNN_WOA_Run3",
)
EXPECTED_HYPERPARAMETERS = {
    "HybridCNN_GWO_Run1": {
        "shared_conv_kernel_size": 5,
        "base_filters": 256,
        "dilation": 1,
        "final_neurons": 256,
        "se_ratio": 8,
        "batch_size": 32,
        "learning_rate": 0.0012351628680148583,
    },
    "HybridCNN_GWO_Run3": {
        "shared_conv_kernel_size": 3,
        "base_filters": 256,
        "dilation": 1,
        "final_neurons": 256,
        "se_ratio": 4,
        "batch_size": 32,
        "learning_rate": 0.0014598678014005684,
    },
    "HybridCNN_WOA_Run3": {
        "shared_conv_kernel_size": 5,
        "base_filters": 256,
        "dilation": 1,
        "final_neurons": 128,
        "se_ratio": 4,
        "batch_size": 32,
        "learning_rate": 0.0014188144847256917,
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def sha256_json(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cifar10():
    transform = transforms.ToTensor()
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

    return official_train, official_test


def make_cifar10_split(official_train, official_test, split_seed: int):
    split_generator = torch.Generator().manual_seed(split_seed)
    train_subset, validation_subset = random_split(
        official_train,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=split_generator,
    )
    train_indices = [int(index) for index in train_subset.indices]
    validation_indices = [int(index) for index in validation_subset.indices]
    if set(train_indices).intersection(validation_indices):
        raise ValueError("CIFAR-10 train and validation indices overlap")
    if len(train_indices) != TRAIN_SIZE or len(validation_indices) != VALIDATION_SIZE:
        raise ValueError("CIFAR-10 train/validation split size mismatch")
    if len(set(train_indices + validation_indices)) != len(official_train):
        raise ValueError("The train/validation split does not cover official train data")

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
        "split_seed": split_seed,
        "train_indices_sha256": sha256_json(train_indices),
        "validation_indices_sha256": sha256_json(validation_indices),
        "test_source": "official CIFAR-10 test split; no overlap with train source",
        "classes": list(CLASS_NAMES),
        "class_indices": list(range(10)),
        "transform": "ToTensor only; no normalization or augmentation",
    }
    indices = {
        "seed": split_seed,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    return data_sets, split_manifest, indices


def verify_hyperparameters(
    source_configs: dict[str, dict[str, Any]],
) -> None:
    for run_name in RUN_NAMES:
        actual = source_configs[run_name]["hyperparameters"]
        for key, expected in EXPECTED_HYPERPARAMETERS[run_name].items():
            value = actual[key]
            if isinstance(expected, float):
                matches = np.isclose(value, expected, rtol=0.0, atol=1e-15)
            else:
                matches = value == expected
            if not matches:
                raise ValueError(
                    f"Verified source hyperparameter discrepancy for {run_name}: "
                    f"{key} is {value!r}, expected {expected!r}. No run started."
                )


def build_run_configs(source_configs: dict[str, dict[str, Any]]):
    run_configs = {}
    for run_name in RUN_NAMES:
        source = source_configs[run_name]
        hp = source["hyperparameters"]
        protocol = dict(pipeline.TRAINING_SETTINGS)
        protocol["seed"] = SEED
        protocol["max_epochs"] = 50
        config = {
            "experiment_id": f"CIFAR10-{run_name}",
            "dataset": "CIFAR10",
            "model": run_name,
            "num_classes": 10,
            "class_names": list(CLASS_NAMES),
            "input_size": [3, 32, 32],
            "resize": "none; CIFAR-10 images are already 32x32",
            "grayscale_to_rgb": False,
            "normalization": "none; ToTensor scales uint8 [0,255] to float32 [0,1]",
            "training": pipeline.experiment_training_config(source),
            "protocol": protocol,
            "data_split_seed": int(source["source_run_seed"]),
            "data_split": None,
            "source_run": source,
            "pretrained": False,
            "initialization": "random; source checkpoint used only for provenance and architecture verification",
            "architecture": {
                "in_channels": 3,
                "img_size": 32,
                "shared_conv_kernel_size": hp["shared_conv_kernel_size"],
                "base_filters": hp["base_filters"],
                "dilation": hp["dilation"],
                "final_neurons": hp["final_neurons"],
                "dropout": hp["dropout"],
                "se_ratio": hp["se_ratio"],
                "num_classes": 10,
            },
        }
        run_configs[run_name] = config
    return run_configs


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def update_summaries(entries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for entry in entries.values():
        config = entry["config"]
        metrics = entry.get("metrics", {})
        rows.append(
            {
                "configuration": config["model"],
                "status": entry["status"],
                "seed": SEED,
                "data_split_seed": config["data_split_seed"],
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
                "run_directory": str(entry["dir"].relative_to(PROJECT_ROOT)),
                "error": entry.get("error"),
            }
        )
    write_json(OUTPUT_ROOT / "summary.json", {"runs": rows})
    csv_path = OUTPUT_ROOT / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUTPUT_ROOT / "results.json", {"experiments": rows})


def validate_completed_run(entry: dict[str, Any], test_size: int) -> None:
    config = entry["config"]
    run_dir = entry["dir"]
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (run_dir / "confusion_matrix.json").open("r", encoding="utf-8") as handle:
        matrix_payload = json.load(handle)
    with (run_dir / "history.json").open("r", encoding="utf-8") as handle:
        history = json.load(handle)["epochs"]

    confusion = np.asarray(matrix_payload["matrix"], dtype=np.int64)
    if confusion.shape != (10, 10):
        raise ValueError(f"Expected a 10x10 confusion matrix, got {confusion.shape}")
    if int(confusion.sum()) != test_size:
        raise ValueError(
            f"Confusion matrix total {confusion.sum()} != CIFAR-10 test size {test_size}"
        )
    recomputed = pipeline.confusion_metrics(confusion)
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
                f"{metric_name} disagrees with the exported confusion matrix for "
                f"{config['model']}"
            )
    if not np.isclose(
        metrics["balanced_accuracy"], metrics["macro_recall"], rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"Balanced accuracy != macro recall for {config['model']}")
    if not np.isclose(
        metrics["accuracy"], float(np.trace(confusion) / confusion.sum()), atol=1e-12
    ):
        raise ValueError(f"Accuracy disagrees with confusion matrix for {config['model']}")

    best_history_epoch = max(
        history, key=lambda record: record["validation_macro_f1"]
    )
    if best_history_epoch["epoch"] != metrics["best_epoch"]:
        raise ValueError(
            f"Checkpoint epoch is not the highest validation Macro-F1 epoch for "
            f"{config['model']}"
        )
    if not np.isclose(
        best_history_epoch["validation_macro_f1"],
        metrics["best_validation_macro_f1"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Validation Macro-F1 checkpoint mismatch for {config['model']}")

    expected_params = config["source_run"]["source_parameter_count"]
    actual_params = metrics.get("trainable_parameters", metrics["parameter_count"])
    if actual_params != expected_params:
        raise ValueError(
            f"Trainable parameter mismatch for {config['model']}: "
            f"{actual_params} != {expected_params}"
        )

    metrics["trainable_parameters"] = actual_params
    entry["metrics"] = metrics


def render_report(entries: dict[str, dict[str, Any]]) -> None:
    split_manifests = {
        name: entry["config"]["data_split"] for name, entry in entries.items()
    }
    lines = [
        "# CIFAR-10 HybridCNN Pipeline Rerun",
        "",
        "Fresh-scratch reruns of the three selected CIFAR-10 HybridCNN configurations "
        "using the shared MedMNIST transfer training/evaluation implementation. "
        "These are single runs, not repeated trials for a new statistical comparison.",
        "",
        "## Data and protocol",
        "",
        "- Official CIFAR-10 train/test splits are used. Each selected run "
        "retains its historical random train/validation split seed.",
    ]
    for name, manifest in split_manifests.items():
        lines.append(
            f"- {name}: train/validation/test = "
            f"{manifest['train_size']}/{manifest['validation_size']}/"
            f"{manifest['test_size']}; split seed {manifest['split_seed']}."
        )
    lines.extend([
        "- CIFAR-10 test data is separate from the official training split and is "
        "evaluated only after selecting the checkpoint by validation Macro-F1.",
        "- Input: 32x32 RGB; ToTensor only; no normalization, augmentation, or "
        "pretrained weights.",
        "- Adam, per-run source learning rate, batch 32, maximum 50 epochs; "
        "ReduceLROnPlateau on validation Macro-F1 (factor 0.5, patience 4); "
        "early stopping patience 10.",
        "- Determinism: Python, NumPy, PyTorch and CUDA seeds set to 42; "
        "cuDNN deterministic enabled and benchmark disabled. CUDA kernels and "
        "software/hardware versions are recorded per run.",
        "",
        "## Results",
        "",
        "| Configuration | Status | Seed | Accuracy | Macro precision | Macro recall | "
        "Macro-F1 | Balanced accuracy | Test loss | Best epoch | Trainable params | "
        "Train seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in RUN_NAMES:
        entry = entries[name]
        metrics = entry.get("metrics", {})
        if not metrics:
            lines.append(
                f"| {name} | {entry['status']} | {SEED} | — | — | — | — | — | "
                f"— | — | — | — |"
            )
            continue
        lines.append(
            f"| {name} | {entry['status']} | {SEED} | {metrics['accuracy']:.6f} | "
            f"{metrics['macro_precision']:.6f} | {metrics['macro_recall']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['balanced_accuracy']:.6f} | "
            f"{metrics['loss']:.6f} | {metrics['best_epoch']} | "
            f"{metrics['trainable_parameters']} | "
            f"{metrics['total_training_seconds']:.2f} |"
        )
    lines.extend(["", "## Run artifacts", ""])
    for name in RUN_NAMES:
        entry = entries[name]
        lines.append(
            f"- {name}: `{entry['dir'].relative_to(PROJECT_ROOT)}` "
            f"({entry['status']})"
        )
    lines.extend(
        [
            "",
            "## Comparison with the historical selected-run summaries",
            "",
            "| Configuration | Metric | Historical | Rerun | Difference (rerun - historical) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name in RUN_NAMES:
        entry = entries[name]
        metrics = entry.get("metrics", {})
        historical = entry["config"]["source_run"]["historical_test_metrics"]
        for metric_name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            old_value = historical.get(metric_name)
            new_value = metrics.get(metric_name)
            if old_value is None or new_value is None:
                lines.append(f"| {name} | {metric_name} | — | — | — |")
            else:
                lines.append(
                    f"| {name} | {metric_name} | {old_value:.6f} | "
                    f"{new_value:.6f} | {new_value - old_value:+.6f} |"
                )
    lines.extend(
        [
            "",
            "Historical metrics are not used for selection. Differences can reflect "
            "the fresh random initialization and the transfer-pipeline preprocessing "
            "policy (ToTensor without the normalization used by the original "
            "CIFAR-10 loader). These are not repeated trials for a new statistical "
            "comparison.",
            "",
        ]
    )
    (OUTPUT_ROOT / "CIFAR10_RERUN_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def finalize_existing_outputs() -> None:
    import yaml

    study_path = OUTPUT_ROOT / "study.json"
    if not study_path.is_file():
        raise FileNotFoundError(f"Missing study metadata: {study_path}")
    with study_path.open("r", encoding="utf-8") as handle:
        study = json.load(handle)
    pipeline.set_global_seed(SEED)
    study["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    study["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    study["deterministic_algorithms"] = (
        torch.are_deterministic_algorithms_enabled()
    )
    write_json(study_path, study)

    entries = {}
    for name in RUN_NAMES:
        run_dir = OUTPUT_ROOT / "runs" / name
        required_files = (
            "config.yaml",
            "metrics.json",
            "history.json",
            "confusion_matrix.json",
            "best_checkpoint.pth",
            "run.log",
        )
        missing = [
            file_name for file_name in required_files
            if not (run_dir / file_name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot finalize {name}; missing artifacts: {', '.join(missing)}"
            )
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["trainable_parameters"] = metrics["parameter_count"]
        write_json(run_dir / "metrics.json", metrics)
        entry = {
            "config": config,
            "dir": run_dir,
            "status": "completed",
            "metrics": metrics,
        }
        validate_completed_run(entry, TEST_SIZE)

        write_json(
            run_dir / "experiment.json",
            {
                "experiment_id": config["experiment_id"],
                "status": "completed",
                "best_epoch": metrics["best_epoch"],
            },
        )
        write_json(
            run_dir / "run_metadata.json",
            {
                "seed": SEED,
                "data_split_seed": config["data_split_seed"],
                "source_run_seed": config["source_run"]["source_run_seed"],
                "status": "completed",
                "device": study["device"],
                "cuda_device_name": study["cuda_device_name"],
                "python_version": study["python_version"],
                "torch_version": study["torch_version"],
                "torchvision_version": study["torchvision_version"],
                "numpy_version": study["numpy_version"],
                "cudnn_deterministic": study["cudnn_deterministic"],
                "cudnn_benchmark": study["cudnn_benchmark"],
                "deterministic_algorithms": study["deterministic_algorithms"],
                "source_checkpoint_loaded": False,
                "checkpoint_sha256": pipeline.file_sha256(
                    run_dir / "best_checkpoint.pth"
                ),
                "total_training_seconds": metrics["total_training_seconds"],
                "test_evaluation_seconds": metrics["test_evaluation_seconds"],
                "historical_test_metrics": config["source_run"][
                    "historical_test_metrics"
                ],
            },
        )
        entries[name] = entry

    update_summaries(entries)
    render_report(entries)


def run() -> int:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing rerun outputs in {OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    source_configs = pipeline.source_configurations()
    verify_hyperparameters(source_configs)
    run_configs = build_run_configs(source_configs)
    official_train, official_test = load_cifar10()
    data_sets_by_run = {}
    split_manifests = {}
    split_indices_by_run = {}
    for name in RUN_NAMES:
        split_seed = int(source_configs[name]["source_run_seed"])
        data_sets, manifest, indices = make_cifar10_split(
            official_train, official_test, split_seed
        )
        data_sets_by_run[name] = data_sets
        split_manifests[name] = manifest
        split_indices_by_run[name] = indices
    write_json(OUTPUT_ROOT / "split_indices.json", split_indices_by_run)
    write_json(OUTPUT_ROOT / "split_manifest.json", split_manifests)

    pipeline.set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = {
        "study": "CIFAR-10 selected HybridCNN rerun",
        "dataset": "CIFAR10",
        "dataset_root": str(DATA_ROOT.relative_to(PROJECT_ROOT)),
        "official_train_size": 50_000,
        "official_test_size": 10_000,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "training_seed": SEED,
        "training_seed_policy": "Same seed 42 for each run, matching the MedMNIST pipeline.",
        "data_split_seeds": {
            name: manifest["split_seed"]
            for name, manifest in split_manifests.items()
        },
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
        "selected_source_runs": source_configs,
    }
    write_json(OUTPUT_ROOT / "study.json", metadata)

    entries = {}
    for name in RUN_NAMES:
        run_dir = OUTPUT_ROOT / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=False)
        config = run_configs[name]
        config["data_split"] = split_manifests[name]
        write_yaml(run_dir / "config.yaml", config)
        write_json(
            run_dir / "experiment.json",
            {"experiment_id": config["experiment_id"], "status": "planned"},
        )
        entries[name] = {
            "config": config,
            "dir": run_dir,
            "status": "planned",
            "metrics": {},
        }
    update_summaries(entries)
    render_report(entries)

    preflight = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in RUN_NAMES:
        try:
            result = pipeline.smoke_test(
                "CIFAR10",
                name,
                data_sets_by_run[name],
                10,
                source_configs[name],
                device,
                SEED,
            )
            result["class_names"] = list(CLASS_NAMES)
            expected_params = source_configs[name]["source_parameter_count"]
            if result["parameter_count"] != expected_params:
                raise ValueError(
                    f"Smoke parameter count {result['parameter_count']} != "
                    f"source architecture count {expected_params}"
                )
            preflight.append(result)
        except Exception as error:
            preflight.append(
                {
                    "status": "failed",
                    "dataset": "CIFAR10",
                    "model": name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    write_json(OUTPUT_ROOT / "preflight.json", {"checks": preflight})
    failed_preflight = [check for check in preflight if check["status"] != "passed"]
    if failed_preflight:
        for check in failed_preflight:
            name = check["model"]
            entries[name]["status"] = "failed"
            entries[name]["error"] = check["error"]
            write_json(
                entries[name]["dir"] / "experiment.json",
                {
                    "experiment_id": entries[name]["config"]["experiment_id"],
                    "status": "failed",
                    "phase": "preflight",
                    "error": check["error"],
                },
            )
        update_summaries(entries)
        render_report(entries)
        return 1

    for name in RUN_NAMES:
        entry = entries[name]
        config = entry["config"]
        run_dir = entry["dir"]
        entry["status"] = "running"
        write_json(
            run_dir / "experiment.json",
            {"experiment_id": config["experiment_id"], "status": "running"},
        )
        update_summaries(entries)
        started = time.perf_counter()
        try:
            result_metrics = pipeline.train_experiment(
                config,
                run_dir,
                data_sets_by_run[name],
                10,
                source_configs[name],
                device,
                epochs=50,
            )
            entry["metrics"] = result_metrics
            entry["metrics"]["trainable_parameters"] = result_metrics[
                "parameter_count"
            ]
            matrix_path = run_dir / "confusion_matrix.json"
            with matrix_path.open("r", encoding="utf-8") as handle:
                matrix_payload = json.load(handle)
            matrix_payload["class_names"] = list(CLASS_NAMES)
            write_json(matrix_path, matrix_payload)
            validate_completed_run(entry, TEST_SIZE)

            checkpoint_path = run_dir / "best_checkpoint.pth"
            checkpoint_digest = pipeline.file_sha256(checkpoint_path)
            run_metadata = {
                "seed": SEED,
                "data_split_seed": config["data_split_seed"],
                "source_run_seed": source_configs[name]["source_run_seed"],
                "status": "completed",
                "device": str(device),
                "cuda_device_name": metadata["cuda_device_name"],
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "numpy_version": np.__version__,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "source_checkpoint_loaded": False,
                "checkpoint_sha256": checkpoint_digest,
                "wall_seconds_including_test_evaluation": time.perf_counter() - started,
                "historical_test_metrics": source_configs[name][
                    "historical_test_metrics"
                ],
            }
            write_json(run_dir / "run_metadata.json", run_metadata)
            entry["status"] = "completed"
        except Exception as error:
            entry["status"] = "failed"
            entry["error"] = f"{type(error).__name__}: {error}"
            write_json(
                run_dir / "experiment.json",
                {
                    "experiment_id": config["experiment_id"],
                    "status": "failed",
                    "error": entry["error"],
                },
            )
        update_summaries(entries)
        render_report(entries)

    return 0 if all(entry["status"] == "completed" for entry in entries.values()) else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize-existing"]:
        finalize_existing_outputs()
    elif sys.argv[1:]:
        raise SystemExit("Usage: run_cifar10_hybrid_rerun.py [--finalize-existing]")
    else:
        raise SystemExit(run())
