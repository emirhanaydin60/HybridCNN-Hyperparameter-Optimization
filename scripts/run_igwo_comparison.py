import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import config_to_dict, load_config
from experiment_runner import build_model, run_single_experiment
from utils import read_json, setup_logging, write_json


DEFAULT_CONFIG = ROOT / "results" / "main_experiment" / "config_used.json"
DEFAULT_OUTPUT = ROOT / "results" / "igwo_comparison"
EXPECTED_SETTINGS = {
    "dataset": "cifar10",
    "population_size": 8,
    "iteration_count": 15,
    "search_epochs": 6,
    "final_epochs": 50,
    "random_seed": 42,
    "runs": 3,
    "train_size": 45000,
    "val_size": 5000,
    "patience": 10,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 4,
}


def _canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _validate_config(config):
    actual = config_to_dict(config)
    differences = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in EXPECTED_SETTINGS.items()
        if actual.get(key) != value
    }
    if differences:
        raise ValueError(f"Comparison requires the recorded CIFAR-10 settings: {differences}")
    if config.subset_size is not None:
        raise ValueError("The comparison must use the full CIFAR-10 training split")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")


def _new_or_existing_manifest(output_dir, source_config, config):
    settings = config_to_dict(config)
    settings.pop("results_dir", None)
    settings.pop("optimizers", None)
    protocol_hash = _canonical_hash(
        {
            "source_config": str(source_config.resolve()),
            "settings": settings,
            "seeds": [config.random_seed + index for index in range(config.runs)],
        }
    )
    manifest_path = output_dir / "comparison_manifest.json"
    if output_dir.exists():
        if not manifest_path.exists():
            raise FileExistsError(
                f"{output_dir} already exists without a matching manifest; "
                "refusing to overwrite existing files"
            )
        manifest = read_json(str(manifest_path))
        if manifest.get("protocol_hash") != protocol_hash:
            raise ValueError(
                f"Existing study at {output_dir} uses a different protocol"
            )
        return manifest, manifest_path

    output_dir.mkdir(parents=True)
    manifest = {
        "study": "IGWO CIFAR-10 comparison",
        "protocol_hash": protocol_hash,
        "source_config": str(source_config.resolve()),
        "source_revision": _git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seeds": [config.random_seed + index for index in range(config.runs)],
        "settings": settings,
        "protocol": {
            "equal_iterations": {
                "algorithms": ["gwo", "igwo"],
                "population_size": config.population_size,
                "iterations": config.iteration_count,
                "search_epochs": config.search_epochs,
            },
            "equal_fitness_budget": {
                "algorithm": "igwo",
                "budget_source": "same-seed GWO cache_misses in equal_iterations",
                "budget_unit": "unique candidate model trainings",
                "maximum_iterations": config.iteration_count,
            },
            "fitness": "validation accuracy",
            "test_set_used_for_selection": False,
        },
        "phases": {
            "equal_iterations_gwo": "pending",
            "equal_iterations_igwo": "pending",
            "equal_fitness_budget_igwo": "pending",
        },
    }
    write_json(str(output_dir / "protocol_config.json"), config_to_dict(config))
    write_json(str(manifest_path), manifest)
    return manifest, manifest_path


def _save_manifest(manifest, manifest_path):
    write_json(str(manifest_path), manifest)


def _run_is_complete(run_dir, expected_seed, expected_budget=None):
    summary_path = run_dir / "summary.json"
    if not run_dir.exists():
        return False
    if not summary_path.exists():
        if any(run_dir.iterdir()):
            raise FileExistsError(
                f"Incomplete run output exists at {run_dir}; refusing to overwrite it"
            )
        return False
    summary = read_json(str(summary_path))
    if summary.get("run_seed") != expected_seed:
        raise ValueError(f"Seed mismatch in existing run: {run_dir}")
    if expected_budget is not None and summary.get("optimizer_state", {}).get(
        "evaluation_budget"
    ) != expected_budget:
        raise ValueError(f"Evaluation budget mismatch in existing run: {run_dir}")
    required_files = [
        "training.log",
        "best_model.pth",
        "metrics.json",
        "history.json",
        "confusion_matrix.json",
    ]
    if run_dir.parent.name.lower() == "igwo":
        required_files.append("igwo_trace.json")
    for filename in required_files:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(f"Completed run is missing {filename}: {run_dir}")
    return True


def _execute_phase(config, output_dir, algorithm, phase_key, manifest, manifest_path, budgets=None):
    phase_root = output_dir / phase_key
    phase_config = copy.deepcopy(config)
    phase_config.results_dir = str(phase_root)
    phase_config.optimizers = [algorithm]
    manifest["phases"][phase_key] = "running"
    _save_manifest(manifest, manifest_path)

    summaries = []
    for index in range(config.runs):
        seed = config.random_seed + index
        budget = budgets[index] if budgets is not None else None
        run_dir = (
            phase_root
            / config.dataset.upper()
            / algorithm.upper()
            / f"run_{index + 1:02d}"
        )
        if _run_is_complete(run_dir, seed, budget):
            summary = read_json(str(run_dir / "summary.json"))
            summaries.append(summary)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(str(run_dir / "training.log"))
        logger.info(
            "study_phase=%s algorithm=%s seed=%s evaluation_budget=%s",
            phase_key,
            algorithm,
            seed,
            budget,
        )
        outcome = run_single_experiment(
            phase_config,
            algorithm,
            index,
            logger,
            evaluation_budget=budget,
        )
        summary = outcome["summary"]
        if summary is None:
            raise RuntimeError(f"Runner returned no summary for {run_dir}")
        summaries.append(summary)

    _validate_phase(phase_root, algorithm, summaries, config, budgets)
    phase_summary = {
        "phase": phase_key,
        "algorithm": algorithm,
        "runs": summaries,
        "seeds": [item["run_seed"] for item in summaries],
    }
    write_json(
        str(phase_root / config.dataset.upper() / algorithm.upper() / "phase_summary.json"),
        phase_summary,
    )
    manifest["phases"][phase_key] = "completed"
    _save_manifest(manifest, manifest_path)
    return summaries


def _validate_phase(phase_root, algorithm, summaries, config, budgets=None):
    if len(summaries) != config.runs:
        raise RuntimeError(f"{algorithm} produced {len(summaries)} of {config.runs} runs")
    expected_test_total = 10000
    for index, summary in enumerate(summaries):
        seed = config.random_seed + index
        if summary["run_seed"] != seed:
            raise AssertionError(f"Unexpected seed in {algorithm} run {index + 1}")
        run_dir = (
            phase_root
            / config.dataset.upper()
            / algorithm.upper()
            / f"run_{index + 1:02d}"
        )
        matrix_data = read_json(str(run_dir / "confusion_matrix.json"))
        matrix = np.asarray(matrix_data["matrix"], dtype=np.int64)
        if matrix.shape != (10, 10) or int(matrix.sum()) != expected_test_total:
            raise AssertionError(f"Invalid test confusion matrix in {run_dir}")

        row_totals = matrix.sum(axis=1)
        col_totals = matrix.sum(axis=0)
        diagonal = np.diag(matrix).astype(float)
        precision_by_class = diagonal / np.maximum(col_totals, 1)
        recall_by_class = diagonal / np.maximum(row_totals, 1)
        f1_by_class = (
            2 * precision_by_class * recall_by_class
            / np.maximum(precision_by_class + recall_by_class, 1e-12)
        )
        recalculated = {
            "accuracy": float(diagonal.sum() / matrix.sum()),
            "macro_precision": float(precision_by_class.mean()),
            "macro_recall": float(recall_by_class.mean()),
            "macro_f1": float(f1_by_class.mean()),
            "balanced_accuracy": float(recall_by_class.mean()),
        }
        reported = {
            "accuracy": summary["best_test_accuracy"],
            "macro_precision": summary["precision"],
            "macro_recall": summary["recall"],
            "macro_f1": summary["f1"],
            "balanced_accuracy": summary["balanced_accuracy"],
        }
        for metric, actual in recalculated.items():
            if not math.isclose(actual, reported[metric], rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(
                    f"{metric} does not match confusion matrix in {run_dir}: "
                    f"{actual} != {reported[metric]}"
                )
        if not math.isclose(
            reported["macro_recall"],
            reported["balanced_accuracy"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Balanced accuracy mismatch in {run_dir}")

        model = build_model(
            summary["best_hyperparameters"],
            in_channels=3,
            img_size=32,
            num_classes=10,
        )
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if parameter_count != summary["trainable_parameter_count"]:
            raise AssertionError(f"Parameter count mismatch in {run_dir}")

        if algorithm == "igwo" and budgets is not None:
            expected_budget = budgets[index]
            if summary["unique_fitness_evaluations"] != expected_budget:
                raise AssertionError(
                    f"IGWO used {summary['unique_fitness_evaluations']} unique "
                    f"evaluations; expected {expected_budget} for seed {seed}"
                )
            if summary["optimizer_state"]["evaluation_budget"] != expected_budget:
                raise AssertionError(f"Recorded budget mismatch in {run_dir}")
        if len(summary["initial_population_signatures"]) != config.population_size:
            raise AssertionError(f"Initial population size mismatch in {run_dir}")


def _write_comparison_csv(output_dir, phase_rows):
    path = output_dir / "comparison_summary.csv"
    fieldnames = [
        "phase",
        "algorithm",
        "seed",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "test_loss",
        "best_validation_accuracy",
        "best_epoch",
        "trainable_parameter_count",
        "unique_fitness_evaluations",
        "cache_hits",
        "search_time_seconds",
        "final_training_time_seconds",
        "runtime_seconds",
        "evaluation_budget",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for phase, algorithm, summary in phase_rows:
            writer.writerow(
                {
                    "phase": phase,
                    "algorithm": algorithm,
                    "seed": summary["run_seed"],
                    "test_accuracy": summary["best_test_accuracy"],
                    "macro_precision": summary["precision"],
                    "macro_recall": summary["recall"],
                    "macro_f1": summary["f1"],
                    "balanced_accuracy": summary["balanced_accuracy"],
                    "test_loss": summary["test_loss"],
                    "best_validation_accuracy": summary["best_validation_accuracy"],
                    "best_epoch": summary["best_final_monitor_epoch"],
                    "trainable_parameter_count": summary["trainable_parameter_count"],
                    "unique_fitness_evaluations": summary["unique_fitness_evaluations"],
                    "cache_hits": summary["cache_hits"],
                    "search_time_seconds": summary["total_search_time_seconds"],
                    "final_training_time_seconds": summary[
                        "final_training_time_seconds"
                    ],
                    "runtime_seconds": summary["runtime_seconds"],
                    "evaluation_budget": summary["optimizer_state"].get(
                        "evaluation_budget"
                    ),
                }
            )


def run_study(config_path, output_dir):
    config = load_config(str(config_path))
    _validate_config(config)
    output_dir = output_dir.resolve()
    manifest, manifest_path = _new_or_existing_manifest(
        output_dir, config_path, config
    )
    phase_rows = []

    try:
        gwo_runs = _execute_phase(
            config,
            output_dir,
            "gwo",
            "equal_iterations_gwo",
            manifest,
            manifest_path,
        )
        igwo_iteration_runs = _execute_phase(
            config,
            output_dir,
            "igwo",
            "equal_iterations_igwo",
            manifest,
            manifest_path,
        )
        for gwo_summary, igwo_summary in zip(gwo_runs, igwo_iteration_runs):
            if (
                gwo_summary["initial_population_signatures"]
                != igwo_summary["initial_population_signatures"]
                or gwo_summary["initial_population_positions"]
                != igwo_summary["initial_population_positions"]
            ):
                raise AssertionError(
                    f"Initial GWO populations differ for seed {gwo_summary['run_seed']}"
                )
            if len(gwo_summary["convergence_history"]) != config.iteration_count:
                raise AssertionError("GWO iteration count differs from protocol")
            if len(igwo_summary["convergence_history"]) != config.iteration_count:
                raise AssertionError("IGWO iteration count differs from protocol")

        budgets = [item["unique_fitness_evaluations"] for item in gwo_runs]
        budget_runs = _execute_phase(
            config,
            output_dir,
            "igwo",
            "equal_fitness_budget_igwo",
            manifest,
            manifest_path,
            budgets=budgets,
        )
        for reference, budget_summary in zip(gwo_runs, budget_runs):
            if (
                reference["initial_population_signatures"]
                != budget_summary["initial_population_signatures"]
                or reference["initial_population_positions"]
                != budget_summary["initial_population_positions"]
            ):
                raise AssertionError(
                    f"Initial population differs in budget run for seed "
                    f"{reference['run_seed']}"
                )

        phase_rows.extend(
            ("equal_iterations", algorithm, summary)
            for algorithm, runs in (
                ("gwo", gwo_runs),
                ("igwo", igwo_iteration_runs),
            )
            for summary in runs
        )
        phase_rows.extend(
            ("equal_fitness_budget", "igwo", summary) for summary in budget_runs
        )
        _write_comparison_csv(output_dir, phase_rows)
        manifest["status"] = "completed"
        _save_manifest(manifest, manifest_path)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _save_manifest(manifest, manifest_path)
        raise

    print(f"IGWO comparison runs completed: {output_dir}")
    print("Next: python scripts/analyze_igwo_comparison.py")


def main():
    parser = argparse.ArgumentParser(
        description="Run matched-seed GWO/IGWO CIFAR-10 comparisons."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_study(args.config, args.output_dir)


if __name__ == "__main__":
    main()
