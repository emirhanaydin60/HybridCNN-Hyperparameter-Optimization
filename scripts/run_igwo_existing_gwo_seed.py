import argparse
import csv
import hashlib
import json
import msvcrt
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import config_to_dict, load_config
from experiment_runner import build_bounds, position_signature, run_single_experiment
from optimizers.gwo import GreyWolfOptimizer
from utils import setup_logging, write_json


DEFAULT_CONFIG = ROOT / "results" / "main_experiment" / "config_used.json"
DEFAULT_GWO_RUNS = ROOT / "results" / "main_experiment" / "GWO"
DEFAULT_STUDY_ROOT = ROOT / "results" / "igwo_comparison" / "paper_config_4x15"
DEFAULT_SEED = 42
IGWO_POPULATION_SIZE = 4
IGWO_ITERATION_COUNT = 15
MATCHED_CONFIG_KEYS = (
    "dataset",
    "model_name",
    "population_size",
    "iteration_count",
    "search_epochs",
    "final_epochs",
    "batch_size",
    "learning_rate",
    "subset_size",
    "random_seed",
    "patience",
    "lr_reduce_factor",
    "lr_reduce_patience",
    "monitor_ratio",
    "train_size",
    "val_size",
    "num_workers",
    "data_dir",
    "search_space",
)


def _read_json(path):
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_baseline(config, baseline_path, seed):
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Existing GWO seed-{seed} summary not found: {baseline_path}"
        )
    baseline = _read_json(baseline_path)
    if baseline.get("algorithm") != "gwo" or baseline.get("run_seed") != seed:
        raise ValueError(f"Unexpected GWO seed-{seed} summary: {baseline_path}")
    baseline_config = baseline.get("config")
    if not isinstance(baseline_config, dict):
        raise ValueError("Existing GWO summary does not contain its run configuration")
    current_config = config_to_dict(config)
    mismatches = {
        key: {
            "gwo": baseline_config.get(key),
            "igwo": current_config.get(key),
        }
        for key in MATCHED_CONFIG_KEYS
        if baseline_config.get(key) != current_config.get(key)
    }
    if mismatches:
        raise ValueError(f"GWO and IGWO protocol settings differ: {mismatches}")
    if baseline.get("optimizer_state", {}).get("cache_misses") is None:
        raise ValueError("GWO summary is missing its unique training count")
    for metric in ("best_validation_accuracy", "f1", "runtime_seconds"):
        if baseline.get(metric) is None:
            raise ValueError(f"GWO summary is missing {metric}")
    baseline_packages = baseline.get("package_versions", {})
    current_packages = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
    }
    package_mismatches = {
        name: {"gwo": baseline_packages.get(name), "igwo": current}
        for name, current in current_packages.items()
        if baseline_packages.get(name) != current
    }
    if package_mismatches:
        raise ValueError(
            "Cannot verify matched starting conditions across software versions: "
            f"{package_mismatches}"
        )
    return baseline


def _reconstruct_gwo_initial_population(config, seed):
    bounds = build_bounds(config)
    optimizer = GreyWolfOptimizer(
        lambda position, **kwargs: 0.0,
        bounds,
        population=config.population_size,
        iterations=config.iteration_count,
        solution_signature_func=lambda position: position_signature(
            position, config
        ),
        random_seed=seed,
    )
    return optimizer._initialize().tolist()


def _check_igwo_run(
    study_dir, summary, expected_initial_positions, seed, evaluation_budget
):
    if summary["run_seed"] != seed:
        raise AssertionError(f"IGWO run did not use seed {seed}")
    if summary["initial_population_positions"] != expected_initial_positions:
        raise AssertionError(
            f"IGWO's raw initial positions differ from the seed-{seed} GWO "
            "population reconstructed from the saved search configuration"
        )
    if len(summary["convergence_history"]) != summary["config"]["iteration_count"]:
        raise AssertionError("IGWO did not complete all configured iterations")
    if summary["config"]["population_size"] != IGWO_POPULATION_SIZE:
        raise AssertionError("IGWO population size differs from the paper protocol")
    if summary["config"]["iteration_count"] != IGWO_ITERATION_COUNT:
        raise AssertionError("IGWO iteration count differs from the paper protocol")
    if summary["unique_fitness_evaluations"] != summary["optimizer_state"][
        "cache_misses"
    ]:
        raise AssertionError("IGWO unique fitness count does not match cache misses")
    if summary["optimizer_state"]["evaluation_budget"] != evaluation_budget:
        raise AssertionError("IGWO evaluation budget differs from its GWO reference")
    if summary["unique_fitness_evaluations"] > evaluation_budget:
        raise AssertionError("IGWO exceeded the unique-training budget")
    if summary["local_search_evaluation_count"] > (
        IGWO_POPULATION_SIZE * IGWO_ITERATION_COUNT
    ):
        raise AssertionError("IGWO local search exceeded its population-sized budget")
    trace_path = study_dir / "CIFAR10" / "IGWO" / "run_01" / "igwo_trace.json"
    trace = _read_json(trace_path)
    local_counts = trace["local_search_evaluations_by_iteration"]
    if any(count > IGWO_POPULATION_SIZE for count in local_counts):
        raise AssertionError("A local-search iteration exceeded the IGWO population")
    if sum(local_counts) != summary["local_search_evaluation_count"]:
        raise AssertionError("IGWO local-search trace count does not match its summary")

    confusion_path = study_dir / "CIFAR10" / "IGWO" / "run_01" / "confusion_matrix.json"
    confusion = _read_json(confusion_path)
    matrix = np.asarray(confusion["matrix"], dtype=np.int64)
    if matrix.shape != (10, 10) or int(matrix.sum()) != 10000:
        raise AssertionError("IGWO test confusion matrix must be 10x10 and total 10,000")

    diagonal = np.diag(matrix).astype(float)
    precision = diagonal / np.maximum(matrix.sum(axis=0), 1)
    recall = diagonal / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    checked = {
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    reported = {
        "accuracy": summary["best_test_accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
    }
    for metric, value in checked.items():
        if not np.isclose(value, reported[metric], rtol=0.0, atol=1e-12):
            raise AssertionError(f"IGWO {metric} differs from its test confusion matrix")
    if not np.isclose(reported["recall"], summary["balanced_accuracy"], atol=1e-12):
        raise AssertionError("IGWO balanced accuracy differs from macro-recall")


def _other_seed_runs(current_seed):
    return {
        str(seed): "separate_run"
        for seed in (42, 43, 44)
        if seed != current_seed
    }


@contextmanager
def _exclusive_gpu_lock(lock_path, logger):
    if lock_path is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0, 2)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        acquired = False
        while not acquired:
            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                logger.info("waiting for exclusive GPU slot: %s", lock_path)
                time.sleep(5)
        try:
            logger.info("acquired exclusive GPU slot: %s", lock_path)
            yield
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _write_comparison(
    study_dir, baseline, igwo, baseline_path, baseline_hash, seed
):
    baseline_state = baseline["optimizer_state"]
    comparison = {
        "seed": seed,
        "test_set_used_for_selection": False,
        "baseline": {
            "algorithm": "gwo",
            "summary_path": str(baseline_path.resolve()),
            "summary_sha256": baseline_hash,
            "test_accuracy": baseline["best_test_accuracy"],
            "test_macro_precision": baseline["precision"],
            "test_macro_recall": baseline["recall"],
            "test_macro_f1": baseline["f1"],
            "best_validation_accuracy": baseline["best_validation_accuracy"],
            "convergence_history": baseline["convergence_history"],
            "unique_fitness_evaluations": baseline_state["cache_misses"],
            "fitness_evaluations": baseline["fitness_evaluations"],
            "cache_hits": baseline_state["cache_hits"],
            "runtime_seconds": baseline["runtime_seconds"],
            "best_hyperparameters": baseline["best_hyperparameters"],
        },
        "igwo": {
            "algorithm": "igwo",
            "test_accuracy": igwo["best_test_accuracy"],
            "test_macro_precision": igwo["precision"],
            "test_macro_recall": igwo["recall"],
            "test_macro_f1": igwo["f1"],
            "best_validation_accuracy": igwo["best_validation_accuracy"],
            "best_validation_macro_f1": igwo["best_validation_metrics"][
                "macro_f1"
            ],
            "convergence_history": igwo["convergence_history"],
            "unique_fitness_evaluations": igwo["unique_fitness_evaluations"],
            "fitness_evaluations": igwo["fitness_evaluations"],
            "cache_hits": igwo["cache_hits"],
            "evaluation_budget": igwo["optimizer_state"]["evaluation_budget"],
            "local_search_evaluation_count": igwo[
                "local_search_evaluation_count"
            ],
            "local_search_global_best_improvements": igwo[
                "local_search_global_best_improvements"
            ],
            "incomplete_local_evaluations": igwo[
                "incomplete_local_evaluations"
            ],
            "runtime_seconds": igwo["runtime_seconds"],
            "best_hyperparameters": igwo["best_hyperparameters"],
        },
        "igwo_minus_gwo": {
            "test_macro_f1": igwo["f1"] - baseline["f1"],
            "best_validation_accuracy": (
                igwo["best_validation_accuracy"]
                - baseline["best_validation_accuracy"]
            ),
            "unique_fitness_evaluations": (
                igwo["unique_fitness_evaluations"]
                - baseline_state["cache_misses"]
            ),
            "runtime_seconds": igwo["runtime_seconds"] - baseline["runtime_seconds"],
            "evaluation_budget": igwo["optimizer_state"]["evaluation_budget"],
        },
        "interpretation": (
            "One matched seed is exploratory evidence, not a statistical "
            "superiority claim."
        ),
        "other_seed_runs": _other_seed_runs(seed),
    }
    write_json(str(study_dir / "comparison_summary.json"), comparison)

    csv_path = study_dir / "comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        fields = [
            "seed",
            "algorithm",
            "test_accuracy",
            "test_macro_precision",
            "test_macro_recall",
            "test_macro_f1",
            "best_validation_accuracy",
            "unique_fitness_evaluations",
            "fitness_evaluations",
            "cache_hits",
            "evaluation_budget",
            "local_search_evaluation_count",
            "local_search_global_best_improvements",
            "incomplete_local_evaluations",
            "runtime_seconds",
        ]
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for algorithm, result in (
            ("gwo", comparison["baseline"]),
            ("igwo", comparison["igwo"]),
        ):
            writer.writerow(
                {
                    "seed": seed,
                    "algorithm": algorithm,
                    "test_accuracy": result["test_accuracy"],
                    "test_macro_precision": result["test_macro_precision"],
                    "test_macro_recall": result["test_macro_recall"],
                    "test_macro_f1": result["test_macro_f1"],
                    "best_validation_accuracy": result[
                        "best_validation_accuracy"
                    ],
                    "unique_fitness_evaluations": result[
                        "unique_fitness_evaluations"
                    ],
                    "fitness_evaluations": result.get("fitness_evaluations"),
                    "cache_hits": result["cache_hits"],
                    "evaluation_budget": result.get("evaluation_budget"),
                    "local_search_evaluation_count": result.get(
                        "local_search_evaluation_count"
                    ),
                    "local_search_global_best_improvements": result.get(
                        "local_search_global_best_improvements"
                    ),
                    "incomplete_local_evaluations": result.get(
                        "incomplete_local_evaluations"
                    ),
                    "runtime_seconds": result["runtime_seconds"],
                }
            )

    figure, axis = plt.subplots(figsize=(8, 5))
    for label, result, color in (
        ("GWO (existing)", baseline, "#1f77b4"),
        ("IGWO", igwo, "#ff7f0e"),
    ):
        curve = result["convergence_history"]
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve,
            marker="o",
            label=label,
            color=color,
        )
    axis.set(
        title=f"Seed {seed} convergence (validation accuracy)",
        xlabel="Iteration",
        ylabel="Best validation accuracy",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(study_dir / f"convergence_seed{seed}.png", dpi=160)
    plt.close(figure)


def run_study(config_path, gwo_runs_dir, study_dir, seed, gpu_lock_path=None):
    config = load_config(str(config_path))
    config.random_seed = seed
    baseline_run = seed - 41
    baseline_path = gwo_runs_dir / f"run_{baseline_run:02d}" / "summary.json"
    baseline = _check_baseline(config, baseline_path, seed)
    if config.population_size != 8 or config.iteration_count != 15:
        raise ValueError("Archived GWO configuration must be population 8 and 15 iterations")
    expected_initial_positions = _reconstruct_gwo_initial_population(
        config, seed
    )[:IGWO_POPULATION_SIZE]
    baseline_hash = _sha256(baseline_path)
    evaluation_budget = int(baseline["optimizer_state"]["cache_misses"])

    config.population_size = IGWO_POPULATION_SIZE
    config.iteration_count = IGWO_ITERATION_COUNT
    config.runs = 1
    config.optimizers = ["igwo"]

    if study_dir.exists():
        raise FileExistsError(
            f"{study_dir} already exists; refusing to overwrite an existing study"
        )
    study_dir.mkdir(parents=True)
    config.results_dir = str(study_dir.resolve())
    study_manifest = {
        "study": f"IGWO CIFAR-10 matched to existing GWO seed {seed}",
        "seed": seed,
        "status": "running",
        "source_config": str(config_path.resolve()),
        "source_gwo_summary": str(baseline_path.resolve()),
        "source_gwo_summary_sha256": baseline_hash,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "igwo_config": config_to_dict(config),
        "evaluation_budget_unique_trainings": evaluation_budget,
        "standard_gwo_reference": {
            "population_size": 8,
            "iterations": 15,
            "unique_fitness_evaluations": evaluation_budget,
            "cache_hits": baseline["optimizer_state"]["cache_hits"],
        },
        "initial_population_verification": {
            "method": (
                f"first {IGWO_POPULATION_SIZE} positions from the reconstructed "
                f"GWO population (seed {seed})"
            ),
            "positions": expected_initial_positions,
        },
        "other_seed_runs": _other_seed_runs(seed),
    }
    write_json(str(study_dir / "study_manifest.json"), study_manifest)
    write_json(str(study_dir / "config_used.json"), config_to_dict(config))
    write_json(
        str(study_dir / "baseline_gwo_reference.json"),
        {
            "summary_path": str(baseline_path.resolve()),
            "summary_sha256": baseline_hash,
            "run_seed": baseline["run_seed"],
            "best_validation_accuracy": baseline["best_validation_accuracy"],
            "test_macro_f1": baseline["f1"],
            "unique_fitness_evaluations": baseline["optimizer_state"]["cache_misses"],
            "runtime_seconds": baseline["runtime_seconds"],
            "convergence_history": baseline["convergence_history"],
            "best_hyperparameters": baseline["best_hyperparameters"],
        },
    )

    run_dir = study_dir / "CIFAR10" / "IGWO" / "run_01"
    run_dir.mkdir(parents=True)
    logger = setup_logging(str(run_dir / "training.log"))
    logger.info(
        "starting IGWO run seed=%s population=%s iterations=%s unique_training_budget=%s; no GWO run will be started",
        seed,
        IGWO_POPULATION_SIZE,
        IGWO_ITERATION_COUNT,
        evaluation_budget,
    )
    try:
        with _exclusive_gpu_lock(gpu_lock_path, logger):
            outcome = run_single_experiment(
                config,
                "igwo",
                0,
                logger,
                evaluation_budget=evaluation_budget,
            )
        igwo_summary = outcome["summary"]
        if igwo_summary is None:
            raise RuntimeError("IGWO run returned no summary")
        _check_igwo_run(
            study_dir,
            igwo_summary,
            expected_initial_positions,
            seed,
            evaluation_budget,
        )
        if _sha256(baseline_path) != baseline_hash:
            raise AssertionError("Historical GWO summary changed during the IGWO run")
        _write_comparison(
            study_dir, baseline, igwo_summary, baseline_path, baseline_hash, seed
        )
    except Exception as error:
        study_manifest["status"] = "failed"
        study_manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(str(study_dir / "study_manifest.json"), study_manifest)
        raise
    study_manifest["status"] = "completed"
    write_json(str(study_dir / "study_manifest.json"), study_manifest)
    print(f"Seed-{seed} IGWO run and comparison completed: {study_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run one matched-budget IGWO seed; defaults to seed 42."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gwo-runs-dir", type=Path, default=DEFAULT_GWO_RUNS)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=DEFAULT_SEED)
    parser.add_argument("--study-dir", type=Path, default=None)
    parser.add_argument("--gpu-lock", type=Path, default=None)
    args = parser.parse_args()
    study_dir = args.study_dir or DEFAULT_STUDY_ROOT / f"seed_{args.seed}"
    run_study(args.config, args.gwo_runs_dir, study_dir, args.seed, args.gpu_lock)


if __name__ == "__main__":
    main()
