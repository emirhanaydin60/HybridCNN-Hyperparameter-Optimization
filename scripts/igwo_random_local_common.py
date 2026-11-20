import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
ADAPTIVE_ROOT = (
    ROOT / "results" / "igwo_comparison" / "paper_config_6x15_adaptive"
)
RANDOM_LOCAL_ROOT = (
    ROOT / "results" / "igwo_comparison" / "paper_config_8x15_random_local"
)
DEFAULT_CONFIG = ROOT / "results" / "main_experiment" / "config_used.json"
SHARED_GPU_LOCK = ADAPTIVE_ROOT / "gpu_training.lock"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _failure(seed, message):
    return {"seed": seed, "status": "failed", "error": message}


def inspect_adaptive_seed(seed):
    seed_root = ADAPTIVE_ROOT / f"seed_{seed}"
    manifest_path = seed_root / "study_manifest.json"
    if not manifest_path.is_file():
        return {"seed": seed, "status": "pending", "error": "manifest missing"}

    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        return _failure(seed, f"invalid manifest: {error}")

    if manifest.get("seed") != seed:
        return _failure(seed, "manifest seed does not match expected seed")
    if manifest.get("status") == "failed":
        return _failure(seed, manifest.get("error", "adaptive run marked failed"))
    if manifest.get("status") != "completed":
        return {"seed": seed, "status": "pending", "error": "run not completed"}

    manifest_config = manifest.get("igwo_config", {})
    if (
        manifest_config.get("population_size") != 6
        or manifest_config.get("iteration_count") != 15
        or manifest.get("local_search_candidates_per_iteration") != 2
    ):
        return _failure(seed, "adaptive manifest has unexpected experiment settings")

    run_root = seed_root / "CIFAR10" / "IGWO" / "run_01"
    required_files = (
        run_root / "summary.json",
        run_root / "metrics.json",
        run_root / "history.json",
        run_root / "confusion_matrix.json",
        run_root / "igwo_trace.json",
        run_root / "config_used.json",
        run_root / "best_model.pth",
        run_root / "training.log",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        return _failure(seed, f"required final outputs missing: {missing}")
    if any(path.stat().st_size == 0 for path in required_files):
        return _failure(seed, "one or more required final outputs are empty")

    try:
        summary = read_json(run_root / "summary.json")
        trace = read_json(run_root / "igwo_trace.json")
        confusion = read_json(run_root / "confusion_matrix.json")
    except (OSError, json.JSONDecodeError) as error:
        return _failure(seed, f"invalid final output: {error}")

    config = summary.get("config", {})
    if (
        summary.get("algorithm") != "igwo"
        or summary.get("run_seed") != seed
        or config.get("population_size") != 6
        or config.get("iteration_count") != 15
        or summary.get("local_search_candidates_per_iteration") != 2
    ):
        return _failure(seed, "final summary does not match the adaptive run")
    for metric in (
        "best_validation_accuracy",
        "best_test_accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "test_loss",
        "runtime_seconds",
    ):
        value = summary.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return _failure(seed, f"final summary has invalid {metric}")

    matrix = confusion.get("matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 10
        or any(not isinstance(row, list) or len(row) != 10 for row in matrix)
        or sum(sum(row) for row in matrix) != 10000
    ):
        return _failure(seed, "final confusion matrix is not a 10x10 CIFAR-10 matrix")
    if (
        trace.get("configured_iterations") != 15
        or trace.get("local_search_candidates_per_iteration") != 2
    ):
        return _failure(seed, "final IGWO trace has unexpected settings")

    return {
        "seed": seed,
        "status": "completed",
        "manifest": str(manifest_path),
        "summary": str(run_root / "summary.json"),
        "checkpoint": str(run_root / "best_model.pth"),
    }


def inspect_all_adaptive_runs():
    return [inspect_adaptive_seed(seed) for seed in SEEDS]


def require_all_adaptive_runs_completed():
    results = inspect_all_adaptive_runs()
    failures = [item for item in results if item["status"] == "failed"]
    pending = [item for item in results if item["status"] != "completed"]
    if failures or pending:
        raise RuntimeError(
            "Adaptive 6x15 runs are not all verified complete: "
            f"failures={failures}; pending={pending}"
        )
    return results
