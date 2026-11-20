import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiment_runner
from config import config_to_dict, load_config
from optimizers.igwo_random_local import (
    RANDOM_LOCAL_SEARCH_METHOD,
    RandomLocalSearchIGWO,
)
from scripts.igwo_random_local_common import (
    DEFAULT_CONFIG,
    RANDOM_LOCAL_ROOT,
    SEEDS,
    require_all_adaptive_runs_completed,
    read_json,
    SHARED_GPU_LOCK,
)
from utils import setup_logging, write_json


POPULATION_SIZE = 8
ITERATION_COUNT = 15
LOCAL_SEARCH_CANDIDATES = POPULATION_SIZE


def _exclusive_gpu_lock(lock_path, logger):
    import msvcrt

    class LockContext:
        def __enter__(self):
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_file = lock_path.open("a+b")
            self.lock_file.seek(0, 2)
            if self.lock_file.tell() == 0:
                self.lock_file.write(b"\0")
                self.lock_file.flush()
            while True:
                try:
                    self.lock_file.seek(0)
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    logger.info("waiting for exclusive GPU slot: %s", lock_path)
                    time.sleep(5)
            logger.info("acquired exclusive GPU slot: %s", lock_path)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.lock_file.seek(0)
            msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            self.lock_file.close()
            return False

    return LockContext()


def run_seed(config_path, study_dir, seed, gpu_lock_path):
    require_all_adaptive_runs_completed()
    if study_dir.exists():
        raise FileExistsError(
            f"{study_dir} already exists; refusing to overwrite an experiment"
        )

    config = load_config(str(config_path))
    config.random_seed = seed
    config.population_size = POPULATION_SIZE
    config.iteration_count = ITERATION_COUNT
    config.runs = 1
    config.optimizers = ["igwo"]
    config.results_dir = str(study_dir.resolve())
    study_dir.mkdir(parents=True)

    run_dir = study_dir / "CIFAR10" / "IGWO" / "run_01"
    run_dir.mkdir(parents=True)
    logger = setup_logging(str(run_dir / "training.log"))
    manifest_path = study_dir / "study_manifest.json"
    manifest = {
        "study": "IGWO 8x15 Random Local Search",
        "seed": seed,
        "status": "running",
        "population_size": POPULATION_SIZE,
        "iterations": ITERATION_COUNT,
        "local_search_candidates_per_iteration": LOCAL_SEARCH_CANDIDATES,
        "local_search": {
            "method": RANDOM_LOCAL_SEARCH_METHOD,
            "candidate_generation": {
                "continuous": "uniform over the configured coordinate bounds",
                "learning_rate": "uniform over existing log10 coordinates",
                "integer": "uniform over valid integer values",
                "categorical": "uniform over valid categories",
            },
            "parameters_changed_per_candidate": 1,
            "retry_limit": 64,
        },
        "cache_key": "full mapped hyperparameter configuration",
        "source_config": str(config_path.resolve()),
        "source_adaptive_study": str(
            (RANDOM_LOCAL_ROOT.parent / "paper_config_6x15_adaptive").resolve()
        ),
        "shared_gpu_lock": str(gpu_lock_path.resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "config": config_to_dict(config),
    }
    write_json(str(manifest_path), manifest)
    write_json(str(study_dir / "config_used.json"), config_to_dict(config))
    logger.info(
        "starting %s seed=%s population=%s iterations=%s local_candidates=%s",
        manifest["study"],
        seed,
        POPULATION_SIZE,
        ITERATION_COUNT,
        LOCAL_SEARCH_CANDIDATES,
    )

    try:
        experiment_runner.OPTIMIZER_REGISTRY["igwo"] = RandomLocalSearchIGWO
        experiment_runner.DEFAULT_LOCAL_SEARCH_CANDIDATES = (
            LOCAL_SEARCH_CANDIDATES
        )
        with _exclusive_gpu_lock(gpu_lock_path, logger):
            outcome = experiment_runner.run_single_experiment(
                config,
                "igwo",
                0,
                logger,
            )
        if outcome["summary"] is None:
            raise RuntimeError("Random-local IGWO returned no run summary")

        summary_path = run_dir / "summary.json"
        summary = read_json(summary_path)
        summary["study_name"] = manifest["study"]
        summary["local_search_method"] = RANDOM_LOCAL_SEARCH_METHOD
        summary["local_search_candidates_per_iteration"] = (
            LOCAL_SEARCH_CANDIDATES
        )
        write_json(str(summary_path), summary)

        trace_path = run_dir / "igwo_trace.json"
        trace = read_json(trace_path)
        trace["local_search_method"] = RANDOM_LOCAL_SEARCH_METHOD
        trace["local_search_candidates_per_iteration"] = (
            LOCAL_SEARCH_CANDIDATES
        )
        write_json(str(trace_path), trace)

        manifest["status"] = "completed"
        manifest["summary"] = str(summary_path.resolve())
        manifest["completed_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        write_json(str(manifest_path), manifest)
        logger.info("random-local IGWO seed=%s completed: %s", seed, study_dir)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(str(manifest_path), manifest)
        logger.exception("random-local IGWO seed=%s failed", seed)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run one IGWO 8x15 random-local-search seed."
    )
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, default=SHARED_GPU_LOCK)
    args = parser.parse_args()
    run_seed(args.config, args.study_dir, args.seed, args.gpu_lock)


if __name__ == "__main__":
    main()
