import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = ROOT / "results" / "igwo_comparison" / "paper_config_6x10"
SEEDS = (42, 43, 44)
METRICS = {
    "validation_accuracy": {
        "baseline": "best_validation_accuracy",
        "igwo": "best_validation_accuracy",
        "higher_is_better": True,
    },
    "test_accuracy": {
        "baseline": "test_accuracy",
        "igwo": "test_accuracy",
        "higher_is_better": True,
    },
    "test_precision": {
        "baseline": "test_macro_precision",
        "igwo": "test_macro_precision",
        "higher_is_better": True,
    },
    "test_recall": {
        "baseline": "test_macro_recall",
        "igwo": "test_macro_recall",
        "higher_is_better": True,
    },
    "test_f1": {
        "baseline": "test_macro_f1",
        "igwo": "test_macro_f1",
        "higher_is_better": True,
    },
    "unique_fitness_evaluations": {
        "baseline": "unique_fitness_evaluations",
        "igwo": "unique_fitness_evaluations",
        "higher_is_better": False,
    },
    "cache_hits": {
        "baseline": "cache_hits",
        "igwo": "cache_hits",
        "higher_is_better": True,
    },
    "runtime_seconds": {
        "baseline": "runtime_seconds",
        "igwo": "runtime_seconds",
        "higher_is_better": False,
    },
}


def _read_json(path):
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _write_json(path, payload):
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_seed(study_root, seed):
    seed_root = study_root / f"seed_{seed}"
    manifest_path = seed_root / "study_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing seed-{seed} manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise RuntimeError(
            f"Seed {seed} is {manifest.get('status')}; aggregate analysis "
            "requires all three completed runs"
        )
    comparison = _read_json(seed_root / "comparison_summary.json")
    baseline_path = Path(comparison["baseline"]["summary_path"])
    if _sha256(baseline_path) != comparison["baseline"]["summary_sha256"]:
        raise AssertionError(f"Historical GWO seed-{seed} summary changed")
    igwo_summary = _read_json(
        seed_root / "CIFAR10" / "IGWO" / "run_01" / "summary.json"
    )
    if igwo_summary["run_seed"] != seed:
        raise AssertionError(f"IGWO run seed mismatch in {seed_root}")
    if igwo_summary["unique_fitness_evaluations"] > comparison["igwo"][
        "evaluation_budget"
    ]:
        raise AssertionError(f"IGWO exceeded its budget for seed {seed}")
    return {
        "seed": seed,
        "manifest": manifest,
        "comparison": comparison,
        "igwo_summary": igwo_summary,
    }


def _stats(values, higher_is_better):
    values = [float(value) for value in values]
    best_index = max(range(len(values)), key=values.__getitem__)
    worst_index = min(range(len(values)), key=values.__getitem__)
    if not higher_is_better:
        best_index, worst_index = worst_index, best_index
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "best_seed": SEEDS[best_index],
        "best_value": values[best_index],
        "worst_seed": SEEDS[worst_index],
        "worst_value": values[worst_index],
    }


def _build_report(rows):
    report = {
        "study": "CIFAR-10 IGWO paper parameters (population 6, iterations 10)",
        "seeds": list(SEEDS),
        "interpretation": (
            "Descriptive results only; three seeds do not establish statistical "
            "superiority."
        ),
        "metrics": {},
        "paired_igwo_minus_gwo": {},
        "local_search_contribution": {},
        "per_seed": {},
    }
    for metric, definition in METRICS.items():
        baseline_values = [
            row["comparison"]["baseline"][definition["baseline"]] for row in rows
        ]
        igwo_values = [
            row["comparison"]["igwo"][definition["igwo"]] for row in rows
        ]
        differences = [
            igwo_value - baseline_value
            for igwo_value, baseline_value in zip(igwo_values, baseline_values)
        ]
        report["metrics"][metric] = {
            "gwo": _stats(baseline_values, definition["higher_is_better"]),
            "igwo": _stats(igwo_values, definition["higher_is_better"]),
        }
        report["paired_igwo_minus_gwo"][metric] = {
            "mean": statistics.mean(differences),
            "sample_std": (
                statistics.stdev(differences) if len(differences) > 1 else 0.0
            ),
            "by_seed": {
                str(row["seed"]): difference
                for row, difference in zip(rows, differences)
            },
        }

    report["local_search_contribution"] = {
        "unique_local_candidates": _stats(
            [
                row["igwo_summary"]["local_search_evaluation_count"]
                for row in rows
            ],
            higher_is_better=True,
        ),
        "global_best_improvements": _stats(
            [
                row["igwo_summary"]["local_search_global_best_improvements"]
                for row in rows
            ],
            higher_is_better=True,
        ),
        "incomplete_local_evaluations": {
            str(row["seed"]): row["igwo_summary"]["incomplete_local_evaluations"]
            for row in rows
        },
    }

    for row in rows:
        baseline = row["comparison"]["baseline"]
        igwo = row["comparison"]["igwo"]
        report["per_seed"][str(row["seed"])] = {
            "gwo": baseline,
            "igwo": igwo,
            "local_search_global_best_improvements": row["igwo_summary"][
                "local_search_global_best_improvements"
            ],
            "local_search_evaluation_count": row["igwo_summary"][
                "local_search_evaluation_count"
            ],
            "incomplete_local_evaluations": row["igwo_summary"][
                "incomplete_local_evaluations"
            ],
        }
    return report


def _write_csv(study_root, rows):
    path = study_root / "aggregate_metrics.csv"
    fields = [
        "seed",
        "algorithm",
        "validation_accuracy",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "unique_fitness_evaluations",
        "fitness_evaluations",
        "cache_hits",
        "evaluation_budget",
        "runtime_seconds",
        "local_search_evaluation_count",
        "local_search_global_best_improvements",
        "incomplete_local_evaluations",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            baseline = row["comparison"]["baseline"]
            igwo = row["comparison"]["igwo"]
            for algorithm, result in (("gwo", baseline), ("igwo", igwo)):
                writer.writerow(
                    {
                        "seed": row["seed"],
                        "algorithm": algorithm,
                        "validation_accuracy": result["best_validation_accuracy"],
                        "test_accuracy": result["test_accuracy"],
                        "test_macro_precision": result["test_macro_precision"],
                        "test_macro_recall": result["test_macro_recall"],
                        "test_macro_f1": result["test_macro_f1"],
                        "unique_fitness_evaluations": result[
                            "unique_fitness_evaluations"
                        ],
                        "fitness_evaluations": result.get("fitness_evaluations"),
                        "cache_hits": result["cache_hits"],
                        "evaluation_budget": result.get("evaluation_budget"),
                        "runtime_seconds": result["runtime_seconds"],
                        "local_search_evaluation_count": result.get(
                            "local_search_evaluation_count"
                        ),
                        "local_search_global_best_improvements": result.get(
                            "local_search_global_best_improvements"
                        ),
                        "incomplete_local_evaluations": result.get(
                            "incomplete_local_evaluations"
                        ),
                    }
                )


def _plot_convergence(study_root, rows):
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, algorithm, color in (
        ("GWO (existing)", "baseline", "#1f77b4"),
        ("IGWO", "igwo", "#ff7f0e"),
    ):
        curves = np.asarray(
            [row["comparison"][algorithm]["convergence_history"] for row in rows],
            dtype=float,
        )
        iterations = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        deviation = (
            curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
        )
        axis.plot(iterations, mean, marker="o", label=label, color=color)
        axis.fill_between(
            iterations,
            mean - deviation,
            mean + deviation,
            color=color,
            alpha=0.16,
        )
    axis.set(
        title="Validation-accuracy convergence (three seeds)",
        xlabel="Iteration",
        ylabel="Best validation accuracy",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(study_root / "convergence_mean.png", dpi=160)
    plt.close(figure)


def _plot_local_contributions(study_root, rows):
    seeds = [row["seed"] for row in rows]
    improvements = [
        row["igwo_summary"]["local_search_global_best_improvements"]
        for row in rows
    ]
    evaluations = [
        row["igwo_summary"]["local_search_evaluation_count"] for row in rows
    ]
    x = np.arange(len(seeds))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - width / 2, improvements, width, label="Global Best improvements")
    axis.bar(x + width / 2, evaluations, width, label="Unique local candidates")
    axis.set(
        title="IGWO local-search contribution",
        xlabel="Seed",
        ylabel="Count",
        xticks=x,
        xticklabels=seeds,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(study_root / "local_search_contribution.png", dpi=160)
    plt.close(figure)


def analyze(study_root):
    study_root = study_root.resolve()
    rows = [_load_seed(study_root, seed) for seed in SEEDS]
    report = _build_report(rows)
    _write_json(study_root / "aggregate_summary.json", report)
    _write_csv(study_root, rows)
    _plot_convergence(study_root, rows)
    _plot_local_contributions(study_root, rows)
    print(f"Three-seed comparison written to {study_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate completed IGWO paper-parameter runs against saved GWO results."
    )
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    args = parser.parse_args()
    analyze(args.study_root)


if __name__ == "__main__":
    main()
