import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "results" / "igwo_comparison"
METRICS = {
    "test_accuracy": ("best_test_accuracy", "Test accuracy"),
    "macro_precision": ("precision", "Test macro-precision"),
    "macro_recall": ("recall", "Test macro-recall"),
    "macro_f1": ("f1", "Test macro-F1"),
    "balanced_accuracy": ("balanced_accuracy", "Test balanced accuracy"),
    "test_loss": ("test_loss", "Test loss"),
    "search_time_seconds": ("total_search_time_seconds", "Search time (s)"),
    "runtime_seconds": ("runtime_seconds", "Total runtime (s)"),
    "unique_fitness_evaluations": (
        "unique_fitness_evaluations",
        "Unique candidate trainings",
    ),
}


def _read_json(path):
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _write_json(path, payload):
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)


def _run_summaries(study_dir, phase_key, algorithm, run_count):
    algorithm_dir = (
        study_dir / phase_key / "CIFAR10" / algorithm.upper()
    )
    summaries = []
    for index in range(run_count):
        run_dir = algorithm_dir / f"run_{index + 1:02d}"
        summaries.append(_read_json(run_dir / "summary.json"))
    return summaries


def _describe(values):
    numeric = [float(value) for value in values]
    return {
        "n": len(numeric),
        "mean": statistics.mean(numeric),
        "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _phase_groups(study_dir, run_count):
    return {
        "equal_iterations_gwo": _run_summaries(
            study_dir, "equal_iterations_gwo", "gwo", run_count
        ),
        "equal_iterations_igwo": _run_summaries(
            study_dir, "equal_iterations_igwo", "igwo", run_count
        ),
        "equal_fitness_budget_igwo": _run_summaries(
            study_dir, "equal_fitness_budget_igwo", "igwo", run_count
        ),
    }


def _validate_completed(manifest, groups):
    if manifest.get("status") != "completed":
        raise RuntimeError(
            f"Study status is {manifest.get('status', 'unknown')}; "
            "analysis requires all runs to complete"
        )
    if any(status != "completed" for status in manifest["phases"].values()):
        raise RuntimeError("Not all comparison phases completed")

    reference = groups["equal_iterations_gwo"]
    for label, rows in groups.items():
        if len(rows) != len(reference):
            raise RuntimeError(f"Run count mismatch in {label}")
        for expected, actual in zip(reference, rows):
            if expected["run_seed"] != actual["run_seed"]:
                raise RuntimeError(f"Seed mismatch in {label}")
            if expected["initial_population_signatures"] != actual[
                "initial_population_signatures"
            ] or expected["initial_population_positions"] != actual[
                "initial_population_positions"
            ]:
                raise RuntimeError(f"Initial population mismatch in {label}")


def _comparison_report(groups):
    report = {
        "interpretation": (
            "Descriptive comparison only; three matched seeds are not sufficient "
            "to claim statistical superiority."
        ),
        "groups": {},
        "paired_differences": {},
    }
    for label, summaries in groups.items():
        report["groups"][label] = {
            metric: _describe([summary[source] for summary in summaries])
            for metric, (source, _) in METRICS.items()
        }

    pairings = {
        "equal_iterations_igwo_minus_gwo": (
            groups["equal_iterations_igwo"],
            groups["equal_iterations_gwo"],
        ),
        "equal_fitness_budget_igwo_minus_gwo": (
            groups["equal_fitness_budget_igwo"],
            groups["equal_iterations_gwo"],
        ),
    }
    for label, (left, right) in pairings.items():
        report["paired_differences"][label] = {}
        for metric, (source, _) in METRICS.items():
            deltas = [
                first[source] - second[source]
                for first, second in zip(left, right)
            ]
            report["paired_differences"][label][metric] = {
                **_describe(deltas),
                "by_seed": [
                    {
                        "seed": first["run_seed"],
                        "difference": delta,
                    }
                    for first, delta in zip(left, deltas)
                ],
            }
    report["run_details"] = {
        label: [
            {
                "seed": summary["run_seed"],
                "best_validation_accuracy": summary[
                    "best_validation_accuracy"
                ],
                "best_epoch": summary["best_final_monitor_epoch"],
                "trainable_parameter_count": summary[
                    "trainable_parameter_count"
                ],
                "unique_fitness_evaluations": summary[
                    "unique_fitness_evaluations"
                ],
                "cache_hits": summary["cache_hits"],
                "evaluation_budget": summary["optimizer_state"].get(
                    "evaluation_budget"
                ),
                "search_time_seconds": summary["total_search_time_seconds"],
                "final_training_time_seconds": summary[
                    "final_training_time_seconds"
                ],
                "runtime_seconds": summary["runtime_seconds"],
                "best_hyperparameters": summary["best_hyperparameters"],
            }
            for summary in summaries
        ]
        for label, summaries in groups.items()
    }
    return report


def _plot_equal_iteration_convergence(study_dir, groups):
    curves = [
        ("GWO", groups["equal_iterations_gwo"], "#1f77b4"),
        ("IGWO", groups["equal_iterations_igwo"], "#ff7f0e"),
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, summaries, color in curves:
        values = np.asarray(
            [summary["convergence_history"] for summary in summaries],
            dtype=float,
        )
        iterations = np.arange(1, values.shape[1] + 1)
        mean = values.mean(axis=0)
        deviation = values.std(axis=0, ddof=1) if values.shape[0] > 1 else 0.0
        axis.plot(iterations, mean, marker="o", label=label, color=color)
        axis.fill_between(
            iterations,
            mean - deviation,
            mean + deviation,
            color=color,
            alpha=0.16,
        )
    axis.set(
        title="Validation-fitness convergence (equal iterations)",
        xlabel="Iteration",
        ylabel="Best validation accuracy",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(study_dir / "convergence_equal_iterations.png", dpi=160)
    plt.close(figure)


def _plot_test_metrics(study_dir, groups):
    names = ["Accuracy", "Macro-P", "Macro-R", "Macro-F1", "Balanced acc."]
    sources = [
        "best_test_accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]
    group_keys = [
        "equal_iterations_gwo",
        "equal_iterations_igwo",
        "equal_fitness_budget_igwo",
    ]
    labels = ["GWO\n(equal iterations)", "IGWO\n(equal iterations)", "IGWO\n(equal budget)"]
    centers = np.arange(len(names), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for group_index, (group_key, label) in enumerate(zip(group_keys, labels)):
        summaries = groups[group_key]
        means = [
            statistics.mean(summary[source] for summary in summaries)
            for source in sources
        ]
        deviations = [
            statistics.stdev(summary[source] for summary in summaries)
            if len(summaries) > 1
            else 0.0
            for source in sources
        ]
        axis.bar(
            centers + (group_index - 1) * width,
            means,
            width,
            yerr=deviations,
            capsize=3,
            label=label,
        )
    axis.set(
        title="CIFAR-10 test metrics across matched seeds",
        ylabel="Score",
        xticks=centers,
        xticklabels=names,
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(study_dir / "test_metrics_by_seed.png", dpi=160)
    plt.close(figure)


def _write_parameter_summary(study_dir, groups):
    counts = Counter()
    improvement_counts = Counter()
    for index in range(len(groups["equal_iterations_igwo"])):
        trace = _read_json(
            study_dir
            / "equal_iterations_igwo"
            / "CIFAR10"
            / "IGWO"
            / f"run_{index + 1:02d}"
            / "igwo_trace.json"
        )
        for attempt in trace["local_search_attempts"]:
            counts[attempt["parameter"]] += 1
            if attempt["improved"]:
                improvement_counts[attempt["parameter"]] += 1
    with (study_dir / "igwo_parameter_search_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["parameter", "attempts", "improvements"],
        )
        writer.writeheader()
        for parameter in sorted(counts):
            writer.writerow(
                {
                    "parameter": parameter,
                    "attempts": counts[parameter],
                    "improvements": improvement_counts[parameter],
                }
            )


def _write_run_metrics_csv(study_dir, groups):
    path = study_dir / "comparison_summary.csv"
    if path.exists():
        return
    fields = [
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
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for label, summaries in groups.items():
            phase, algorithm = (
                ("equal_iterations", "gwo")
                if label == "equal_iterations_gwo"
                else (
                    ("equal_iterations", "igwo")
                    if label == "equal_iterations_igwo"
                    else ("equal_fitness_budget", "igwo")
                )
            )
            for summary in summaries:
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
                        "best_validation_accuracy": summary[
                            "best_validation_accuracy"
                        ],
                        "best_epoch": summary["best_final_monitor_epoch"],
                        "trainable_parameter_count": summary[
                            "trainable_parameter_count"
                        ],
                        "unique_fitness_evaluations": summary[
                            "unique_fitness_evaluations"
                        ],
                        "cache_hits": summary["cache_hits"],
                        "search_time_seconds": summary[
                            "total_search_time_seconds"
                        ],
                        "final_training_time_seconds": summary[
                            "final_training_time_seconds"
                        ],
                        "runtime_seconds": summary["runtime_seconds"],
                        "evaluation_budget": summary["optimizer_state"].get(
                            "evaluation_budget"
                        ),
                    }
                )


def analyze(study_dir):
    study_dir = study_dir.resolve()
    manifest = _read_json(study_dir / "comparison_manifest.json")
    run_count = len(manifest["seeds"])
    groups = _phase_groups(study_dir, run_count)
    _validate_completed(manifest, groups)
    report = _comparison_report(groups)
    report["protocol_hash"] = manifest["protocol_hash"]
    report["seeds"] = manifest["seeds"]
    report["test_set_usage"] = (
        "Independent test metrics were calculated after best-checkpoint selection; "
        "they did not control search fitness or checkpoint selection."
    )
    _write_json(study_dir / "analysis_summary.json", report)
    _write_run_metrics_csv(study_dir, groups)
    _write_parameter_summary(study_dir, groups)
    _plot_equal_iteration_convergence(study_dir, groups)
    _plot_test_metrics(study_dir, groups)
    print(f"Analysis written to {study_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize completed matched-seed GWO/IGWO studies."
    )
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    analyze(args.study_dir)


if __name__ == "__main__":
    main()
