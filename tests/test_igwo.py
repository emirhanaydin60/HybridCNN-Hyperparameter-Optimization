import unittest

import numpy as np

from optimizers.gwo import GreyWolfOptimizer
from optimizers.igwo import ImprovedGreyWolfOptimizer


class ScriptedOptimizer(ImprovedGreyWolfOptimizer):
    def __init__(self, *args, initial_population, mutations, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_population = np.asarray(initial_population, dtype=float)
        self.mutations = list(mutations)

    def _initialize(self):
        return self.initial_population.copy()

    def _mutate_dimension(
        self, position, dimension, excluded_values=None, step_size=None
    ):
        value = self.mutations.pop(0)
        candidate = position.copy()
        candidate[dimension] = value
        return candidate


def spec(name="x", kind="continuous", values=None):
    result = {"name": name, "kind": kind}
    if values is not None:
        result["values"] = values
    return result


class ImprovedGreyWolfOptimizerTests(unittest.TestCase):
    def test_local_search_targets_two_unique_candidates_per_iteration(self):
        evaluated = set()

        def fitness(position, **kwargs):
            evaluated.add(tuple(np.round(position, 8)))
            return float(np.sum(position))

        optimizer = ImprovedGreyWolfOptimizer(
            fitness,
            [(0.0, 1.0), (0.0, 1.0)],
            population=3,
            iterations=1,
            random_seed=15,
        )
        result = optimizer.optimize()

        self.assertEqual(result["local_search_evaluation_counts"], [2])
        self.assertEqual(result["incomplete_local_evaluations"], 0)
        self.assertEqual(len(result["local_search_history"]), 2)
        self.assertTrue(
            all(
                "step_size" in attempt
                and "step_size_before" in attempt
                and "step_size_after" in attempt
                and "candidate_fitness" in attempt
                and "cache_hit" in attempt
                and "improved" in attempt
                for attempt in result["local_search_history"]
            )
        )
        self.assertEqual(len(evaluated), len(result["local_search_history"]) + 6)

    def test_only_one_coordinate_changes_per_local_candidate(self):
        fitness = lambda position, **kwargs: float(np.sum(position))
        optimizer = ImprovedGreyWolfOptimizer(
            fitness,
            [(0.0, 2.0), (0.0, 1.0), (0.0, 3.0)],
            population=2,
            iterations=1,
            solution_signature_func=lambda position: tuple(position.tolist()),
            dimension_specs=[
                spec("category", "categorical", [3, 5, 7]),
                spec("dropout"),
                spec("layers", "categorical", [1, 2, 3, 4]),
            ],
            random_seed=7,
        )
        optimizer._initialize = lambda: np.asarray(
            [[0.0, 0.2, 0.0], [1.0, 0.4, 1.0]], dtype=float
        )
        result = optimizer.optimize()

        for attempt in result["local_search_history"]:
            if attempt["status"] in (
                "duplicate_candidate_retry",
                "retry_exhausted",
                "no_neighbor",
                "budget_blocked",
            ):
                continue
            changed = np.flatnonzero(
                np.asarray(attempt["base_position"])
                != np.asarray(attempt["candidate_position"])
            )
            self.assertEqual(changed.tolist(), [
                ["category", "dropout", "layers"].index(attempt["parameter"])
            ])
            if attempt["parameter"] == "category":
                self.assertIn(round(attempt["new_value"]), [3, 5, 7])
            if attempt["parameter"] == "layers":
                self.assertIn(round(attempt["new_value"]), [1, 2, 3, 4])

    def test_next_local_candidate_uses_the_updated_global_best(self):
        def fitness(position, **kwargs):
            return float(position[0])

        optimizer = ScriptedOptimizer(
            fitness,
            [(0.0, 1.0)],
            population=2,
            iterations=1,
            solution_signature_func=lambda position: (round(float(position[0]), 8),),
            dimension_specs=[spec("x")],
            initial_population=[[0.2], [0.3]],
            mutations=[0.5, 0.8],
            local_search_candidates=2,
            random_seed=1,
        )
        best_position, best_fitness, history, count, improvements, missing, stopped = (
            optimizer._local_search(1, np.asarray([0.2]), 0.2)
        )

        self.assertEqual(count, 2)
        self.assertEqual(improvements, 2)
        self.assertEqual(missing, 0)
        self.assertFalse(stopped)
        self.assertEqual(history[1]["base_position"], [0.5])
        self.assertEqual(best_position.tolist(), [0.8])
        self.assertEqual(best_fitness, 0.8)

    def test_cache_hit_retries_the_same_dimension_and_does_not_count(self):
        calls = []

        def fitness(position, **kwargs):
            cache_hit = round(float(position[0]), 8) == 0.3
            fitness.last_cache_hit = cache_hit
            fitness.cache_hits += int(cache_hit)
            fitness.cache_misses += int(not cache_hit)
            calls.append((float(position[0]), cache_hit))
            return -1.0 if cache_hit else float(position[0])

        fitness.last_cache_hit = False
        fitness.cache_hits = 0
        fitness.cache_misses = 1
        fitness.is_cached = lambda position: round(float(position[0]), 8) == 0.3
        optimizer = ScriptedOptimizer(
            fitness,
            [(0.0, 1.0)],
            population=1,
            iterations=1,
            solution_signature_func=lambda position: (round(float(position[0]), 8),),
            dimension_specs=[spec("x")],
            initial_population=[[0.2]],
            mutations=[0.3, 0.4],
            local_search_candidates=1,
            random_seed=2,
        )
        best_position, _, history, count, _, missing, _ = optimizer._local_search(
            1, np.asarray([0.2]), 0.2
        )

        self.assertEqual(calls, [(0.3, True), (0.4, False)])
        self.assertEqual(count, 1)
        self.assertEqual(missing, 0)
        self.assertEqual([row["parameter"] for row in history], ["x", "x"])
        self.assertEqual(history[0]["status"], "cache_hit_retry")
        self.assertEqual(best_position.tolist(), [0.4])

    def test_categorical_cache_hits_try_each_alternative_only_once(self):
        calls = []

        def fitness(position, **kwargs):
            fitness.cache_hits += 1
            fitness.last_cache_hit = True
            calls.append(int(position[0]))
            return 0.0

        fitness.cache_hits = 0
        fitness.cache_misses = 1
        fitness.last_cache_hit = False
        fitness.is_cached = lambda position: True
        optimizer = ImprovedGreyWolfOptimizer(
            fitness,
            [(0.0, 2.0)],
            population=1,
            iterations=1,
            dimension_specs=[spec("choice", "categorical", [10, 20, 30])],
            local_search_candidates=1,
            random_seed=11,
        )

        _, _, history, count, _, missing, stopped = optimizer._local_search(
            1, np.asarray([0.0]), 0.0
        )

        self.assertEqual(count, 0)
        self.assertEqual(missing, 1)
        self.assertFalse(stopped)
        self.assertEqual(sorted(calls), [1, 2])
        self.assertEqual(len(history), 3)
        self.assertTrue(
            all(row["status"] == "cache_hit_retry" for row in history[:2])
        )
        self.assertEqual(history[-1]["status"], "no_neighbor")

    def test_exact_cache_miss_is_not_rejected_by_a_coarse_signature(self):
        def fitness(position, **kwargs):
            fitness.cache_misses += 1
            fitness.last_cache_hit = False
            return float(position[0])

        fitness.cache_misses = 1
        fitness.cache_hits = 0
        fitness.last_cache_hit = False
        fitness.is_cached = lambda position: False
        optimizer = ScriptedOptimizer(
            fitness,
            [(0.0, 1.0)],
            population=1,
            iterations=1,
            solution_signature_func=lambda position: ("rounded-collision",),
            dimension_specs=[spec("x")],
            initial_population=[[0.2]],
            mutations=[0.4],
            local_search_candidates=1,
            random_seed=4,
        )
        optimizer._known_signatures.add(("rounded-collision",))
        best_position, best_fitness, _, count, _, missing, _ = (
            optimizer._local_search(1, np.asarray([0.2]), 0.2)
        )

        self.assertEqual(count, 1)
        self.assertEqual(missing, 0)
        self.assertEqual(best_position.tolist(), [0.4])
        self.assertEqual(best_fitness, 0.4)

    def test_budget_reports_missing_local_evaluations(self):
        calls = []

        def fitness(position, **kwargs):
            calls.append(position.copy())
            fitness.cache_misses += 1
            fitness.last_cache_hit = False
            return float(position[0])

        fitness.cache_misses = 1
        fitness.cache_hits = 0
        fitness.last_cache_hit = False
        fitness.is_cached = lambda position: False
        optimizer = ScriptedOptimizer(
            fitness,
            [(0.0, 1.0)],
            population=1,
            iterations=1,
            solution_signature_func=lambda position: (round(float(position[0]), 8),),
            dimension_specs=[spec("x")],
            evaluation_budget=1,
            initial_population=[[0.2]],
            mutations=[0.4],
            local_search_candidates=1,
            random_seed=3,
        )
        best_position, _, _, count, _, missing, stopped = optimizer._local_search(
            1, np.asarray([0.2]), 0.2
        )

        self.assertEqual(count, 0)
        self.assertEqual(missing, 1)
        self.assertTrue(stopped)
        self.assertEqual(calls, [])
        self.assertEqual(best_position.tolist(), [0.2])

    def test_seed_reproduces_igwo_search(self):
        def fitness(position, **kwargs):
            return float(-np.sum((position - 0.4) ** 2))

        kwargs = {
            "fitness_func": fitness,
            "bounds": [(0.0, 1.0), (0.0, 1.0)],
            "population": 3,
            "iterations": 2,
            "random_seed": 23,
        }
        first = ImprovedGreyWolfOptimizer(**kwargs).optimize()
        second = ImprovedGreyWolfOptimizer(**kwargs).optimize()

        self.assertEqual(
            first["initial_population_signatures"],
            second["initial_population_signatures"],
        )
        self.assertEqual(
            first["initial_population_positions"],
            second["initial_population_positions"],
        )
        self.assertEqual(first["global_bests"], second["global_bests"])
        self.assertEqual(first["local_search_history"], second["local_search_history"])

    def test_adaptive_step_sizes_shrink_on_improvement_and_grow_otherwise(self):
        optimizer = ImprovedGreyWolfOptimizer(
            lambda position, **kwargs: float(position[0]),
            [(0.0, 1.0)],
            dimension_specs=[spec("dropout")],
            random_seed=5,
        )

        initial, after_no_improvement = optimizer._adapt_step_size(0, False)
        before_improvement, after_improvement = optimizer._adapt_step_size(0, True)

        self.assertAlmostEqual(initial, 0.1)
        self.assertAlmostEqual(after_no_improvement, 0.2)
        self.assertAlmostEqual(before_improvement, 0.2)
        self.assertAlmostEqual(after_improvement, 0.1)
        for _ in range(10):
            _, step = optimizer._adapt_step_size(0, True)
        self.assertAlmostEqual(step, 0.01)
        _, step = optimizer._adapt_step_size(0, False)
        self.assertAlmostEqual(step, 0.02)

    def test_log_scaled_continuous_step_is_applied_in_log_coordinates(self):
        optimizer = ImprovedGreyWolfOptimizer(
            lambda position, **kwargs: float(position[0]),
            [(-5.0, -2.0)],
            dimension_specs=[
                {
                    "name": "learning_rate",
                    "kind": "continuous",
                    "coordinate_scale": "log10",
                }
            ],
            random_seed=8,
        )
        start = np.asarray([-3.0])
        candidate = optimizer._mutate_dimension(start, 0, step_size=0.3)

        delta = abs(candidate[0] - start[0])
        self.assertGreaterEqual(delta, 0.03)
        self.assertLessEqual(delta, 0.3)
        self.assertAlmostEqual(
            10 ** candidate[0] / (10 ** start[0]),
            10 ** (candidate[0] - start[0]),
        )

    def test_integer_search_uses_bounded_integer_neighbors(self):
        optimizer = ImprovedGreyWolfOptimizer(
            lambda position, **kwargs: float(position[0]),
            [(1, 5)],
            dimension_specs=[spec("layers", "integer")],
            random_seed=13,
        )
        start = np.asarray([3.0])
        adjacent = optimizer._mutate_dimension(start, 0)
        self.assertIn(int(adjacent[0]), (2, 4))

        _, grown = optimizer._adapt_step_size(0, False)
        self.assertEqual(grown, 2.0)
        farther = optimizer._mutate_dimension(start, 0, step_size=grown)
        self.assertIn(int(farther[0]), (1, 5))

    def test_gwo_and_igwo_start_from_identical_seeded_populations(self):
        fitness = lambda position, **kwargs: float(-np.sum(position**2))
        shared = {
            "fitness_func": fitness,
            "bounds": [(0.0, 1.0), (0.0, 1.0)],
            "population": 3,
            "iterations": 2,
            "random_seed": 31,
        }
        gwo_result = GreyWolfOptimizer(**shared).optimize()
        igwo_result = ImprovedGreyWolfOptimizer(**shared).optimize()

        self.assertEqual(
            gwo_result["initial_population_positions"],
            igwo_result["initial_population_positions"],
        )
        self.assertEqual(
            gwo_result["initial_population_signatures"],
            igwo_result["initial_population_signatures"],
        )


if __name__ == "__main__":
    unittest.main()
