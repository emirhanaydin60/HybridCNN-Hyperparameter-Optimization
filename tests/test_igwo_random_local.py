import unittest

import numpy as np

from optimizers.igwo_random_local import (
    RANDOM_LOCAL_SEARCH_METHOD,
    RandomLocalSearchIGWO,
)
from scripts.supervise_igwo_8x15_after_adaptive import _dead_before_success


def spec(name, kind="continuous", values=None, coordinate_scale=None):
    result = {"name": name, "kind": kind}
    if values is not None:
        result["values"] = values
    if coordinate_scale is not None:
        result["coordinate_scale"] = coordinate_scale
    return result


class ScriptedRandomLocalOptimizer(RandomLocalSearchIGWO):
    def __init__(self, *args, candidates=(), **kwargs):
        self.candidates = list(candidates)
        super().__init__(*args, **kwargs)

    def _select_dimension(self, eligible):
        return eligible[0]

    def _randomize_dimension(self, position, dimension, excluded_values=None):
        value = self.candidates.pop(0)
        candidate = position.copy()
        candidate[dimension] = value
        return candidate


class RandomLocalSearchIGWOTests(unittest.TestCase):
    def test_random_reinitialization_changes_one_coordinate_in_valid_domains(self):
        optimizer = RandomLocalSearchIGWO(
            lambda position, **kwargs: float(position[0]),
            [(0, 2), (0.1, 0.9), (-5, -2), (1, 7)],
            population=2,
            iterations=1,
            dimension_specs=[
                spec("category", "categorical", [3, 5, 7]),
                spec("dropout"),
                spec("learning_rate", coordinate_scale="log10"),
                spec("depth", "integer", [1, 3, 5, 7]),
            ],
            random_seed=14,
            local_search_candidates=2,
        )
        position = np.asarray([1.0, 0.4, -4.0, 3.0])

        for dimension in range(4):
            candidate = optimizer._randomize_dimension(position, dimension)
            changed = np.flatnonzero(candidate != position)
            self.assertEqual(changed.tolist(), [dimension])
            self.assertGreaterEqual(candidate[dimension], optimizer.bounds[dimension][0])
            self.assertLessEqual(candidate[dimension], optimizer.bounds[dimension][1])
            if dimension == 0:
                self.assertIn(int(candidate[dimension]), (0, 2))
            elif dimension == 2:
                self.assertTrue(-5 <= candidate[dimension] <= -2)
            elif dimension == 3:
                self.assertIn(int(candidate[dimension]), (1, 5, 7))

    def test_cache_hit_retries_same_parameter_and_does_not_count(self):
        cache = {(0.2,): 0.1}

        def fitness(position, **kwargs):
            key = tuple(position.tolist())
            cached = key in cache
            fitness.last_cache_hit = cached
            if cached:
                fitness.cache_hits += 1
                return cache[key]
            fitness.cache_misses += 1
            cache[key] = 0.3
            return cache[key]

        fitness.cache_hits = 0
        fitness.cache_misses = 0
        fitness.last_cache_hit = False
        fitness.is_cached = lambda position: tuple(position.tolist()) in cache
        optimizer = ScriptedRandomLocalOptimizer(
            fitness,
            [(0.0, 1.0)],
            population=1,
            iterations=1,
            dimension_specs=[spec("dropout")],
            candidates=[0.2, 0.4],
            random_seed=2,
            local_search_candidates=1,
        )

        best_position, best_fitness, history, count, improvements, missing, _ = (
            optimizer._local_search(1, np.asarray([0.0]), 0.1)
        )

        self.assertEqual(count, 1)
        self.assertEqual(improvements, 1)
        self.assertEqual(missing, 0)
        self.assertEqual(best_position.tolist(), [0.4])
        self.assertEqual(best_fitness, 0.3)
        self.assertEqual(
            [item["status"] for item in history],
            ["cache_hit_retry", "improved"],
        )
        self.assertEqual(
            [item["parameter"] for item in history], ["dropout", "dropout"]
        )
        self.assertEqual([item["candidate_index"] for item in history], [1, 1])
        self.assertTrue(all(item["cache_hit"] for item in history[:1]))

    def test_next_random_candidate_uses_the_updated_global_best(self):
        def fitness(position, **kwargs):
            fitness.cache_misses += 1
            fitness.cache_hits = 0
            fitness.last_cache_hit = False
            return float(position[0])

        fitness.cache_misses = 0
        optimizer = ScriptedRandomLocalOptimizer(
            fitness,
            [(0.0, 1.0)],
            population=1,
            iterations=1,
            dimension_specs=[spec("dropout")],
            candidates=[0.4, 0.8],
            random_seed=4,
            local_search_candidates=2,
        )

        best_position, _, history, count, improvements, missing, _ = (
            optimizer._local_search(1, np.asarray([0.1]), 0.1)
        )

        self.assertEqual(count, 2)
        self.assertEqual(improvements, 2)
        self.assertEqual(missing, 0)
        self.assertEqual(best_position.tolist(), [0.8])
        self.assertEqual(history[1]["base_position"], [0.4])
        self.assertEqual(
            {item["search_method"] for item in history},
            {RANDOM_LOCAL_SEARCH_METHOD},
        )

    def test_optimizer_targets_unique_local_evaluations_per_iteration(self):
        cache = {}

        def fitness(position, **kwargs):
            key = tuple(np.round(position, 8).tolist())
            cached = key in cache
            fitness.last_cache_hit = cached
            if cached:
                fitness.cache_hits += 1
                return cache[key]
            fitness.cache_misses += 1
            cache[key] = float(-np.sum((position - 0.7) ** 2))
            return cache[key]

        fitness.cache_hits = 0
        fitness.cache_misses = 0
        fitness.last_cache_hit = False
        fitness.is_cached = lambda position: (
            tuple(np.round(position, 8).tolist()) in cache
        )
        optimizer = RandomLocalSearchIGWO(
            fitness,
            [(0.0, 1.0), (0.0, 1.0)],
            population=3,
            iterations=2,
            dimension_specs=[spec("x"), spec("y")],
            random_seed=7,
            local_search_candidates=2,
        )

        result = optimizer.optimize()

        self.assertEqual(result["local_search_evaluation_counts"], [2, 2])
        self.assertEqual(result["incomplete_local_evaluations"], 0)
        self.assertEqual(result["local_search_candidates_per_iteration"], 2)
        self.assertEqual(len(result["local_search_history"]), 4)
        self.assertTrue(
            all(
                item["search_method"] == RANDOM_LOCAL_SEARCH_METHOD
                for item in result["local_search_history"]
            )
        )
        for item in result["local_search_history"]:
            changed = np.flatnonzero(
                np.asarray(item["candidate_position"])
                != np.asarray(item["base_position"])
            )
            self.assertEqual(len(changed), 1)


class SupervisorCompletionGateTests(unittest.TestCase):
    def test_completed_seed_process_exit_does_not_block_other_pending_seeds(self):
        processes = {
            42: {"pid": 20104, "active_expected_runner": False},
            43: {"pid": 9112, "active_expected_runner": True},
            44: {"pid": 2412, "active_expected_runner": True},
        }
        results = [
            {"seed": 42, "status": "completed"},
            {"seed": 43, "status": "pending"},
            {"seed": 44, "status": "pending"},
        ]

        self.assertEqual(_dead_before_success(processes, results), [])

    def test_exited_process_without_verified_outputs_is_a_failure(self):
        processes = {
            42: {"pid": 20104, "active_expected_runner": False},
            43: {"pid": 9112, "active_expected_runner": True},
            44: {"pid": 2412, "active_expected_runner": True},
        }
        results = [
            {"seed": 42, "status": "pending"},
            {"seed": 43, "status": "pending"},
            {"seed": 44, "status": "pending"},
        ]

        self.assertEqual(
            _dead_before_success(processes, results),
            [{"seed": 42, "pid": 20104, "status": "pending"}],
        )


if __name__ == "__main__":
    unittest.main()
