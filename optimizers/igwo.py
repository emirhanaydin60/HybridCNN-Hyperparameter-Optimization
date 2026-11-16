import time

import numpy as np

from .base_optimizer import BaseOptimizer


class ImprovedGreyWolfOptimizer(BaseOptimizer):
    """Grey Wolf Optimizer with sequential, global-best-centered local search."""

    def __init__(
        self,
        fitness_func,
        bounds,
        population=8,
        iterations=15,
        solution_signature_func=None,
        random_seed=None,
        dimension_specs=None,
        describe_solution_func=None,
        evaluation_budget=None,
        logger=None,
        retry_limit=64,
    ):
        super().__init__(
            fitness_func,
            bounds,
            population=population,
            iterations=iterations,
            solution_signature_func=solution_signature_func,
            random_seed=random_seed,
        )
        if self.population < 1:
            raise ValueError("population must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if retry_limit < 1:
            raise ValueError("retry_limit must be positive")
        if evaluation_budget is not None and evaluation_budget < self.population:
            raise ValueError("evaluation_budget must allow the initial population")
        self.dimension_specs = dimension_specs or [
            {"name": f"dimension_{index}", "kind": "continuous"}
            for index in range(self.dim)
        ]
        if len(self.dimension_specs) != self.dim:
            raise ValueError("dimension_specs must match the number of bounds")
        self.describe_solution_func = describe_solution_func or (
            lambda position: self._signature(position)
        )
        self.evaluation_budget = evaluation_budget
        self.logger = logger
        self.retry_limit = retry_limit
        self._known_signatures = set()

    def _fitness_cache_misses(self):
        return int(getattr(self.fitness_func, "cache_misses", 0))

    def _fitness_cache_hits(self):
        return int(getattr(self.fitness_func, "cache_hits", 0))

    def _is_cached(self, position):
        is_cached = getattr(self.fitness_func, "is_cached", None)
        if is_cached is not None:
            return bool(is_cached(position))
        return self._signature(position) in self._known_signatures

    def _budget_blocks(self, position):
        return (
            self.evaluation_budget is not None
            and self._fitness_cache_misses() >= self.evaluation_budget
            and not self._is_cached(position)
        )

    def _log(self, level, message, *args):
        if self.logger is not None:
            getattr(self.logger, level)(message, *args)

    def _mutate_dimension(self, position, dimension, excluded_values=None):
        candidate = position.copy()
        lower, upper = self.bounds[dimension]
        spec = self.dimension_specs[dimension]
        excluded_values = excluded_values or set()

        if spec["kind"] == "categorical":
            values = spec.get("values")
            if not values:
                return None
            current_index = int(round(position[dimension]))
            alternatives = [
                index
                for index in range(len(values))
                if index != current_index and index not in excluded_values
            ]
            if not alternatives:
                return None
            candidate[dimension] = self.random_state.choice(alternatives)
            return self._clip(candidate)

        if spec["kind"] != "continuous":
            raise ValueError(f"Unsupported dimension kind: {spec['kind']}")

        width = upper - lower
        if width <= 0:
            return None
        current = float(position[dimension])
        for _ in range(16):
            step = float(self.random_state.normal(0.0, width * 0.1))
            value = float(np.clip(current + step, lower, upper))
            if value != current and value not in excluded_values:
                candidate[dimension] = value
                return self._clip(candidate)

        for _ in range(16):
            value = float(self.random_state.uniform(lower, upper))
            if value != current and value not in excluded_values:
                candidate[dimension] = value
                return self._clip(candidate)
        return None

    def _best_dimension(self, global_best, excluded_dimensions=None):
        excluded_dimensions = excluded_dimensions or set()
        eligible = []
        for index, spec in enumerate(self.dimension_specs):
            if index in excluded_dimensions:
                continue
            if spec["kind"] == "continuous":
                lower, upper = self.bounds[index]
                if upper > lower:
                    eligible.append(index)
            elif spec["kind"] == "categorical":
                values = spec.get("values", [])
                if len(values) > 1:
                    eligible.append(index)
        if not eligible:
            return None
        return int(self.random_state.choice(eligible))

    def _record_attempt(
        self,
        history,
        iteration,
        candidate_index,
        dimension,
        base_position,
        candidate,
        previous_fitness,
        candidate_fitness,
        cache_hit,
        status,
        best_position,
        best_fitness,
    ):
        before = self.describe_solution_func(base_position)
        after = self.describe_solution_func(candidate)
        name = self.dimension_specs[dimension]["name"]
        row = {
            "iteration": iteration,
            "candidate_index": candidate_index,
            "parameter": name,
            "previous_value": before[name]
            if isinstance(before, dict) and name in before
            else float(base_position[dimension]),
            "new_value": after[name]
            if isinstance(after, dict) and name in after
            else float(candidate[dimension]),
            "previous_fitness": float(previous_fitness),
            "candidate_fitness": (
                None if candidate_fitness is None else float(candidate_fitness)
            ),
            "cache_hit": bool(cache_hit),
            "status": status,
            "improved": status == "improved",
            "global_best_fitness": float(best_fitness),
            "global_best_position": np.asarray(best_position).tolist(),
            "base_position": np.asarray(base_position).tolist(),
            "candidate_position": np.asarray(candidate).tolist(),
        }
        history.append(row)
        self._log(
            "info" if status != "retry_exhausted" else "warning",
            "igwo_local_search iteration=%s candidate=%s parameter=%s "
            "previous=%s new=%s fitness=%s cache_hit=%s status=%s best=%s",
            iteration,
            candidate_index,
            name,
            row["previous_value"],
            row["new_value"],
            row["candidate_fitness"],
            row["cache_hit"],
            status,
            row["global_best_fitness"],
        )

    def _local_search(self, iteration, best_position, best_fitness):
        history = []
        successful_evaluations = 0
        improvements = 0
        missing = 0
        attempts_without_progress = 0
        exhausted_dimensions = set()
        max_total_attempts = self.population * self.retry_limit * max(self.dim, 1)

        while successful_evaluations < self.population:
            if attempts_without_progress >= max_total_attempts:
                missing = self.population - successful_evaluations
                self._log(
                    "error",
                    "IGWO local search stopped at iteration=%s: %s unique "
                    "evaluations remain after %s attempts",
                    iteration,
                    missing,
                    attempts_without_progress,
                )
                break

            dimension = self._best_dimension(best_position, exhausted_dimensions)
            if dimension is None:
                missing = self.population - successful_evaluations
                self._log(
                    "error",
                    "IGWO local search stopped at iteration=%s: no mutable "
                    "hyperparameter remains; %s unique evaluations are missing",
                    iteration,
                    missing,
                )
                break

            tried_values = set()
            found_unique = False
            dimension_exhausted = False
            for _ in range(self.retry_limit):
                base_position = best_position.copy()
                previous_fitness = best_fitness
                candidate = self._mutate_dimension(
                    base_position, dimension, excluded_values=tried_values
                )
                attempts_without_progress += 1
                if candidate is None:
                    dimension_exhausted = True
                    break
                tried_values.add(float(candidate[dimension]))
                if np.array_equal(candidate, base_position):
                    self._record_attempt(
                        history,
                        iteration,
                        successful_evaluations + 1,
                        dimension,
                        base_position,
                        candidate,
                        previous_fitness,
                        None,
                        False,
                        "duplicate_candidate_retry",
                        best_position,
                        best_fitness,
                    )
                    continue

                cached = self._is_cached(candidate)
                if self._budget_blocks(candidate):
                    missing = self.population - successful_evaluations
                    self._log(
                        "warning",
                        "IGWO evaluation budget reached at iteration=%s; "
                        "%s unique local evaluations remain",
                        iteration,
                        missing,
                    )
                    return (
                        best_position,
                        best_fitness,
                        history,
                        successful_evaluations,
                        improvements,
                        missing,
                        True,
                    )

                candidate_fitness = self._evaluate(
                    candidate,
                    iteration=iteration,
                    agent_id=self.population + successful_evaluations + 1,
                )
                actual_cache_hit = bool(
                    getattr(self.fitness_func, "last_cache_hit", cached)
                )
                signature = self._signature(candidate)
                signature_is_cached = (
                    not hasattr(self.fitness_func, "is_cached")
                    and signature in self._known_signatures
                )
                if actual_cache_hit or signature_is_cached:
                    self._record_attempt(
                        history,
                        iteration,
                        successful_evaluations + 1,
                        dimension,
                        base_position,
                        candidate,
                        previous_fitness,
                        candidate_fitness,
                        True,
                        "cache_hit_retry",
                        best_position,
                        best_fitness,
                    )
                    continue

                self._known_signatures.add(signature)
                successful_evaluations += 1
                attempts_without_progress = 0
                status = "evaluated"
                if candidate_fitness > best_fitness:
                    best_position = candidate.copy()
                    best_fitness = float(candidate_fitness)
                    improvements += 1
                    status = "improved"
                    exhausted_dimensions.clear()
                self._record_attempt(
                    history,
                    iteration,
                    successful_evaluations,
                    dimension,
                    base_position,
                    candidate,
                    previous_fitness,
                    candidate_fitness,
                    False,
                    status,
                    best_position,
                    best_fitness,
                )
                found_unique = True
                break
            if dimension_exhausted:
                exhausted_dimensions.add(dimension)
            elif not found_unique:
                exhausted_dimensions.add(dimension)
                self._log(
                    "warning",
                    "IGWO local search could not find a unique value for "
                    "parameter=%s after %s retries; selecting another parameter",
                    self.dimension_specs[dimension]["name"],
                    self.retry_limit,
                )
                self._record_attempt(
                    history,
                    iteration,
                    successful_evaluations + 1,
                    dimension,
                    best_position,
                    best_position,
                    best_fitness,
                    None,
                    False,
                    "retry_exhausted",
                    best_position,
                    best_fitness,
                )

        return (
            best_position,
            best_fitness,
            history,
            successful_evaluations,
            improvements,
            missing,
            False,
        )

    def optimize(self):
        total_start = time.perf_counter()
        wolves = self._initialize()
        initial_population_positions = [wolf.tolist() for wolf in wolves]
        initial_population_signatures = [
            list(self._signature(wolf)) for wolf in wolves
        ]
        fitness = []
        evaluation_count = 0
        for index, wolf in enumerate(wolves):
            if self._budget_blocks(wolf):
                raise RuntimeError(
                    "Evaluation budget ended before the initial population "
                    "could be evaluated"
                )
            signature = self._signature(wolf)
            score = self._evaluate(wolf, iteration=0, agent_id=index + 1)
            fitness.append(score)
            evaluation_count += 1
            if not bool(getattr(self.fitness_func, "last_cache_hit", False)):
                self._known_signatures.add(signature)

        pbest_pos = [wolf.copy() for wolf in wolves]
        pbest_fit = fitness.copy()
        best_index = int(np.argmax(pbest_fit))
        global_best_pos = pbest_pos[best_index].copy()
        global_best_fitness = float(pbest_fit[best_index])
        global_bests = []
        local_bests = []
        iteration_times = []
        iteration_summaries = [
            self._summarize_iteration(
                0, fitness, wolves, pbest_fit, total_start
            )
        ]
        local_search_history = []
        gwo_improvements = []
        local_improvements = []
        local_evaluation_counts = []
        incomplete_local_evaluations = 0
        budget_exhausted = False

        for iteration in range(1, self.iterations + 1):
            iter_start = time.perf_counter()
            prior_best_fitness = global_best_fitness
            ranking = np.argsort(fitness)[::-1]
            alpha = wolves[ranking[0]]
            beta = wolves[ranking[1]] if self.population > 1 else wolves[ranking[0]]
            delta = wolves[ranking[2]] if self.population > 2 else wolves[ranking[0]]
            a = 2 - (iteration - 1) * (2 / max(self.iterations - 1, 1))

            previous_wolves = wolves.copy()
            previous_fitness = fitness.copy()
            for wolf_id in range(self.population):
                position = previous_wolves[wolf_id]
                a1 = 2 * a * self.random_state.random(self.dim) - a
                c1 = 2 * self.random_state.random(self.dim)
                d_alpha = np.abs(c1 * alpha - position)
                x1 = alpha - a1 * d_alpha

                a2 = 2 * a * self.random_state.random(self.dim) - a
                c2 = 2 * self.random_state.random(self.dim)
                d_beta = np.abs(c2 * beta - position)
                x2 = beta - a2 * d_beta

                a3 = 2 * a * self.random_state.random(self.dim) - a
                c3 = 2 * self.random_state.random(self.dim)
                d_delta = np.abs(c3 * delta - position)
                x3 = delta - a3 * d_delta
                wolves[wolf_id] = self._clip((x1 + x2 + x3) / 3.0)

            current_fitness = previous_fitness.copy()
            evaluated_indices = []
            for wolf_id, wolf in enumerate(wolves):
                if self._budget_blocks(wolf):
                    wolves[wolf_id:] = previous_wolves[wolf_id:]
                    current_fitness[wolf_id:] = previous_fitness[wolf_id:]
                    budget_exhausted = True
                    break
                signature = self._signature(wolf)
                score = self._evaluate(
                    wolf, iteration=iteration, agent_id=wolf_id + 1
                )
                evaluation_count += 1
                current_fitness[wolf_id] = score
                evaluated_indices.append(wolf_id)
                if not bool(getattr(self.fitness_func, "last_cache_hit", False)):
                    self._known_signatures.add(signature)

            for wolf_index in evaluated_indices:
                if current_fitness[wolf_index] > pbest_fit[wolf_index]:
                    pbest_fit[wolf_index] = current_fitness[wolf_index]
                    pbest_pos[wolf_index] = wolves[wolf_index].copy()
            fitness = current_fitness
            best_index = int(np.argmax(pbest_fit))
            if pbest_fit[best_index] > global_best_fitness:
                global_best_pos = pbest_pos[best_index].copy()
                global_best_fitness = float(pbest_fit[best_index])
            gwo_improvements.append(
                int(global_best_fitness > prior_best_fitness)
            )

            if budget_exhausted:
                local_evaluation_counts.append(0)
                local_improvements.append(0)
                iteration_times.append(time.perf_counter() - iter_start)
                iteration_summaries.append(
                    self._summarize_iteration(
                        iteration, fitness, wolves, pbest_fit, iter_start
                    )
                )
                break

            (
                global_best_pos,
                global_best_fitness,
                local_history,
                local_count,
                local_improvement_count,
                missing,
                stopped_for_budget,
            ) = self._local_search(
                iteration, global_best_pos, global_best_fitness
            )
            local_search_history.extend(local_history)
            local_evaluation_counts.append(local_count)
            local_improvements.append(local_improvement_count)
            incomplete_local_evaluations += missing
            if stopped_for_budget:
                budget_exhausted = True

            pbest_index = int(np.argmax(pbest_fit))
            if global_best_fitness > pbest_fit[pbest_index]:
                worst_index = int(np.argmin(fitness))
                wolves[worst_index] = global_best_pos.copy()
                fitness[worst_index] = global_best_fitness
                pbest_pos[worst_index] = global_best_pos.copy()
                pbest_fit[worst_index] = global_best_fitness
                best_index = worst_index
            elif global_best_fitness == pbest_fit[pbest_index]:
                global_best_pos = pbest_pos[pbest_index].copy()
                best_index = pbest_index

            local_bests.append(pbest_fit.copy())
            global_bests.append(float(global_best_fitness))
            iteration_times.append(time.perf_counter() - iter_start)
            summary = self._summarize_iteration(
                iteration, fitness, wolves, pbest_fit, iter_start
            )
            summary.update(
                {
                    "gwo_improved_global_best": gwo_improvements[-1],
                    "local_search_evaluations": local_count,
                    "local_search_improvements": local_improvement_count,
                    "local_search_missing_evaluations": missing,
                    "global_best_after_local_search": float(global_best_fitness),
                }
            )
            iteration_summaries.append(summary)
            if budget_exhausted:
                break

        unique_evaluations = self._fitness_cache_misses()
        cache_hits = self._fitness_cache_hits()
        if self.evaluation_budget is not None and unique_evaluations > self.evaluation_budget:
            raise RuntimeError("IGWO exceeded its unique fitness evaluation budget")

        return {
            "best_pos": global_best_pos,
            "best_conf": global_best_pos,
            "best_fitness": float(global_best_fitness),
            "best_index": best_index + 1,
            "initial_population_positions": initial_population_positions,
            "initial_population_signatures": initial_population_signatures,
            "global_bests": global_bests,
            "local_bests": local_bests,
            "iteration_times": iteration_times,
            "iteration_summaries": iteration_summaries,
            "total_optimization_time": time.perf_counter() - total_start,
            "evaluation_count": evaluation_count,
            "unique_fitness_evaluations": unique_evaluations,
            "cache_hits": cache_hits,
            "gwo_improvements": gwo_improvements,
            "local_search_improvements": local_improvements,
            "local_search_evaluation_counts": local_evaluation_counts,
            "local_search_history": local_search_history,
            "local_search_success_rate": (
                sum(local_improvements)
                / max(sum(local_evaluation_counts), 1)
            ),
            "incomplete_local_evaluations": incomplete_local_evaluations,
            "evaluation_budget": self.evaluation_budget,
            "budget_exhausted": budget_exhausted,
        }
