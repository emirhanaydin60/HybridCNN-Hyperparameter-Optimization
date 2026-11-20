import numpy as np

from .igwo import ImprovedGreyWolfOptimizer


RANDOM_LOCAL_SEARCH_METHOD = "random_single_parameter_reinitialization"


class RandomLocalSearchIGWO(ImprovedGreyWolfOptimizer):
    """IGWO with uniformly reinitialized single-coordinate local candidates."""

    def _initialize_step_sizes(self):
        self._initial_step_sizes = [None] * self.dim
        self._step_sizes = [None] * self.dim
        self._step_size_bounds = [None] * self.dim

    def _step_size_report(self):
        return []

    def _record_attempt(self, *args, **kwargs):
        super()._record_attempt(*args, **kwargs)
        args[0][-1]["search_method"] = RANDOM_LOCAL_SEARCH_METHOD

    def _eligible_dimensions(self, excluded_dimensions):
        eligible = []
        for index, spec in enumerate(self.dimension_specs):
            if index in excluded_dimensions:
                continue
            lower, upper = self.bounds[index]
            kind = spec["kind"]
            if kind == "continuous" and upper > lower:
                eligible.append(index)
            elif kind == "categorical":
                values = spec.get("values")
                domain_size = (
                    len(values)
                    if values is not None
                    else int(np.floor(upper) - np.ceil(lower) + 1)
                )
                if domain_size > 1:
                    eligible.append(index)
            elif kind == "integer":
                values = spec.get("values")
                if values:
                    valid_values = [
                        int(value)
                        for value in values
                        if np.ceil(lower) <= int(value) <= np.floor(upper)
                    ]
                else:
                    valid_values = list(
                        range(int(np.ceil(lower)), int(np.floor(upper)) + 1)
                    )
                if len(set(valid_values)) > 1:
                    eligible.append(index)
            else:
                raise ValueError(f"Unsupported dimension kind: {kind}")
        return eligible

    def _select_dimension(self, eligible):
        return int(self.random_state.choice(eligible))

    def _randomize_dimension(self, position, dimension, excluded_values=None):
        candidate = position.copy()
        lower, upper = self.bounds[dimension]
        spec = self.dimension_specs[dimension]
        excluded_values = excluded_values or set()
        kind = spec["kind"]
        current = position[dimension]

        if kind == "continuous":
            coordinate_scale = spec.get("coordinate_scale", "linear")
            if coordinate_scale not in ("linear", "log10"):
                raise ValueError(
                    f"Unsupported coordinate scale: {coordinate_scale}"
                )
            for _ in range(64):
                value = float(self.random_state.uniform(lower, upper))
                if value != current and value not in excluded_values:
                    candidate[dimension] = value
                    return self._clip(candidate)
            return None

        if kind == "categorical":
            values = spec.get("values")
            if values is None:
                valid_values = list(
                    range(int(np.ceil(lower)), int(np.floor(upper)) + 1)
                )
            else:
                valid_values = list(range(len(values)))
            current_value = int(round(current))
            choices = [
                value
                for value in valid_values
                if value != current_value and value not in excluded_values
            ]
            if not choices:
                return None
            candidate[dimension] = self.random_state.choice(choices)
            return self._clip(candidate)

        if kind == "integer":
            values = spec.get("values")
            if values:
                valid_values = [
                    int(value)
                    for value in values
                    if np.ceil(lower) <= int(value) <= np.floor(upper)
                ]
            else:
                valid_values = list(
                    range(int(np.ceil(lower)), int(np.floor(upper)) + 1)
                )
            current_value = int(round(current))
            choices = [
                value
                for value in valid_values
                if value != current_value and value not in excluded_values
            ]
            if not choices:
                return None
            candidate[dimension] = self.random_state.choice(choices)
            return self._clip(candidate)

        raise ValueError(f"Unsupported dimension kind: {kind}")

    def _local_search(self, iteration, best_position, best_fitness):
        history = []
        successful_evaluations = 0
        improvements = 0
        target_evaluations = self.local_search_candidates
        attempts_since_success = 0
        exhausted_dimensions = set()
        max_attempts = (
            target_evaluations * self.retry_limit * max(self.dim, 1)
        )

        while successful_evaluations < target_evaluations:
            if attempts_since_success >= max_attempts:
                break

            eligible = self._eligible_dimensions(exhausted_dimensions)
            if not eligible:
                break
            dimension = self._select_dimension(eligible)
            tried_values = set()
            found_unique = False

            for _ in range(self.retry_limit):
                base_position = best_position.copy()
                previous_fitness = best_fitness
                candidate = self._randomize_dimension(
                    base_position,
                    dimension,
                    excluded_values=tried_values,
                )
                attempts_since_success += 1
                if candidate is None:
                    self._record_attempt(
                        history,
                        iteration,
                        successful_evaluations + 1,
                        dimension,
                        base_position,
                        base_position,
                        previous_fitness,
                        None,
                        False,
                        "no_neighbor",
                        best_position,
                        best_fitness,
                    )
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
                    missing = target_evaluations - successful_evaluations
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
                        "budget_blocked",
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
                        True,
                    )

                candidate_fitness = self._evaluate(
                    candidate,
                    iteration=iteration,
                    agent_id=self.population + successful_evaluations + 1,
                )
                signature = self._signature(candidate)
                cache_hit = bool(
                    cached
                    or getattr(self.fitness_func, "last_cache_hit", False)
                    or (
                        not hasattr(self.fitness_func, "is_cached")
                        and signature in self._known_signatures
                    )
                )
                improved = candidate_fitness > best_fitness
                if improved:
                    best_position = candidate.copy()
                    best_fitness = float(candidate_fitness)
                    improvements += 1
                    exhausted_dimensions.clear()

                if cache_hit:
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
                        "cache_hit_improved" if improved else "cache_hit_retry",
                        best_position,
                        best_fitness,
                    )
                    continue

                self._known_signatures.add(signature)
                successful_evaluations += 1
                attempts_since_success = 0
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
                    "improved" if improved else "evaluated",
                    best_position,
                    best_fitness,
                )
                found_unique = True
                break

            if not found_unique:
                exhausted_dimensions.add(dimension)
                if attempts_since_success < max_attempts:
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

        missing = target_evaluations - successful_evaluations
        if missing:
            self._log(
                "error",
                "IGWO random local search iteration=%s ended with %s missing "
                "unique evaluations after %s attempts",
                iteration,
                missing,
                attempts_since_success,
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
