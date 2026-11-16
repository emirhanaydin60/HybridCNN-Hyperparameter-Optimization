import time

import numpy as np

from .base_optimizer import BaseOptimizer


class GreyWolfOptimizer(BaseOptimizer):
    def optimize(self):
        total_start = time.perf_counter()
        wolves = self._initialize()
        initial_population_positions = [wolf.tolist() for wolf in wolves]
        initial_population_signatures = [
            list(self._signature(wolf)) for wolf in wolves
        ]
        fitness = [self._evaluate(wolf, iteration=0, agent_id=index + 1) for index, wolf in enumerate(wolves)]
        pbest_pos = [wolf.copy() for wolf in wolves]
        pbest_fit = fitness.copy()
        global_bests = []
        local_bests = []
        iteration_times = []
        iteration_summaries = [self._summarize_iteration(0, fitness, wolves, pbest_fit, total_start)]
        evaluation_count = len(wolves)

        for iteration in range(1, self.iterations + 1):
            iter_start = time.perf_counter()
            ranking = np.argsort(fitness)[::-1]
            alpha = wolves[ranking[0]]
            beta = wolves[ranking[1]] if self.population > 1 else wolves[ranking[0]]
            delta = wolves[ranking[2]] if self.population > 2 else wolves[ranking[0]]
            a = 2 - (iteration - 1) * (2 / max(self.iterations - 1, 1))

            for wolf_id in range(self.population):
                position = wolves[wolf_id]
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

            fitness = [self._evaluate(wolf, iteration=iteration, agent_id=index + 1) for index, wolf in enumerate(wolves)]
            evaluation_count += len(wolves)

            for wolf_index in range(self.population):
                if fitness[wolf_index] > pbest_fit[wolf_index]:
                    pbest_fit[wolf_index] = fitness[wolf_index]
                    pbest_pos[wolf_index] = wolves[wolf_index].copy()

            local_bests.append(pbest_fit.copy())
            global_bests.append(float(np.max(pbest_fit)))
            iteration_times.append(time.perf_counter() - iter_start)
            iteration_summaries.append(self._summarize_iteration(iteration, fitness, wolves, pbest_fit, iter_start))

        best_index = int(np.argmax(pbest_fit))
        best_pos = pbest_pos[best_index]
        return {
            "best_pos": best_pos,
            "best_conf": best_pos,
            "best_fitness": float(pbest_fit[best_index]),
            "best_index": best_index + 1,
            "initial_population_positions": initial_population_positions,
            "initial_population_signatures": initial_population_signatures,
            "global_bests": global_bests,
            "local_bests": local_bests,
            "iteration_times": iteration_times,
            "iteration_summaries": iteration_summaries,
            "total_optimization_time": time.perf_counter() - total_start,
            "evaluation_count": evaluation_count,
        }
