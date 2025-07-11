# Multi-Objective Artificial Bee Colony (MOABC) Algorithm
# ======================================================
# Author: [Erdemhan ÖZDİN github.com/Erdemhan erdemhan@erciyes.edu]
# Description:
# This Python class implements a variant of the Artificial Bee Colony (ABC) algorithm tailored for multi-objective optimization.
# It is designed to tune hyperparameters for reinforcement learning agents, but the design is modular and can be adapted to other optimization tasks.
# 
# The optimization is based on simulated evaluations of "food sources" (candidate solutions) and incorporates all 3 core ABC phases:
# - Employed Bees Phase: Exploration via neighborhood search.
# - Onlooker Bees Phase: Exploitation based on fitness-proportional selection.
# - Scout Bees Phase: Random exploration to escape local minima.
#
# Example usage:
# - Define a fitness function that takes in a parameter dictionary and returns a tuple of scores (e.g., accuracy, latency).
# - Define parameter bounds as a dictionary.
# - Instantiate MOABC and call `optimize()` to run the optimization.

import numpy as np
import random
from typing import Callable, Dict, Tuple, List
from multiprocessing import Pool, cpu_count
import os
from datetime import datetime
import timeit
from abc_logger import ABCLogger

class MOABC:
    def __init__(
        self,
        fitness_func: Callable[[Dict[str, float]], Tuple[float, float, int]],
        param_bounds: Dict[str, Tuple[float, float]],
        colony_size: int = 10,
        limit: int = 5,
        max_iter: int = 30,
        log_path: str = "abc_log.json",
        seed: int = 42,
        score_weights: Tuple[float, float] = (1, 0),
        algorithm_type: str = "DDPG"
    ):
        """
        Constructor for MOABC optimizer.
        Initializes the algorithm with provided configuration, sets up logging and generates initial population.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.fitness_func = fitness_func
        self.param_bounds = param_bounds
        self.colony_size = colony_size
        self.limit = limit
        self.max_iter = max_iter
        self.log_path = log_path
        self.score_weights = score_weights
        self.algorithm_type = algorithm_type

        self.param_names = list(param_bounds.keys())
        self.lb = np.array([param_bounds[k][0] for k in self.param_names])
        self.ub = np.array([param_bounds[k][1] for k in self.param_names])

        self.logger = ABCLogger(
            algorithm_name=self.algorithm_type,
            log_dir=os.path.splitext(log_path)[0],
            meta_info={
                "colony_size": colony_size,
                "limit": limit,
                "max_iter": max_iter,
                "seed": seed,
                "score_weights": score_weights,
                "param_bounds": param_bounds
            }
        )

        self.probabilities = np.ones(self.colony_size) / self.colony_size
        self.cycle = 0
        self.initialize_population()

    def initialize_population(self):
        """
        Generate the initial population and evaluate their fitness.
        """
        self.food_sources = np.random.uniform(self.lb, self.ub, (self.colony_size, len(self.param_names)))
        self.trial = np.zeros(self.colony_size)
        self.sim_ids = [None] * self.colony_size

        self.logger.log_text(f"Starting evaluation of initial population at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = timeit.default_timer()
        self.fitness = self.evaluate_parallel(self.food_sources)
        self.logger.log_text(f"Initial population evaluation completed in {timeit.default_timer() - start_time:.2f} seconds")

        self.best_solution = None
        self.best_score = (-1, -1)
        self.memorize_best_source()

        initial_data = []
        for idx in range(self.colony_size):
            initial_data.append({
                "params": self.to_dict(self.food_sources[idx]),
                "agent_score": self.fitness[idx][0],
                "market_score": self.fitness[idx][1],
                "sim_id": self.fitness[idx][2]
            })
        self.logger.log_initial_population(initial_data)
        self.logger.log_text(f"📌 Initial Best: {self.to_dict(self.best_solution)} Score: {self.best_score}")

    def memorize_best_source(self):
        """
        Update the best solution found so far by comparing all current food sources.
        If a better composite score is found, it is stored.
        """
        for i in range(self.colony_size):
            if self.composite_score(self.fitness[i]) > self.composite_score(self.best_score):
                self.best_score = self.fitness[i]
                self.best_solution = self.food_sources[i].copy()

    def to_dict(self, position: np.ndarray) -> Dict[str, float]:
        """
        Convert a NumPy position array into a dictionary with parameter names as keys.
        Applies rounding or integer conversion depending on parameter type.
        """
        return {
            name: (
                int(val) if 'ITERATION' in name.upper() or 'FAILSTACK' in name.upper() or "PPO_PPO_EPOCHS" in name.upper()
                else round(float(val), 5)
            )
            for name, val in zip(self.param_names, position)
        }

    def evaluate_parallel(self, positions: List[np.ndarray]) -> List[Tuple[float, float, int]]:
        """
        Evaluate a list of positions using multiprocessing.
        Each position is first converted to a parameter dictionary and passed to the fitness function.
        """
        param_list = [self.to_dict(pos) for pos in positions]

        for param in param_list:
            param["ALGORITHM"] = self.algorithm_type

        self.logger.log_text(f"🔬 Evaluating batch of {len(param_list)} individuals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        num_processes = min(cpu_count(), len(param_list))
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.fitness_func, param_list)
        return results

    def composite_score(self, score: Tuple[float, float]) -> float:
        """
        Calculate the weighted composite score used for optimization.
        You can modify the weights (score_weights) to prioritize objectives.
        """
        w1, w2 = self.score_weights
        return w1 * score[0] + w2 * score[1]

    def dominates(self, a, b):
        """
        Simple dominance check for Pareto comparisons.
        By default, prioritizes maximizing the first objective.
        Modify if needed for proper multi-objective comparisons.
        """
        return a[0] > b[0]

    def calculate_probabilities(self):
        """
        Calculate selection probabilities for the onlooker bee phase.
        The probabilities are based on normalized composite fitness scores.
        """
        scores = np.array([self.composite_score(f[:2]) for f in self.fitness])
        max_score = np.max(scores)

        if max_score == 0:
            self.probabilities = np.ones(self.colony_size) / self.colony_size
        else:
            self.probabilities = (0.9 * (scores / max_score)) + 0.1

# (Previously defined content remains unchanged...)
# ...

    def employed_bees_phase(self):
        """
        Employed bees phase of ABC algorithm.
        Each employed bee modifies its current solution by choosing a neighbor and applying a perturbation.
        If the new solution dominates the old one, it is accepted; otherwise, the trial count increases.
        """
        new_solutions = []
        indices = []
        for i in range(self.colony_size):
            k = random.choice([x for x in range(self.colony_size) if x != i])
            phi = np.random.uniform(-1, 1, len(self.param_names))
            new_sol = self.food_sources[i] + phi * (self.food_sources[i] - self.food_sources[k])
            new_sol = np.clip(new_sol, self.lb, self.ub)
            new_solutions.append(new_sol)
            indices.append(i)

        new_scores = self.evaluate_parallel(new_solutions)
        for i, new_sol, new_score in zip(indices, new_solutions, new_scores):
            if self.dominates(new_score[:2], self.fitness[i][:2]):
                self.food_sources[i] = new_sol
                self.fitness[i] = new_score
                self.trial[i] = 0
            else:
                self.trial[i] += 1
        self.memorize_best_source()

    def onlooker_bees_phase(self):
        """
        Onlooker bees phase where bees watch the dance (i.e., performance) of employed bees.
        Higher probability food sources are more likely to be chosen and modified.
        """
        self.calculate_probabilities()
        i = 0
        t = 0
        new_solutions = []
        indices = []

        while t < self.colony_size:
            r = random.random()
            if r < self.probabilities[i]:
                k = random.choice([x for x in range(self.colony_size) if x != i])
                phi = np.random.uniform(-1, 1, len(self.param_names))
                new_sol = self.food_sources[i] + phi * (self.food_sources[i] - self.food_sources[k])
                new_sol = np.clip(new_sol, self.lb, self.ub)
                new_solutions.append(new_sol)
                indices.append(i)
                t += 1
            i = (i + 1) % self.colony_size

        if new_solutions:
            new_scores = self.evaluate_parallel(new_solutions)
            for i, new_sol, new_score in zip(indices, new_solutions, new_scores):
                if self.dominates(new_score[:2], self.fitness[i][:2]):
                    self.food_sources[i] = new_sol
                    self.fitness[i] = new_score
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1
        self.memorize_best_source()

    def scout_bees_phase(self):
        """
        Scout bees phase replaces stagnated solutions with new random solutions.
        Any food source whose trial count exceeds the limit is replaced.
        """
        scout_indices = [i for i in range(self.colony_size) if self.trial[i] >= self.limit]
        if scout_indices:
            scout_solutions = [np.random.uniform(self.lb, self.ub) for _ in scout_indices]
            scout_scores = self.evaluate_parallel(scout_solutions)
            for i, new_sol, new_score in zip(scout_indices, scout_solutions, scout_scores):
                self.food_sources[i] = new_sol
                self.fitness[i] = new_score
                self.trial[i] = 0
        self.memorize_best_source()

    def find_pareto_front(self):
        """
        Identify the Pareto front of current solutions.
        A solution is added to the front if it is not dominated by any other.
        """
        pareto = []
        for i, fi in enumerate(self.fitness):
            dominated = False
            for j, fj in enumerate(self.fitness):
                if i != j and self.dominates(fj[:2], fi[:2]):
                    dominated = True
                    break
            if not dominated:
                pareto.append((i, fi))
        return pareto

    def log_iteration(self, iteration):
        """
        Log the current iteration's population and scores.
        Useful for visualization or post-optimization analysis.
        """
        population_data = []
        for idx in range(self.colony_size):
            params = self.to_dict(self.food_sources[idx])
            score = self.fitness[idx]
            population_data.append({
                "params": params,
                "agent_score": score[0],
                "market_score": score[1],
                "sim_id": score[2]
            })
        self.logger.log_iteration(iteration, population_data)

    def optimize(self):
        """
        The main loop of the ABC algorithm.
        Runs for a specified number of iterations and logs performance metrics.
        """
        for it in range(self.max_iter):
            self.logger.log_text(f"\n🔁 Iteration {it + 1}/{self.max_iter} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = timeit.default_timer()
            np.random.seed(self.seed + it)
            random.seed(self.seed + it)

            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()

            self.logger.log_text(f" -- Trial states: {[f'{i+1}- {int(self.trial[i])}/{self.limit}' for i in range(self.colony_size)]}")
            best_agent = max([s[0] for s in self.fitness])
            best_market = max([s[1] for s in self.fitness])
            self.logger.log_text(f"[Iter {it+1:02}/{self.max_iter}] Best Agent: {best_agent:.3f}, Market: {best_market:.3f}")
            self.log_iteration(it)
            self.logger.log_text(f"Iteration {it + 1} completed in {timeit.default_timer() - start_time:.2f} seconds")
            self.cycle += 1

        pareto_solutions = self.find_pareto_front()
        pareto_params = [self.to_dict(self.food_sources[i]) for i, _ in pareto_solutions]
        pareto_scores = [score[:2] for _, score in pareto_solutions]
        best_params = self.to_dict(self.best_solution)
        self.logger.log_text(f"\n✅ Global Best: {best_params} Score: {self.best_score}")
        return pareto_params, pareto_scores, best_params, self.best_score
