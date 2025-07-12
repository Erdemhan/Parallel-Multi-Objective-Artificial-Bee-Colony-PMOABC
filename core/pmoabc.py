# ===============================================
# pmoabc.py
# -----------------------------------------------
# Parallel Multi-Objective Artificial Bee Colony (PMOABC)
# Author: Erdemhan Özdin (github.com/Erdemhan)
#
# This module implements a general-purpose, extendable
# MOABC optimizer with logging and external simulation support.
#
# Key Concepts:
# - Population-based metaheuristic inspired by bee behavior
# - Multi-objective support with weighted scoring
# - Plug-and-play fitness function: works with RL or toy problems
# - Logs every stage to disk (via ABCLogger)
# - Supports external multiprocessing evaluation (RL simulation, etc.)
#
# External Simulator Integration (Optional):
# - Evaluation function can call a simulator (e.g., Django-based RL simulator)
# - Results must include 2 objectives (f1, f2) and optionally sim_id
#
# Compatible with:
# - simple_fitness() from `example_problem.py`
# - simulate_multi() from external RL simulator
# ===============================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import random
from typing import Callable, Dict, Tuple, List
from multiprocessing import Pool, cpu_count
from datetime import datetime
import timeit
from core.abc_logger import ABCLogger

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
        algorithm_type: str = "Generic"
    ):
        """
        Initialize the MOABC optimizer.

        Parameters:
        - fitness_func: Callable that takes parameter dict and returns (obj1, obj2, sim_id)
        - param_bounds: Dict with param names as keys, each value is a (min, max) tuple
        - colony_size: Number of candidate solutions (food sources)
        - limit: Trial limit before scout replaces a stagnant solution
        - max_iter: Number of optimization cycles
        - log_path: Where to store logging output
        - seed: For reproducibility
        - score_weights: Tuple defining relative importance of objectives
        - algorithm_type: Just a label for logging/identification
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

        # Extract param ranges for sampling
        self.param_names = list(param_bounds.keys())
        self.lb = np.array([param_bounds[k][0] for k in self.param_names])
        self.ub = np.array([param_bounds[k][1] for k in self.param_names])

        # Create logger instance
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
        Creates initial random population and evaluates them.
        Stores best solution based on composite score.
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
                "objective_1": self.fitness[idx][0],
                "objective_2": self.fitness[idx][1],
                "sim_id": self.fitness[idx][2]
            })
        self.logger.log_initial_population(initial_data)
        self.logger.log_text(f"📌 Initial Best: {self.to_dict(self.best_solution)} Score: objective_1={self.best_score[0]:.4f}, objective_2={self.best_score[1]:.4f}")

    def evaluate_parallel(self, positions: List[np.ndarray]) -> List[Tuple[float, float, int]]:
        """
        Evaluate a list of solutions using multiprocessing.
        Automatically handles integration with external simulators (e.g., Django).
        Each evaluation must return 2 objective scores and optionally a sim_id.
        """
        param_list = [self.to_dict(pos) for pos in positions]
        for param in param_list:
            param["ALGORITHM"] = self.algorithm_type  # Pass algorithm label if needed externally

        self.logger.log_text(f"🔬 Evaluating batch of {len(param_list)} individuals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        num_processes = min(cpu_count(), len(param_list))
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.fitness_func, param_list)
        adjusted_results = [r if len(r) == 3 else (r[0], r[1], None) for r in results]
        return adjusted_results

    def composite_score(self, score: Tuple[float, float]) -> float:
        """
        Returns scalar score using user-defined weights on (objective_1, objective_2).
        If you want pure Pareto front, use this only for selection purposes.
        """
        w1, w2 = self.score_weights
        return w1 * score[0] + w2 * score[1]

    def memorize_best_source(self):
        """
        Updates internal memory of the best solution based on current scores.
        Called after each major ABC phase (employed, onlooker, scout).
        """
        for i in range(self.colony_size):
            if self.best_solution is None:
                self.best_score = self.fitness[i]
                self.best_solution = self.food_sources[i].copy()
            elif self.composite_score(self.fitness[i]) > self.composite_score(self.best_score):
                self.best_score = self.fitness[i]
                self.best_solution = self.food_sources[i].copy()

    def to_dict(self, position: np.ndarray) -> Dict[str, float]:
        """
        Converts internal NumPy array into dict of parameter values.
        Rounds or casts to int depending on param name convention.
        """
        return {
            name: (
                int(val) if 'ITERATION' in name.upper() or 'FAILSTACK' in name.upper() or "PPO_PPO_EPOCHS" in name.upper()
                else round(float(val), 5)
            )
            for name, val in zip(self.param_names, position)
        }
    
    def calculate_probabilities(self):
            """
            Normalizes the composite scores to derive probabilities
            for selection in the onlooker bee phase. Ensures that
            all candidates have non-zero selection chance.
            """
            scores = np.array([self.composite_score(f[:2]) for f in self.fitness])
            max_score = np.max(scores)
            if max_score == 0:
                self.probabilities = np.ones(self.colony_size) / self.colony_size
            else:
                self.probabilities = (0.9 * (scores / max_score)) + 0.1

    def dominates(self, a, b):
        """
        Check if solution a dominates b.
        Only first objective is used by default.
        Extend this for proper Pareto dominance if needed.
        """
        return a[0] > b[0]

    def employed_bees_phase(self):
        """
        Employed bees modify their current solutions by comparing
        against randomly selected neighbors. New solutions are
        evaluated and kept if better.
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
        Onlooker bees select food sources based on probability
        and generate new candidate solutions. New solutions
        are accepted if they dominate the old ones.
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
        Replace stagnated solutions (exceeding limit) with
        entirely new random solutions.
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
        Identify non-dominated solutions in the population
        to construct a simple Pareto front.
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
        Logs detailed info of population for this iteration.
        Each individual's params and scores are saved.
        """
        population_data = []
        for idx in range(self.colony_size):
            params = self.to_dict(self.food_sources[idx])
            score = self.fitness[idx]
            population_data.append({
                "params": params,
                "objective_1": score[0],
                "objective_2": score[1]
            })
        self.logger.log_iteration(iteration, population_data)

    def optimize(self):
        """
        Main loop of PMOABC algorithm.
        Executes phases in order and logs progress.
        Returns Pareto front and best solution.
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
