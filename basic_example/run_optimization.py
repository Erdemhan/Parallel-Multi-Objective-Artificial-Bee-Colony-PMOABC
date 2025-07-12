import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pmoabc import MOABC
from example_problem import simple_fitness

# Ortak ayarlar
COLONY_SIZE = 10
MAX_ITER = 20
LIMIT = 5
SEED = 42
SCORE_WEIGHTS = (0.5, 0.5)
ALGORITHM_TYPE = "SimpleDemo"

if __name__ == "__main__":

    with open("basic_example/param_bounds_example.json") as f:
        bounds = json.load(f)

    optimizer = MOABC(
        fitness_func=simple_fitness,
        param_bounds=bounds,
        colony_size=COLONY_SIZE,
        max_iter=MAX_ITER,
        limit=LIMIT,
        seed=SEED,
        score_weights=SCORE_WEIGHTS,  # her iki amaca eşit önem
        algorithm_type=ALGORITHM_TYPE
    )

    pareto_params, pareto_scores, best_params, best_score = optimizer.optimize()

    print("Pareto Front:")
    for p, s in zip(pareto_params, pareto_scores):
        print("Params:", p)
        print(f"Objective 1: {s[0]:.3f}, Objective 2: {s[1]:.3f}")
        print("---")
