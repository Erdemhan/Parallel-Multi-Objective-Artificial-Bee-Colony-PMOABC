import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from core.pmoabc import MOABC
from yolo_fitness import yolo_fitness

# Ortak ayarlar
COLONY_SIZE = 6
MAX_ITER = 10
LIMIT = 5
SEED = 42
SCORE_WEIGHTS = (1.0, 0.0)  # mAP50'e odaklan

if __name__ == "__main__":

    with open("param_bounds.json") as f:
        bounds = json.load(f)

    optimizer = MOABC(
        fitness_func=yolo_fitness,
        param_bounds=bounds,
        colony_size=COLONY_SIZE,
        max_iter=MAX_ITER,
        limit=LIMIT,
        seed=SEED,
        score_weights=SCORE_WEIGHTS,
        algorithm_type="YOLOv12_ABC"
    )

    pareto_params, pareto_scores, best_params, best_score = optimizer.optimize()

    print("Pareto Front:")
    for p, s in zip(pareto_params, pareto_scores):
        print("Params:", p)
        print(f"mAP50: {s[0]:.3f}")
        print("---")
