import json
from moabc import MOABC
from example_problem import simple_fitness

with open("param_bounds.json") as f:
    bounds = json.load(f)

optimizer = MOABC(
    fitness_func=simple_fitness,
    param_bounds=bounds,
    colony_size=10,
    max_iter=20,
    limit=5,
    seed=42,
    score_weights=(0.5, 0.5),  # her iki amaca eşit önem
    algorithm_type="SimpleDemo"
)

pareto_params, pareto_scores, best_params, best_score = optimizer.optimize()

print("Pareto Front:")
for p, s in zip(pareto_params, pareto_scores):
    print("Params:", p, "Scores:", s)
