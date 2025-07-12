# ===============================================
# example_problem.py
# -----------------------------------------------
# This module defines a dummy fitness function for testing
# the PMOABC optimizer in a general context.
# It uses two simple objectives: proximity to two fixed points.
# This example does NOT require any external simulator.
# Ideal for debugging or demonstration of MOABC behavior.
# ===============================================

from typing import Dict, Tuple

# ---------------------------------------------------
# Fitness function that defines two synthetic objectives:
# f1: distance from point (2, 3)
# f2: distance from point (7, 5)
# These are used as surrogate objectives for optimization.
# ---------------------------------------------------
def simple_fitness(params: Dict[str, float]) -> Tuple[float, float, int]:
    x = params["x"]
    y = params["y"]

    # Objective 1: maximize negative Euclidean distance to (2,3)
    f1 = -((x - 2) ** 2 + (y - 3) ** 2)

    # Objective 2: maximize negative Euclidean distance to (7,5)
    f2 = -((x - 7) ** 2 + (y - 5) ** 2)

    return f1, f2, -1  # `-1` is a placeholder sim_id; ignored by general-purpose runs