from typing import Dict, Tuple

def simple_fitness(params: Dict[str, float]) -> Tuple[float, float, int]:
    x = params["x"]
    y = params["y"]

    # Objective 1: maximize negative distance from (2,3)
    f1 = -((x - 2) ** 2 + (y - 3) ** 2)

    # Objective 2: maximize negative distance from (7,5)
    f2 = -((x - 7) ** 2 + (y - 5) ** 2)

    return f1, f2, -1  # -1 sim_id yerine placeholder
