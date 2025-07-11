from hyperparam_optimization import optimize_hyperparams
from abc_logger import ABCLogger
import timeit
from datetime import datetime
import os

# Ortak ayarlar
COLONY_SIZE = 2
MAX_ITER = 2
LIMIT = 5
SEED = 17081999
ALGORITHMS = ["DDPG"] #"RDPG", "DDPG",  "TD3", "PPO", "SAC"

if __name__ == "__main__":
    for algorithm in ALGORITHMS:
        logger = ABCLogger(
            algorithm_name=algorithm,
            log_dir="abc_logs",
            meta_info={
                "colony_size": COLONY_SIZE,
                "max_iter": MAX_ITER,
                "limit": LIMIT,
                "seed": SEED
            }
        )

        logger.log_text(f"\n>>> Starting {algorithm} optimization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<")

        start = timeit.default_timer()
        optimize_hyperparams(
            colony_size=COLONY_SIZE,
            max_iter=MAX_ITER,
            limit=LIMIT,
            seed=SEED,
            algorithm_type=algorithm
        )
        elapsed = timeit.default_timer() - start
        logger.log_text(f"✅ {algorithm} optimization completed in {elapsed:.2f} seconds")
