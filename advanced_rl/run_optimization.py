# ============================================
# ADVANCED RL INTEGRATION - PMOABC OPTIMIZATION
# ============================================
# This script manages hyperparameter optimization for advanced RL agents (DDPG, PPO, SAC, etc.)
# using the PMOABC (Multi-Objective Artificial Bee Colony) algorithm.
# It integrates tightly with an external Django-based simulator found at:
# https://github.com/Erdemhan/abmem_web
# --------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hyperparam_optimization import optimize_hyperparams  # Core entry to run optimization loop using external simulator
from core.abc_logger import ABCLogger  # Logger handles .json/.txt/.xlsx structured outputs
import timeit
from datetime import datetime

# ========== USER CONFIGURATION SECTION ==========
# You may change these to control population size and exploration depth.
COLONY_SIZE = 2      # Number of candidate solutions (bees)
MAX_ITER = 2         # Number of main optimization iterations
LIMIT = 5            # Limit for scout bee replacement (when a solution stagnates)
SEED = 17081999      # For reproducibility

# Algorithms to be evaluated in batch mode. These must be supported in both `simulate_multi()` logic and RL simulator.
ALGORITHMS = ["DDPG", "RDPG", "DDPG", "TD3", "PPO", "SAC"]

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    for algorithm in ALGORITHMS:
        # Create dedicated logger for each algorithm
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

        # Main optimization logic; internally creates MOABC instance and runs simulation-based fitness evaluations
        optimize_hyperparams(
            colony_size=COLONY_SIZE,
            max_iter=MAX_ITER,
            limit=LIMIT,
            seed=SEED,
            algorithm_type=algorithm
        )

        elapsed = timeit.default_timer() - start
        logger.log_text(f"✅ {algorithm} optimization completed in {elapsed:.2f} seconds")
        logger.log_text(f">>> Finished {algorithm} optimization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<")