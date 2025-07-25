# ==============================================
# hyperparam_optimization.py
# ----------------------------------------------
# This script defines how the MOABC algorithm interfaces with an external
# reinforcement learning simulator (Django-based project `abmem_web`).
# It creates simulations, runs them, evaluates outputs, and acts as the fitness function
# for optimization.
# 
# ✅ External dependency: https://github.com/Erdemhan/abmem_web
# This file assumes:
#   - You have a Django simulation model with proxy simulations.
#   - Offers are logged at `abm_ddpg/sim_data/simulation_{sim_id}_offers.json`
#   - Simulation state transitions match enum logic in `abmem.models.enums`
# ==============================================

import sys
import os
import numpy as np
import json
import pandas as pd
import copy

# Add project root to path for Django module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Set up Django context for database/model access
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
import django
django.setup()

from datetime import datetime
from abmem.services.simulation.simulation_service import run  # 🧠 External simulation execution logic
from abmem.models.simulation import Simulation
from abmem.models import enums
from django.shortcuts import get_object_or_404

from core.pmoabc import MOABC
from core.abc_logger import ABCLogger

# ========================
# Load hyperparameter bounds
# ========================
ALGORITHM_TYPE = None
with open("param_bounds.json", "r") as f:
    PARAM_BOUNDS = json.load(f)

# =====================================================
# Create a deep copy of an existing simulation proxy
# Used for starting clean RL simulations with same setup
# -----------------------------------------------------
# ⚠️ Requires that `abmem_web` database contains simulation with given proxy_id
# =====================================================
def create_simulation_from_proxy(proxy_id: int) -> Simulation:
    simulation = get_object_or_404(Simulation, id=proxy_id)
    if simulation.proxy or simulation.state != enums.SimulationState.CREATED:
        # Deep copy simulation
        newSimulation = copy.deepcopy(simulation)
        newSimulation.id = None
        newSimulation.state = enums.SimulationState.CREATED
        newSimulation.currentPeriod = -1
        newSimulation.proxy = False
        newSimulation.save()

        # Copy associated market
        newMarket = copy.deepcopy(simulation.market)
        newMarket.id = None
        newMarket.state = enums.MarketState.CREATED
        newMarket.simulation = newSimulation
        newMarket.proxy = False
        newMarket.save()

        # Copy each agent, its portfolio, and plants
        for agent in list(simulation.market.agent_set.all()):
            newAgent = copy.deepcopy(agent)
            newAgent.id = None
            newAgent.market = newMarket
            newAgent.proxy = False
            newAgent.save()

            newPortfolio = copy.deepcopy(agent.portfolio)
            newPortfolio.id = None
            newPortfolio.proxy = False
            newPortfolio.agent = newAgent
            newPortfolio.save()

            for plant in list(agent.portfolio.plant_set.all()):
                newPlant = copy.deepcopy(plant)
                newPlant.id = None
                newPlant.portfolio = newPortfolio
                newPlant.proxy = False
                newPlant.save()

        return newSimulation

    return simulation

# =====================================================
# Evaluate a finished simulation by reading its output JSON
# The output file must contain period-based market and agent offers
# =====================================================
def evaluate_result(sim_id: int) -> tuple:
    path = f"abm_ddpg/sim_data/simulation_{sim_id}_offers.json"  # ⚠️ Static path assumption
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for period, pdata in data["offers_by_period"].items():
        if not isinstance(pdata, dict): continue
        mcp = float(pdata.get("market_price", 0))
        for agent, offers in pdata.items():
            if agent == "market_price": continue
            for offer in offers:
                records.append({
                    "period": int(period),
                    "agent": offer["agent"],
                    "budget": float(offer["budget"]),
                    "accepted": offer["acceptance"],
                    "offer_price": float(offer["offerPrice"]),
                    "market_price": mcp,
                    "amount": float(offer["amount"]),
                    "resource": offer["resource"]
                })

    df = pd.DataFrame(records)
    if df.empty:
        return 0.0, 0.0  # Fallback in case of empty or invalid output

    # --- Agent score calculation ---
    price_cap = 200
    num_periods = df['period'].nunique()
    agents = df['agent'].unique()

    agent_scores = {}
    for agent in agents:
        adf = df[df['agent'] == agent]
        capacity = adf['amount'].iloc[0]
        initial_budget = adf.iloc[0]['budget']
        final_budget = adf.iloc[-1]['budget']
        raw_budget_change = final_budget - initial_budget
        max_possible_revenue = capacity * num_periods * price_cap
        norm_budget = raw_budget_change / max_possible_revenue if max_possible_revenue > 0 else 0
        norm_budget = max(-1.0, min(norm_budget, 1.0))

        accepted = adf[adf['accepted']].copy()
        total_profit = raw_budget_change
        total_amount = accepted['amount'].sum()
        avg_profit_per_unit = total_profit / total_amount if total_amount > 0 else 0
        norm_profit = avg_profit_per_unit / price_cap if price_cap > 0 else 0
        norm_profit = max(-1.0, min(norm_profit, 1.0))

        rmse = np.sqrt(((adf['offer_price'] - adf['market_price']) ** 2).mean())
        similarity = 1 - (rmse / price_cap)

        score = 0.4 * norm_budget + 0.4 * norm_profit + 0.2 * similarity
        score = max(0, min(score, 1))
        agent_scores[agent] = score

    agent_score = np.mean(list(agent_scores.values()))

    # --- Market score calculation ---
    mcp_avg = df['market_price'].mean()
    avg_offer_price = df['offer_price'].mean()
    mci = mcp_avg / price_cap
    mpi = mcp_avg / avg_offer_price if avg_offer_price > 0 else 0
    market_score = 0.6 * (1 - mci) + 0.4 * mpi

    return agent_score, market_score

# =====================================================
# Main simulation wrapper. Performs validation, simulation run,
# and evaluation — all in one. This function is used as fitness_func in MOABC.
# =====================================================
def simulate_multi(params: dict) -> tuple:
    if ALGORITHM_TYPE == "PPO" and not params.get("PPO_GAMMA"):
        raise ValueError("PPO_GAMMA eksik.")
    if ALGORITHM_TYPE == "SAC" and not params.get("LEARNING_RATE_ALPHA"):
        raise ValueError("LEARNING_RATE_ALPHA eksik.")

    sim = create_simulation_from_proxy(529)  # ⚠️ You must ensure proxy with id=529 exists in DB
    run(sim, hyperparams=params)             # 🔧 Core sim execution (abmem.services.simulation)
    agent_score, market_score = evaluate_result(sim.id)
    return agent_score, market_score, sim.id

# =====================================================
# Main callable for launching an optimization session
# Can be used standalone or via run_optimization.py
# =====================================================
def optimize_hyperparams(colony_size=2, max_iter=2, limit=10, seed=17081999,
                         algorithm_type="DDPG", logger: ABCLogger = None) -> None:
    global ALGORITHM_TYPE, PARAM_BOUNDS
    ALGORITHM_TYPE = algorithm_type

    abc = MOABC(
        fitness_func=simulate_multi,
        param_bounds=PARAM_BOUNDS,
        colony_size=colony_size,
        max_iter=max_iter,
        limit=limit,
        log_path=f"abc_logs/{algorithm_type}_log.json",
        seed=seed,
        algorithm_type=algorithm_type
    )

    pareto_params, pareto_scores, best_params, best_score = abc.optimize()

    if logger:
        logger.log_text(f"Global Best: {best_params} Score: {best_score}")
    else:
        print("Global Best:", best_params, "Score:", best_score)

    for p, s in zip(pareto_params, pareto_scores):
        out = f"Params: {p}\nAgent Score: {s[0]:.3f}, Market Score: {s[1]:.3f}\n---"
        if logger:
            logger.log_text(out)
        else:
            print(out)

# Manual run mode (if file is run directly)
if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    logger = ABCLogger("DDPG", "abc_logs")
    logger.log_text(f">>> Manual run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    optimize_hyperparams(logger=logger)
    logger.log_text(f"✅ Manual run finished in {timeit.default_timer() - start:.2f} seconds")
