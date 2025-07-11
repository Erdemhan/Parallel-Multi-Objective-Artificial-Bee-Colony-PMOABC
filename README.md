# 🚀 Parallel Multi-Objective Artificial Bee Colony (MOABC) Optimizer

Welcome to **MOABC**, a flexible and parallelizable implementation of the **Multi-Objective Artificial Bee Colony** algorithm designed for **hyperparameter optimization**, especially in **reinforcement learning** scenarios.

---

## 💡 What is MOABC?

MOABC is an advanced variant of the Artificial Bee Colony algorithm that:

* Supports **multi-objective optimization** with customizable score weights
* Evaluates candidate solutions **in parallel** using multiprocessing
* Logs rich metadata, parameter sets, and performance metrics
* Can be adapted for **any black-box optimization problem**

---

## 🌐 Repository Structure

```
.
├── abc_logger.py              # Logging utility for MOABC
├── moabc.py                   # Main MOABC class
├── hyperparam_optimization.py # RL simulator integration
├── optimizer.py               # Entry point for batch algorithm optimization
├── run_optimization.py        # Simple test example
├── example_problem.py         # Toy benchmark for testing
├── param_bounds.json          # Hyperparameter search ranges
└── README.md
```

---

## 🔄 Evaluation Integration with RL Simulator

This optimizer **relies on an external RL simulator** for evaluation. The evaluate phase in `hyperparam_optimization.py` is tightly integrated with the following repository:

> ⚡ **[https://github.com/Erdemhan/abmem\_web](https://github.com/Erdemhan/abmem_web)**

To use the `simulate_multi` function:

* Set up the Django-based simulator project `abmem_web`.
* Adjust `DJANGO_SETTINGS_MODULE` in `hyperparam_optimization.py` if needed.
* Ensure simulation `id=529` or a suitable proxy simulation exists.
* The simulation must write offer results to `abm_ddpg/sim_data/simulation_{sim_id}_offers.json` for scoring.

---

## 🌟 Example Usage

### 🔧 Minimal Setup

```bash
python run_optimization.py
```

This runs MOABC on a toy problem (`example_problem.py`) with two objectives.

### 🔬 RL Hyperparameter Optimization

```bash
python optimizer.py
```

This will launch MOABC to optimize RL hyperparameters for DDPG (or others).

---

## 📖 Adapting MOABC to Your Problem

To integrate MOABC with your own problem:

### 1. 🔢 Define a Fitness Function

Write a Python function that takes a `Dict[str, float]` and returns a `Tuple[float, float, int]`:

```python
def my_fitness(params: Dict[str, float]) -> Tuple[float, float, int]:
    # Your simulation or scoring logic here
    return score1, score2, sim_id  # sim_id is optional (used for logging)
```

### 2. 📊 Create Parameter Bounds File

```json
{
  "LEARNING_RATE": [1e-5, 1e-2],
  "DISCOUNT": [0.9, 0.999]
}
```

### 3. 📄 Initialize MOABC

```python
from moabc import MOABC
with open("param_bounds.json") as f:
    bounds = json.load(f)

optimizer = MOABC(
    fitness_func=my_fitness,
    param_bounds=bounds,
    colony_size=10,
    max_iter=30,
    limit=5,
    score_weights=(0.5, 0.5),
    algorithm_type="MyAlgorithm"
)
optimizer.optimize()
```

---

## 🔍 Logging & Output

* Logs stored in: `abc_logs/{algorithm_name}_log_*.json/.txt/.xlsx`
* Initial population and each iteration's data is saved for post-analysis.
* Composite scores use `score_weights=(w1, w2)` to weight the two objectives.

---

## 🛠️ Advanced Options

| Option          | Description                                     |
| --------------- | ----------------------------------------------- |
| `colony_size`   | Number of candidate solutions (bees)            |
| `limit`         | Trial limit before scout replacement            |
| `max_iter`      | Number of optimization iterations               |
| `score_weights` | Tuple `(w1, w2)` to weigh multi-objective goals |
| `seed`          | Random seed for reproducibility                 |

---

## 🎓 License

This work is licensed under a **Creative Commons Attribution-NonCommercial 4.0 International License**.

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

> You are free to:
>
> * Share — copy and redistribute the material
> * Adapt — remix, transform, and build upon
>
> Under the following terms:
>
> * ✉ Attribution required
> * ❌ NonCommercial use only

---

## 📈 Citation

If you use this repository in academic work, please cite:

```text
@software{ozdin_moabc,
  author = {Erdemhan Özdin},
  title = {Parallel Multi-Objective ABC Optimization},
  url = {https://github.com/Erdemhan},
  year = 2025
}
```
