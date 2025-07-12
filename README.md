# 📘 PMOABC: Parallel Multi-Objective Artificial Bee Colony Optimizer

---

PMOABC is a flexible, general-purpose implementation of the Multi-Objective Artificial Bee Colony algorithm. It is designed for **hyperparameter optimization**, especially in **reinforcement learning (RL)** tasks, but can also be applied to generic multi-objective problems without modification.

---

## 💡 What is MOABC?

MOABC is an advanced variant of the Artificial Bee Colony algorithm that:

* Supports **multi-objective optimization** with customizable score weights
* Evaluates candidate solutions **in parallel** using multiprocessing
* Logs rich metadata, parameter sets, and performance metrics
* Can be adapted for **any black-box optimization problem**


## ✨ Features

- ✅ **General-Purpose Support**: Works with any fitness function returning two objectives.
- 🔄 **Parallel Evaluation**: Multiprocessing support for fast evaluation.
- 🧠 **Modular & Extensible**: Can be used with toy benchmarks or integrated RL simulators.
- 🐝 **ABC Phases**: Employed, Onlooker, and Scout bee logic implemented.
- 📊 **Advanced Logging**: Logs in `.json`, `.txt`, and `.xlsx` formats via `ABCLogger`.
- 📁 **Separation of Concerns**: `core/`, `basic_example/`, and `advanced_rl/` organization for clarity.

---

## 🔧 Folder Structure

```text
.
├── core/
│   ├── pmoabc.py               # General-purpose PMOABC optimizer
│   └── abc_logger.py           # Modular logger for text/JSON/Excel export
├── basic_example/
│   ├── run_optimization.py     # Simple benchmark usage
│   └── example_problem.py      # Toy multi-objective test case
├── advanced_rl/
│   ├── run_optimization.py     # RL-specific optimization runner
│   ├── hyperparam_optimization.py # RL simulator integration
│   └── param_bounds_advanced.json # RL hyperparameter bounds
├── README.md
└── ...
```

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Erdemhan/Parallel-Multi-Objective-Artificial-Bee-Colony-PMOABC.git
cd Parallel-Multi-Objective-Artificial-Bee-Colony-PMOABC
```

### Run Toy Example
```bash
cd basic_example
python run_optimization.py
```

### Run RL Example
Requires external RL simulator (see below).
```bash
cd advanced_rl
python run_optimization.py
```

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

## 🛠️ Advanced Options

| Option          | Description                                     |
| --------------- | ----------------------------------------------- |
| `colony_size`   | Number of candidate solutions (bees)            |
| `limit`         | Trial limit before scout replacement            |
| `max_iter`      | Number of optimization iterations               |
| `score_weights` | Tuple `(w1, w2)` to weigh multi-objective goals |
| `seed`          | Random seed for reproducibility                 |

---


## 📈 Logging Output

Each run produces a timestamped set of logs:

- `*.json`: All iterations, parameters, and objective values (machine-readable)
- `*.txt`: Human-readable summary logs with cycle-by-cycle commentary
- `*.xlsx`: Excel export of all evaluations for filtering, visualization, etc.

### What Logs Contain
Each log file records:
- Initial population and their fitness values
- Iteration-wise population state
- Trial counters for each food source
- Best solutions found across cycles
- Final Pareto front and global best solution

> 🔎 All logs are stored under a folder auto-named using the `algorithm_name` and current timestamp. This folder is derived from the `log_path` parameter.

---

## 🔄 Evaluation Integration with RL Simulator

This optimizer **relies on an external RL simulator** for evaluation in advanced use cases. The evaluate phase in `hyperparam_optimization.py` is tightly integrated with the following repository:

> ⚡ **[https://github.com/Erdemhan/abmem_web](https://github.com/Erdemhan/abmem_web)**

To use the `simulate_multi` function:

1. Set up the Django-based simulator project `abmem_web`
2. Adjust `DJANGO_SETTINGS_MODULE` in `hyperparam_optimization.py` if needed
3. Ensure simulation `id=529` or a suitable proxy simulation exists
4. The simulation must write offer results to:
   ```bash
   abm_ddpg/sim_data/simulation_{sim_id}_offers.json
   ```

The optimizer includes helper functions such as `create_simulation_from_proxy` and `simulate_multi`, which allow evaluation of candidate solutions within the external simulator. These act as bridges between PMOABC and the RL-based market environment.

---

## 📌 Integration Notes

- `fitness_func` must return **exactly 3 values**: `(objective_1, objective_2, extra_info)`
    - `extra_info` can be `None`, a simulation ID, or a placeholder (e.g. -1)
- `param_bounds_advanced.json` holds hyperparameter ranges for each algorithm
- Logger auto-generates a folder based on algorithm name and timestamp
- If your fitness function accidentally returns only 2 values, PMOABC will raise an error
- `score_weights=(1, 0)` will fully prioritize `objective_1`; adjust for custom priorities
- All runs are reproducible via `seed` parameter

---

## 👤 Author

Developed by **Erdemhan Özdin**  
📬 [erdemhan@erciyes.edu.tr](mailto:erdemhan@erciyes.edu.tr)  
🔗 [github.com/Erdemhan](https://github.com/Erdemhan)


---

## 📈 Citation

If you use this repository in academic work, please cite:

```text
@software{ozdin_moabc,
  author = {Erdemhan Özdin},
  title = {Parallel Multi-Objective ABC Optimization},
  url = {https://github.com/Erdemhan/Parallel-Multi-Objective-Artificial-Bee-Colony-PMOABC.git},
  year = 2025
}
```

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


