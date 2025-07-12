# ===============================================
# abc_logger.py
# -----------------------------------------------
# Logging utility class for PMOABC algorithm.
# Author: Erdemhan Özdin (github.com/Erdemhan)
#
# Provides logging to:
# - Terminal (via print)
# - Plain text log file (.txt)
# - Structured JSON log file (.json)
# - Flat table in Excel format (.xlsx) for analysis
#
# Integration:
# - Used internally by MOABC to log iterations, population stats
# - Meta info and population states are structured for reproducibility
# ===============================================

import os
import json
import pandas as pd
from datetime import datetime

class ABCLogger:
    def __init__(self, algorithm_name: str, log_dir="abc_logs", meta_info: dict = None):
        # Ensure the logging directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create unique filenames with timestamp for json, txt, and Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{algorithm_name}_log_{timestamp}"
        self.json_path = os.path.join(log_dir, f"{base_name}.json")
        self.excel_path = os.path.join(log_dir, f"{base_name}.xlsx")
        self.txt_path = os.path.join(log_dir, f"{base_name}.txt")

        self.algorithm_name = algorithm_name
        self.meta_info = meta_info or {}
        self.data = []  # will hold structured iteration logs

    def log_text(self, message: str):
        """
        Logs a simple text message to both terminal and .txt file.
        """
        print(message)
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_iteration(self, iteration: int, population: list):
        """
        Stores structured data of population for the given iteration.
        Each parameter set and its scores are saved.
        Logs are stored in both JSON and Excel formats.
        """
        entry = {
            "algorithm": self.algorithm_name,
            "iteration": iteration,
            "population": population
        }

        # Write meta info only on first iteration
        if iteration == 0 and self.meta_info:
            entry["meta"] = self.meta_info

        self.data.append(entry)

        # Write JSON log
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        # Flatten for Excel
        flat_rows = []
        for e in self.data:
            for p in e["population"]:
                row = {"algorithm": self.algorithm_name, "iteration": e.get("iteration", -1), **p}
                flat_rows.append(row)

        df = pd.DataFrame(flat_rows)
        df.to_excel(self.excel_path, index=False)

    def log_initial_population(self, population: list):
        """
        Logs the initial randomly generated population before any iteration.
        Helps to track initial diversity or randomness.
        """
        entry = {
            "algorithm": self.algorithm_name,
            "type": "initial_population",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "population": population
        }
        self.data.append(entry)

        # Update JSON
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        # Flatten for Excel again
        flat_rows = []
        for e in self.data:
            if "population" not in e:
                continue
            for p in e["population"]:
                row = {"algorithm": self.algorithm_name, "type": e.get("type", "iteration"), **p}
                if "iteration" in e:
                    row["iteration"] = e["iteration"]
                flat_rows.append(row)

        df = pd.DataFrame(flat_rows)
        df.to_excel(self.excel_path, index=False)
