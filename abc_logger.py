# abc_logger.py
import os
import json
import pandas as pd
from datetime import datetime

class ABCLogger:
    def __init__(self, algorithm_name: str, log_dir="abc_logs", meta_info: dict = None):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ✅ Algoritma ismi dosya adının başına gelecek
        base_name = f"{algorithm_name}_log_{timestamp}"
        
        self.json_path = os.path.join(log_dir, f"{base_name}.json")
        self.excel_path = os.path.join(log_dir, f"{base_name}.xlsx")
        self.txt_path = os.path.join(log_dir, f"{base_name}.txt")
        
        self.algorithm_name = algorithm_name
        self.meta_info = meta_info or {}
        self.data = []

    def log_text(self, message: str):
        print(message)
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


    def log_iteration(self, iteration: int, population: list):
        entry = {
            "algorithm": self.algorithm_name,
            "iteration": iteration,
            "population": population
        }

        if iteration == 0 and self.meta_info:
            entry["meta"] = self.meta_info  # sadece ilk iterasyona yaz

        self.data.append(entry)

        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        flat_rows = []
        for e in self.data:
            for p in e["population"]:
                row = {"algorithm": self.algorithm_name, "iteration": iteration, **p}
                flat_rows.append(row)
        df = pd.DataFrame(flat_rows)
        df.to_excel(self.excel_path, index=False)


    def log_initial_population(self, population: list):
        """
        Başlangıç popülasyonu parametre ve skorlarını 'initial' adıyla kaydeder.
        """
        entry = {
            "algorithm": self.algorithm_name,
            "type": "initial_population",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "population": population
        }
        self.data.append(entry)

        # JSON ve Excel güncelle
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

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

    def log_text(self, message: str):
        """
        Hem terminale yaz, hem de txt log dosyasına ekle.
        """
        print(message)
        txt_path = self.json_path.replace(".json", ".txt")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
