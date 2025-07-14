# yolo_fitness.py
import os
import shutil
import uuid
from ultralytics import YOLO

# Sabitler
MODEL_PATH = "yolo12s.pt"
DATASET_YAML = "dataset.yaml"  # Kullanıcının verdiği .yaml dosyası
OUTPUT_DIR = "runs/abc_yolo"
LOSS_MAP = {0: "auto", 1: "bce", 2: "focal"}

def yolo_fitness(params):
    # Benzersiz isimlendirme
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Kategorik parametreyi dönüştür
    loss_type = LOSS_MAP.get(int(round(params.get("loss_function", 0))), "auto")

    # Modeli yükle
    model = YOLO(MODEL_PATH)

    # Loss fonksiyonunu elle ayarla
    if loss_type != "auto":
        model.model.loss = model.model.loss.__class__(loss_type=loss_type)

    # Eğitimi başlat
    try:
        results = model.train(
            data=DATASET_YAML,
            epochs=int(params.get("epochs", 50)),
            imgsz=int(params.get("imgsz", 640)),
            batch=int(params.get("batch", 16)),
            lr0=float(params.get("lr0", 0.001)),
            momentum=float(params.get("momentum", 0.9)),
            project=OUTPUT_DIR,
            name=run_id,
            exist_ok=True,
            verbose=False
        )
    except Exception as e:
        print(f"Training failed for {run_id}: {e}")
        return (0.0, 0.0, run_id)  # objective1, objective2, sim_id

    # Sonuçları al
    try:
        metrics = model.trainer.metrics
        map50 = metrics.get("metrics/mAP50(B)" , 0.0)
        return (map50, 0.0, run_id)
    except:
        return (0.0, 0.0, run_id)
    finally:
        # Dilersen run_dir'i temizle: shutil.rmtree(run_dir)
        pass