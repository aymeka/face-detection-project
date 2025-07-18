from pathlib import Path
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
YOLO_YAML = DATA_DIR / "yolo" / "data.yaml"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

def save_confusion_matrix(cm, labels, model_name):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"confusion_{model_name.lower()}.png")
    plt.close()

# ================================================================

print("\n✅ YOLOv8 kontrol ediliyor…")
runs_detect = Path.cwd() / "runs" / "detect"
candidates = sorted(
    [p for p in runs_detect.glob("yolov8_face*") if (p / "weights" / "best.pt").exists()],
    key=lambda x: x.stat().st_mtime
)

if candidates:
    latest_run = candidates[-1]
    src_best = latest_run / "weights" / "best.pt"
    print(f"✅ Önceden eğitilmiş YOLOv8 bulundu: {src_best}")
else:
    print(f"🚀 YOLOv8 eğitiliyor…")
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.train(data=str(YOLO_YAML), epochs=15, imgsz=640, name="yolov8_face")
    metrics_yolo = yolo_model.val(data=str(YOLO_YAML), split='test')
    latest_run = sorted([p for p in runs_detect.glob("yolov8_face*")], key=lambda x: x.stat().st_mtime)[-1]
    src_best = latest_run / "weights" / "best.pt"

yolo_weights = MODELS_DIR / "yolov8.pt"
copyfile(src_best, yolo_weights)

metrics_yolo = yolo_model.val(data=str(YOLO_YAML), split='test')

results.append({
    "model": "YOLOv8",
    "mAP50": metrics_yolo.box.map50,
    "mAP50-95": metrics_yolo.box.map,
    "precision": metrics_yolo.box.precision,
    "recall": metrics_yolo.box.recall,
    "weight_path": str(yolo_weights)
})


df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

best_row = df.loc[df['mAP50'].idxmax()]
best_model_path = MODELS_DIR / "best.pt"
copyfile(best_row['weight_path'], best_model_path)

print(f"\n🏆 En iyi model: {best_row['model']} (mAP50={best_row['mAP50']:.2f})")
print(f"✅ En iyi model {best_model_path} olarak kaydedildi.")
