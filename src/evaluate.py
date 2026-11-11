from pathlib import Path
import json
import yaml
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["evaluate"]

model = joblib.load(ROOT / "models/iris_random_forest.joblib")
X_test = pd.read_csv(ROOT / "data/processed/X_test.csv")
y_test = pd.read_csv(ROOT / "data/processed/y_test.csv")["target"]
pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_weighted": float(precision_score(y_test, pred, average=params["average"], zero_division=0)),
    "recall_weighted": float(recall_score(y_test, pred, average=params["average"], zero_division=0)),
    "f1_weighted": float(f1_score(y_test, pred, average=params["average"], zero_division=0))
}
report_dir = ROOT / "reports"
report_dir.mkdir(exist_ok=True)
(report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(report_dir / "classification_report.txt").write_text(
    classification_report(y_test, pred, zero_division=0), encoding="utf-8"
)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred))
disp.plot(cmap="Blues")
plt.title("Iris Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(report_dir / "confusion_matrix.png", dpi=160)
print(json.dumps(metrics, indent=2))
