from pathlib import Path
import yaml
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["train"]

X_train = pd.read_csv(ROOT / "data/processed/X_train.csv")
y_train = pd.read_csv(ROOT / "data/processed/y_train.csv")["target"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    min_samples_split=params["min_samples_split"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

out = ROOT / "models"
out.mkdir(exist_ok=True)
joblib.dump(model, out / "iris_random_forest.joblib")
print("Model trained and saved.")
