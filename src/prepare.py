from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["prepare"]

data = pd.read_csv(ROOT / "data/raw/iris.csv")
X = data.drop(columns=["target", "target_name"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"],
    random_state=params["random_state"], stratify=y
)

out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)
X_train.to_csv(out / "X_train.csv", index=False)
X_test.to_csv(out / "X_test.csv", index=False)
y_train.to_csv(out / "y_train.csv", index=False, header=["target"])
y_test.to_csv(out / "y_test.csv", index=False, header=["target"])
joblib.dump(list(X.columns), out / "feature_names.joblib")
print(f"Prepared train={len(X_train)} test={len(X_test)}")
