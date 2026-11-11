from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

OUT = Path("data/raw/iris.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df = df.rename(columns={"target": "target"})
df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))
df.to_csv(OUT, index=False)
print(f"Generated {len(df)} rows at {OUT}")
