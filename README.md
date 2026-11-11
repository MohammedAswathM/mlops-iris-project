# Reproducible Iris ML Pipeline with Git, DVC and CML

A small end-to-end MLOps lab demonstrating data preparation, model training,
evaluation, DVC pipelines, experiment comparison, and GitHub Actions + CML.

## 1. Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Generate the source dataset and run the pipeline

```bash
python src/generate_data.py
dvc repro
```

The dataset is the canonical Iris dataset distributed by scikit-learn.
The script writes a deterministic CSV so the same source can be recreated in CI.

## 3. Git and DVC

```bash
git init
dvc init
dvc add data/raw/iris.csv
git add .
git commit -m "Initialize reproducible Iris MLOps project"
```

Configure a DVC remote (example: an S3 bucket):

```bash
dvc remote add -d storage s3://YOUR-BUCKET/YOUR-PREFIX
# Configure AWS credentials using your normal secure credential mechanism.
dvc push
```

Do not commit access keys. For GitHub Actions, store cloud credentials as
repository secrets and configure the corresponding environment variables.

A local remote can be used for a classroom demonstration, but it is not shared
storage and will not be available to GitHub-hosted runners:

```bash
mkdir -p ../dvc-storage
dvc remote add -d localstorage ../dvc-storage
dvc push
```

## 4. Compare experiments

Run the baseline first and commit the resulting metrics/lock file:

```bash
dvc repro
git add dvc.lock reports/metrics.json
git commit -m "Record baseline experiment"
```

Change `train.n_estimators` in `params.yaml` (for example, 100 to 200), then:

```bash
dvc repro
dvc metrics diff
dvc params diff
```

Record the output and explain whether the metric changed. On Iris, a small test
set can produce identical scores for different settings; do not invent an
improvement. The important evidence is that DVC detects the parameter change
and reruns the affected stages.

## 5. GitHub Actions and CML

The workflow is in `.github/workflows/ml-pipeline.yaml`. Push this project to
GitHub, create a branch and pull request, and inspect the Actions run and PR
comment. For CML, the workflow uses `cml comment create` to publish the metrics.

For a public repository, `GITHUB_TOKEN` may be sufficient for the PR comment if
the workflow has `pull-requests: write` and `contents: read`. For private or
restricted organization repositories, follow the organization's token policy.

## 6. Jupyter notebook evidence

Open `notebooks/iris_experiment.ipynb`, run all cells, and save the notebook
with the printed accuracy, weighted F1-score and confusion matrix.

## Reproducibility notes

- Fixed random seeds and stratified splitting are used.
- Hyperparameters are externalized in `params.yaml`.
- DVC records pipeline dependencies, parameters, outputs and metrics.
- Git records source code and lightweight metadata; DVC stores large data/model
  artifacts in its cache and configured remote.
- Accuracy and F1 are reported on the held-out test split.
