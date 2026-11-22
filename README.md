# Predictive Maintenance for Turbofan Engines

NASA C-MAPSS experiments for remaining useful life (RUL) prediction and survival analysis. Active experiments use FD001; the other supplied subsets are retained for future work.

## Project layout

```text
.
|-- notebooks/             # Numbered analysis and training workflows
|-- src/                   # Reusable model inference and FastAPI application
|-- scripts/               # Serving-release preparation and HTTP benchmark
|-- data/
|   |-- raw/cmapss/         # Original FD001-FD004 training, test and RUL files
|   `-- processed/         # Prepared sequences and model input tables
|-- models/                # Current LSTM and Cox models, scalers and metrics
|-- reports/
|   |-- figures/           # Current LSTM/SHAP and survival plots
|   `-- tables/            # Survival predictions and statistical reports
|-- references/            # Dataset documentation and original project notes
|-- docs/                  # Project status and reorganization manifest
|-- archive/               # Unfinished deployment drafts and one source backup
|-- tests/                 # Schema, inference, HTTP, startup and project checks
|-- requirements.txt       # Python dependencies
`-- environment.yml        # Optional new Conda environment
```

## Open and run the notebooks

Use your existing `exam_env` environment in VS Code/Jupyter. Restart its kernel after dependency changes.

```powershell
conda activate exam_env
python -m pip install -r requirements.txt
```

For a separate environment, run `conda env create -f environment.yml` from this directory, then select `predictive-maintenance` as the notebook environment.

| Order | Notebook | Purpose |
|---|---|---|
| 01 | [Exploratory analysis](notebooks/01_exploratory_analysis.ipynb) | Optional statistical exploration |
| 02 | [Data preprocessing](notebooks/02_data_preprocessing.ipynb) | Prepare corrected LSTM arrays and training-only scaler |
| 03 | [ML baselines](notebooks/03_ml_baselines.ipynb) | Historical linear/SVR/tree baselines; comparison limitations remain |
| 04 | [LSTM training](notebooks/04_lstm_training.ipynb) | Train, evaluate, save the model and generate SHAP plots |
| 05 | [Survival analysis](notebooks/05_survival_analysis.ipynb) | Fit and evaluate the fixed-landmark Cox baseline |
| 06 | [Anomaly detection](notebooks/06_anomaly_detection.ipynb) | Compare Isolation Forest, a window autoencoder and z-score baseline |

Run each notebook from the top. The first code cell locates the project root, so paths work when the kernel starts here or in `notebooks/`. Existing outputs were retained; older printed path messages may show the former layout.

Prepared data and current models are present locally. Moving files did not require retraining. If rebuilding from raw data, run notebook 02 before 04 or 05. Notebook 03 uses the CSV exported by 02, not the LSTM arrays.

Notebook 06 also requires the split metadata from 02. It fits detectors on the first 50 cycles of training engines, assumed healthy; uses validation engines for thresholds and early stopping; and evaluates on separate test engines with failure-derived pseudo-labels. The default policy uses the healthy-validation 98th percentile, five-cycle causal smoothing, and three consecutive exceedances. It is limited to FD001. Scores are not failure probabilities.

See the [anomaly evaluation report](reports/anomaly_fd001_report.md) for results and limitations. Test average precision was 0.765 for Isolation Forest, 0.615 for the autoencoder and 0.814 for the simple z-score baseline; the more complex model did not win this comparison.

## Current recorded results

| Model | Evaluation | Recorded result |
|---|---|---:|
| Corrected LSTM | Best validation RMSE over held-out-engine windows | 11.85 cycles |
| Corrected LSTM | Test RMSE, final window of each of 100 engines | 13.36 cycles |
| Corrected LSTM | Test R² | 0.8888 |
| Cox baseline | Validation C-index | 0.681 |
| Cox baseline | Test C-index for failure time since cycle 30 | 0.576 |

LSTM targets are capped at 125 cycles. Cox uses uncapped survival durations and first-30-cycle sensor means; later survival updates use age but not new sensor readings. Its test ranking is weak. These metrics measure different tasks and should not be compared numerically as if they were the same score.

Authoritative results: [LSTM metrics](models/lstm_fd001_v2/metrics.json) and [Cox metrics](models/cox_fd001/metrics.json). See [project status](docs/project_status.md) for remaining methodology and deployment work.

## Saved artifacts

- `models/lstm_fd001_v2/`: final model, best checkpoint, matching scaler, data-split metadata, training history and metrics. Both Keras files are retained: the final exported model and training checkpoint serve different purposes.
- `models/cox_fd001/`: Cox model/scaler bundle, configuration and metrics.
- `models/anomaly_fd001/`: Isolation Forest/scaler bundle, autoencoder/checkpoint, both threshold policies, configuration, reconstruction history and primary-policy test metrics.
- `reports/tables/anomaly_fd001/`: per-cycle scores and flags, threshold searches, test-window sensitivity, healthy false-positive rates, and unit-level alert lead times. `reports/figures/anomaly_fd001/` holds the corresponding figures.
- `reports/figures/lstm_fd001_v2/`: six figures from the latest recorded LSTM run, including SHAP.
- `reports/figures/survival_fd001/` and `reports/tables/survival_fd001/`: survival curves, failure probabilities, hazard ratios, assumptions and evaluation results.

The deployment code under `archive/deployment_drafts/` is unfinished, with unresolved conflicts and old filenames/shapes. It is not a runnable application.

Step 7 is implemented in `src/predictive_maintenance/api/`. It serves RUL, conditional survival, full-history anomaly scores and SHAP explanations from the existing artifacts. Start it from this project root:

```powershell
conda activate exam_env
python -m uvicorn predictive_maintenance.api.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --no-proxy-headers
```

Open <http://127.0.0.1:8000/docs> for working real-data examples. See the [API guide](docs/api_guide.md) for setup, input contracts, authentication, configuration, tests and Docker instructions, and the [validation report](reports/api_validation.md) for measured results. Docker execution remains unverified because Docker is not installed here. The [Step 7 specification](docs/step_07_fastapi_specification.md) records the design.

## Step 8: streaming and stored results

Step 8 adds durable reading ingestion, a worker calling the existing models, SQLite prediction history, in-application alert episodes and `/v1/stream` dashboard feeds. Restart an already-running Step 7 API to load these routes. With the API running, open a second project terminal:

```powershell
conda activate exam_env
python scripts/replay_stream.py --units 55 --max-cycle 40 --run-id demo-step8
```

This simulates engine observations and processes them into `data/runtime/stream.sqlite3`. Open <http://127.0.0.1:8000/v1/stream/runs/demo-step8/units/55/latest> to inspect the stored result, or use the Step 9 dashboard's **Live replay** page to watch the stream.

See the [Step 8 guide](docs/step_08_streaming.md) for resume/retry behavior, feed contracts, alert rules and the separate worker, and the [Step 8 validation report](reports/step08_validation.md) for real replay/restart evidence. The local design uses SQLite and HTTP; a broker, Redis and external notifications remain future integrations.

## Step 9: dashboard and live replay

```powershell
conda activate exam_env
python scripts/run_dashboard.py
```

Open <http://127.0.0.1:8050>. Selecting an engine/cycle automatically calculates its predictions, sampled trend and SHAP explanation. The dashboard provides sensor charts, an RUL gauge, anomaly history, conditional survival, alert review and PNG/PDF reports. Wait for the calculation status to finish; changing displayed sensors does not rerun models. The launcher starts or reuses the model API. No stream replay starts automatically.

For a fresh environment, install `requirements-dashboard.txt` after the serving requirements. See the [dashboard guide](docs/step_09_dashboard.md) for controls, login configuration and limitations. The Cox probability remains experimental; it is not combined with anomaly scores into a fabricated risk measure.

For live updates, open <http://127.0.0.1:8050/live/> and run this in a second terminal:

```powershell
conda activate exam_env
python scripts/replay_stream.py --units 1 55 --interval 0.5
```

Select the run ID printed by the simulator and an engine. The page refreshes saved readings, predictions and alerts every two seconds. Pause updates to inspect a snapshot; request SHAP explicitly. Completed runs stay available after restart. This reuses the existing simulator and requires no physical sensors or messaging broker.

## Checks and version control

```powershell
conda run -n exam_env python -m unittest discover -s tests -v
```

Large raw/processed data, binary models and the source-backup archive are ignored by Git but remain on disk. No Git merge or index changes were performed; the repository already had unresolved conflicts. See [the cleanup record](docs/reorganization.md).
