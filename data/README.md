# Data

- `raw/cmapss/`: original `train_FD00*.txt`, `test_FD00*.txt` and `RUL_FD00*.txt`. Do not overwrite source data. All four supplied subsets are retained, although active models use FD001.
- `processed/lstm_fd001/`: corrected training/validation sequences, targets, engine IDs, window endpoints, training-only scaler and split metadata. Rebuild with notebook 02.
- `processed/ml_baselines/df_clean_FD001.csv`: historical tabular data still used by notebook 03. It is not the corrected LSTM training input; read the methodology limitations in `docs/project_status.md`.
- `processed/survival_fd001/`: raw-scale covariate/duration/event tables for training, validation and censored test engines. Rebuild with notebook 05.

Dataset documentation is under `references/`. Large data files are ignored by Git; existing local copies remain available.
