# v3 Data Pipeline Contract

The v3 pipeline uses a strict leakage boundary:

1. Load the raw CICIoT2023 table.
2. Add a stable `__row_id` for split/client disjointness checks.
3. Split raw rows into `train`, `val`, and `test` before preprocessing.
4. Fit `StandardScaler` only on the raw `train` split.
5. Apply the train-fitted scaler to `train`, `val`, and `test` without refit.
6. Save explicit per-client NPZ files:
   - `data/processed/<scenario>/<node>/train_preprocessed.npz`
   - `data/processed/<scenario>/<node>/val_preprocessed.npz`
   - `data/processed/<scenario>/<node>/test_preprocessed.npz`
7. Generate scenario-specific class weights from the scenario train manifest:
   - `artifacts/class_weights_<scenario>.pkl`

`src.data.dataloader.create_dataloaders_for_node` refuses to create a random
post-preprocessing validation split. Missing split files are treated as a data
pipeline error.

