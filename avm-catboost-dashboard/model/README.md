The trained model lives in `backend/`, not here.

`Nested_CV.py` writes `catboost_model.cbm` and `catboost_scaler.csv` into its
results folder. Copy both into `backend/` and start the service; the
interface calls it at `/predict`.
