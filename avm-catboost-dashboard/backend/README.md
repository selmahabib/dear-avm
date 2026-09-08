# CatBoost inference API

The classification model is a tree ensemble. It cannot run in the browser,
and TreeSHAP cannot be computed there, so the interface calls this service
instead.

## Exporting the model

`Nested_CV.py` writes two files into its results folder:

    catboost_model.cbm      the full-cohort refit of the configuration
                            selected in every outer fold
    catboost_scaler.csv     the StandardScaler constants

Copy both next to `app.py`. The scaler is not optional - the model was
fitted on scaled inputs and will return meaningless probabilities without it.

## Running

    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Check it loaded:

    curl http://localhost:8000/health

## Undocumented findings

Tri-state fields accept `null`. CatBoost routes missing values through the
default direction it learned during training, so they are passed through as
NaN rather than imputed.

Feeder territory count is the exception: undocumented status is its own
feature in the model, so a null there sets `Feeder_Count_Unknown` to 1 and
`Single_Feeding_Artery` to 0, matching how the cohort was encoded.

## Deployment

CORS is open to localhost:3000 only. Widen `allow_origins` in `app.py`
before hosting this anywhere else.
