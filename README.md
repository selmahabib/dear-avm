# bAVM Rupture Prediction
 
Analysis code and dashboard for the two models of brain arteriovenous
malformation (bAVM) rupture.

The repository contains:
 
- **Cross-sectional analysis**: CatBoost classification of hemorrhagic
  presentation, with nested cross-validation and model comparison.
- **Longitudinal analysis**: ridge-penalized Cox model of time to rupture,
  and the integer score derived from it.
- **Dashboard**: a FastAPI inference service and a Next.js interface that
  run both models on a single patient.

The two models answer different questions.

| | Cross-sectional | Longitudinal |
|---|---|---|
| Question | Did this bAVM present with hemorrhage? | When might an unruptured bAVM bleed? |
| Outcome | `Ever_Ruptured` at presentation | Time to rupture during follow-up |
| Model | CatBoost gradient boosting | Ridge-penalized Cox proportional hazards |
| Predictors | 8 inputs | 3 inputs |
| Output | Probability + TreeSHAP attribution | Survival at 2, 5 and 10 years / integer score |

The classification model is not prognostic. It describes presentation at
diagnosis. Only the Cox model estimates future risk.

## Structure

The analyses are standalone. The dashboard is one application made of two
processes that run together:
 
```
Cross_Sectional_Analysis/   CatBoost fitting and evaluation (Python)
Longitudinal_Analysis/      Ridge Cox fitting (Python)
 
avm-dashboard/
    backend/                FastAPI inference service (Python)
    src/                    Next.js interface (Node)
```
 
Neither analysis is imported by the dashboard and neither runs at request
time. Each Python part has its own conda environment; the backend and
interface are started together and talk to each other over HTTP.

---

## 1. Cross-sectional analysis

Fits and evaluates the classification model for hemorrhagic presentation: 7 machine-learning models and the R2eD score.

**Protocol.** 10 outer × 5 inner nested cross-validation. Hyperparameters are selected in the inner loop; every outer fold contributes held-out predictions, so reported discrimination is never measured on data used for tuning.

**Set up the environment.** From the repository root:

```
conda create -n bavm-classification python=3.12
conda activate bavm-classification
cd Cross_Sectional_Analysis
python -m pip install -r requirements.txt
```

Creating the environment and installing are one-time steps. In every new terminal, run `conda activate bavm-classification` before running anything.

**Run it.** Pass the path to your registry with `--data`, or set it once in `DATA_PATHS` in `config.py`. Then:

```
python main.py --data path/to/registry.xlsx
```

This runs the nested CV, model comparison, calibration, SHAP and DeLong tests, then the main figure, rank stability and risk bands. Results go to `bavm_cross_sectional_analysis_<date>/` next to the data file.

Optional, separate steps:

```
python table1.py --data path/to/registry.xlsx --out <folder>           # univariate analysis
python shap_dependence.py <results folder> --data path/to/registry.xlsx # size SHAP dependence
python sensitivity.py --data path/to/registry.xlsx                      # undocumented feeder count
```

---

## 2. Longitudinal analysis

Fits the time-to-rupture model on three predictors: age, exclusively deep nidus location, and prior rupture at presentation.

**Model.** Ridge-penalized Cox model with a time-varying effect of prior rupture. The prior-rupture hazard ratio is allowed to differ in the first year after diagnosis and afterwards (cut-point 1 year, pre-specified). Age and deep location have a constant effect.

**Integer score.** An integer 0–7 score derived by rounding the fitted coefficients to whole points for bedside use: age 0–3 (one point per 20-year bracket: <20, 20–39, 40–59, ≥60), exclusive deep location 2, and prior rupture 2 in the first year and 1 afterwards. Patients are stratified into three tiers (Low 0–1, Medium 2–3, High 4–7), and a patient with prior rupture can move down a tier at 1 year. HR 2.62 per tier. The score is the model rounded; it is not a separate fit.

**Set up the environment.** Use a separate environment from the cross-sectional analysis; the two have their own dependencies and should not share one. From the repository root:

```
conda create -n bavm-surv python=3.12
conda activate bavm-surv
cd Longitudinal_Analysis
python -m pip install -r requirements.txt
```

**Run it.** Set `DATA_PATH` in `config.py` first, then:

```
python main.py                                  # full analysis
python graphical_abstract.py <results folder>   # graphical-abstract KM plot, after main.py
python exhaustive_search.py                     # feature selection, run separately
```

**Constants are hardcoded.** Nothing from this analysis is loaded at runtime. The fitted values live in `avm-dashboard/src/lib/ruptureRisk.ts`:
- the four coefficients (age, deep location, prior rupture year 1, prior rupture after year 1) from `tv_model_penalized_coefs.csv`. They are on the original scale, so no scaler mean or SD is needed.
- the cut-point (1 year).
- the baseline cumulative hazard H0 at 1, 2, 5 and 10 years: 0.00446, 0.01548, 0.02928, 0.06595.

Predicted risk by year t (for t > 1) is 1 − exp(−[exp(LP_early)·H0(1) + exp(LP_late)·(H0(t) − H0(1))]). LP_early uses the year-1 prior-rupture coefficient and LP_late the later one.

If you refit the model, those constants must be updated manually. Nothing in the build will warn you if they drift out of sync with the analysis.
---

## 3. Dashboard — backend service
 
The classification model is a tree ensemble and can't run in the browser,
nor can TreeSHAP be computed there. The interface calls this service instead.
 
The trained model and scaler are committed in `avm-dashboard/backend/` so
there is no export step. You can clone and run directly.
 
```bash
conda create -n bavm-api python=3.12
conda activate bavm-api
cd avm-dashboard/backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
 
**Verify:**
 
```bash
curl http://localhost:8000/health
```
 
This should return `{"status":"ok"}` with the nine model features.
The endpoint only responds after the model, scaler, and TreeSHAP explainer
have all loaded so a successful call confirms all three.
 
**Endpoints.** `GET /health`, `POST /predict`. Interactive docs at
http://localhost:8000/docs.
 
**Missing values.** The fields can accept `null`. CatBoost routes missing
values through the default direction learned during training, so they are
passed through as NaN rather than imputed. Feeder territory count is the
exception. Undocumented status is its own feature, so a null there sets
`Feeder_Count_Unknown` to 1 and `Single_Feeding_Artery` to 0, matching how
the cohort was encoded.
 
---
 
## 4. Dashboard — interface
 
Next.js app. Requires Node 18.18+.
 
```bash
cd avm-dashboard
npm ci
npm run dev
```
 
Open http://localhost:3000. Run this alongside the backend. The interface
is a separate process and the API doesn't serve it.
 
Use `npm ci` rather than `npm install`. Several dependencies are specified as
`"latest"`, so `npm install` re-resolves them on every run; the lockfile is
what keeps builds reproducible.
 
**Port 3000 is required.** CORS in `app.py` allows that origin only. If 3000
is occupied, Next will start on 3001 and every prediction fails with a CORS
error that looks like a broken backend. Free the port, or widen
`allow_origins`.
 
**Configuration.** The API base URL defaults to `http://localhost:8000`. No
`.env.local` is needed for local development.
 
The interface switches between the two models with the mode toggle. In
classification mode it shows the predicted probability, the R²eD score on the
same inputs and a SHAP waterfall. In longitudinal mode it shows 2-,
5-, and 10-year risk with the Cox contribution waterfall and the integer score view.
 
