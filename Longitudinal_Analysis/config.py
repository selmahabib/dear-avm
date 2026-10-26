"""Settings for the time-varying bAVM rupture analysis; set DATA_PATH before running."""
import numpy as np

DATA_PATH = r"path\to\your_data.xslx"
SEED = 42

# Cross-validation and bootstrap
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOT = 1000
N_BOOT_TDAUC = 200
N_BOOT_CAL = 200

# Predictors
AGE = "Age_dx"
DEEP = "Deep_Loc_excl"
PRIOR = "Ever_Ruptured"
BASE = [AGE, DEEP, PRIOR]
CANDIDATE_FEATURES = [               # pool for exhaustive_search.py
    "Age_dx", "Gender", "Race_nonwhite", "Hypertension", "Smoking",
    "Alcohol", "Substance", "max_Size", "Size_small", "Deep_Loc_excl",
    "Excl_Deep_Drain", "Single_Feeder", "Drainage_vein_varices",
    "Drainage_vein_stenosis", "Aneurysm_intranidal", "Ever_Ruptured",
]
R2ED_FEATS = ["Race_nonwhite", "Deep_Loc_excl", "Size_small",
              "Excl_Deep_Drain", "Single_Feeder"]
MIN_TIME = 1 / 365

# Time-varying model
FIXED_CUT = 1.0                     # None = choose by profile likelihood in each fold
CUT_GRID = np.round(np.arange(0.5, 5.01, 0.25), 2)
PIECEWISE_CUTS = (1, 5)
PENALIZERS = [0.0, 0.001, 0.01, 0.05, 0.1]
WIDE_PENALIZERS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                   0.1, 0.2, 0.5, 1.0]
TUNE_BY = "loglik"                  # "loglik" or "cindex"
ALPHA_GRID = [0.1, 0.5, 1.0, 5.0]   # R2eD comparator

# Integer score
MARS_CUTS = (20, 40, 60)
AGE_PTS = (0, 1, 2, 3)
AGE_STEP_YR = 20
SCORE_POINTS_FROM = "penalized"     # "penalized" or "unpenalized"
SCORE_RECALIBRATE = True
SCORE_VARIANTS = [("unpenalized", False), ("unpenalized", True),
                  ("penalized", False), ("penalized", True)]
SCORE_CALIB_GROUPS = "quartiles"    # "quartiles" or "tiers"

# Evaluation
CALIB_HORIZON = 2.0
CALIB_QUANTILES = 4
CALIB_HORIZONS = [1.0, 1.5, 2.0, 5.0, 10.0]
SUPP_HORIZONS = (2.0, 5.0, 10.0)
TD_AUC_TIMES = np.array([0.5, 1, 1.5, 2, 3, 5, 7, 10], dtype=float)
CAL_KNOTS = 3
TABLE4_HORIZONS = (1.0, 2.0, 5.0)
PROFILES = [(17, 0, 0), (36, 0, 0), (36, 0, 1), (36, 1, 0),
            (36, 1, 1), (55, 1, 0), (55, 1, 1), (73, 1, 1)]

# Figures
CAL_TICK = 0.02
FIG3_CAL_XMAX = 0.13
FIG3_CAL_YMAX = 0.18
FIG3_SHOW_BAND = True
FIG3_SHOW_HIST = True
KM_T_MAX = 10.0