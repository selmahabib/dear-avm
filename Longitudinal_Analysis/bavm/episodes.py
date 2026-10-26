"""Splits follow-up into counting-process episodes and fits time-varying Cox models."""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

MIN_TIME = 1 / 365


def split(df, covs, cuts, time_col="time", event_col="event"):
    """(start, stop] episodes per patient, split at `cuts` (years)."""
    edges = np.r_[0.0, np.sort(np.asarray(cuts, float)), np.inf]
    T = np.maximum(df[time_col].to_numpy(float), MIN_TIME)
    E = df[event_col].to_numpy(int)
    Xc = df[covs].to_numpy(float)
    rows = []
    for i in range(len(df)):
        for k in range(len(edges) - 1):
            a, b = edges[k], edges[k + 1]
            if T[i] <= a:
                break
            stop = min(T[i], b)
            rows.append((i, a, stop, int(E[i] == 1 and T[i] <= b), k, *Xc[i]))
    return pd.DataFrame(rows, columns=["_id", "start", "stop", "_ev",
                                       "interval", *covs])


def fit_ctv(ep, cols, penalizer=0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CoxTimeVaryingFitter(penalizer=penalizer).fit(
            ep[["_id", "start", "stop", "_ev"] + cols], id_col="_id",
            event_col="_ev", start_col="start", stop_col="stop")
