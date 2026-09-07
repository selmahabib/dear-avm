"""
Data loading and feature engineering for the longitudinal cohort
"""
import numpy as np
import pandas as pd

DEEP = ["Location_thalamus", "Location_basal_ganglia", "Location_brainstem",
        "Location_Cerebellar_hemisphere", "Location_Cerebellar_vermis"]
LOBAR = ["Location_frontal", "Location_temporal",
         "Location_parietal", "Location_occipital"]

NUM = ["Age_dx", "max_Size", "Fup_Duration_Year", "NumFupHemorrhages",
       "Ever_Ruptured", "Hypertension", "Smoking", "Alcohol", "Substance",
       "Feeder_Territory_Number", "Drainage_vein_deep",
       "Drainage_vein_superficial", "Drainage_vein_varices",
       "Drainage_vein_stenosis", "Aneurysm_intranidal", "Aneurysm_prenidal",
       "Aneurysm_postnidal", "Aneurysm_unrelated",
       "Location_frontal", "Location_temporal", "Location_parietal",
       "Location_occipital", "Location_thalamus", "Location_basal_ganglia",
       "Location_brainstem", "Location_Cerebellar_hemisphere",
       "Location_Cerebellar_vermis", "Follow_hemorrhge"]

FINAL_FEATURES = ["Age_dx", "Deep_Loc_excl", "Ever_Ruptured"]

CANDIDATE_FEATURES = [
    "Age_dx", "Gender", "Race_nonwhite", "Hypertension", "Smoking",
    "Alcohol", "Substance", "max_Size", "Size_small", "Deep_Loc_excl",
    "Excl_Deep_Drain", "Single_Feeder", "Drainage_vein_varices",
    "Drainage_vein_stenosis", "Aneurysm_intranidal", "Ever_Ruptured",
]

R2ED_FEATS = ["Race_nonwhite", "Deep_Loc_excl", "Size_small",
              "Excl_Deep_Drain", "Single_Feeder"]


def load_and_engineer(data_path):
    """Read the registry and build the longitudinal analysis frame.

    Events are hemorrhages during follow-up. Patients who underwent
    microsurgery, radiosurgery or embolization are censored at treatment, so
    the at-risk interval covers the untreated natural history only. Patients
    without a recorded follow-up duration drop out here.
    """
    df_raw = pd.read_excel(data_path)
    df_raw.columns = df_raw.columns.str.strip()

    er_cols = [c for c in df_raw.columns if "Ever_Ruptured" in c]
    df_raw["Ever_Ruptured"] = pd.to_numeric(df_raw[er_cols[0]], errors="coerce")

    for c in NUM:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    df_raw.replace(99, np.nan, inplace=True)

    if "Gender" in df_raw.columns:
        gs = df_raw["Gender"].astype(str).str.strip().str.lower()
        df_raw["Gender"] = gs.map({"2_female": 1, "female": 1, "2": 1, "2.0": 1,
                                   "1_male": 0, "male": 0, "1": 0, "1.0": 0})

    if "Race" in df_raw.columns:
        rs = df_raw["Race"].astype(str).str.strip().str.lower()
        df_raw["Race_nonwhite"] = rs.map({
            "1_white": "W", "1": "W", "1.0": "W", "white": "W",
            "2_black": "NW", "2": "NW", "2.0": "NW", "black": "NW",
            "3_asian": "NW", "3": "NW", "3.0": "NW", "asian": "NW",
            "4_hispanic": "NW", "4": "NW", "4.0": "NW",
            "5_other": "NW", "5": "NW", "5.0": "NW",
        }).map({"W": 0, "NW": 1})

    # Deep location, exclusive definition: nonlobar involvement with no lobar
    # component. Matches the original R2eD score, and differs from the broader
    # definition used in the cross-sectional analysis.
    for c in DEEP + LOBAR:
        if c not in df_raw.columns:
            df_raw[c] = np.nan
    df_raw["Deep_Loc_excl"] = (
        (df_raw[DEEP].max(axis=1) == 1) &
        (df_raw[LOBAR].max(axis=1).fillna(0) == 0)).astype(float)
    df_raw.loc[df_raw[DEEP].isna().all(axis=1), "Deep_Loc_excl"] = np.nan

    for c in ["Drainage_vein_deep", "Drainage_vein_superficial"]:
        if c not in df_raw.columns:
            df_raw[c] = np.nan
    df_raw["Excl_Deep_Drain"] = (
        (df_raw["Drainage_vein_deep"] == 1) &
        (df_raw["Drainage_vein_superficial"].fillna(1) == 0)).astype(float)
    df_raw.loc[df_raw["Drainage_vein_deep"].isna(), "Excl_Deep_Drain"] = np.nan

    if "Feeder_Territory_Number" in df_raw.columns:
        ftn = pd.to_numeric(df_raw["Feeder_Territory_Number"], errors="coerce")
        df_raw["Single_Feeder"] = (ftn == 1).astype(float)
        df_raw.loc[ftn.isna(), "Single_Feeder"] = np.nan

    if "max_Size" in df_raw.columns:
        sz = pd.to_numeric(df_raw["max_Size"], errors="coerce")
        df_raw["Size_small"] = (sz < 3).astype(float)
        df_raw.loc[sz.isna(), "Size_small"] = np.nan

    for c in ["Aneurysm_intranidal", "Aneurysm_prenidal",
              "Aneurysm_postnidal", "Aneurysm_unrelated"]:
        if c not in df_raw.columns:
            df_raw[c] = np.nan

    for c in ["FirstFupHemoDate", "FirstSurgeryDate", "FirstSRSDate",
              "FirstEmboDate"]:
        if c in df_raw.columns:
            df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")
    if all(c in df_raw.columns
           for c in ["FirstSurgeryDate", "FirstSRSDate", "FirstEmboDate"]):
        df_raw["FirstTxDate"] = df_raw[
            ["FirstSurgeryDate", "FirstSRSDate", "FirstEmboDate"]].min(axis=1)

    def make_event(row):
        # A bleed only counts if it happened before treatment; anything after
        # treatment is censored, since the lesion is no longer untreated.
        treated = str(row.get("TreatedAVMs", "")).strip() == "1_Treated"
        bled = row.get("Follow_hemorrhge", 0) == 1
        if not treated:
            return int(bled)
        if not bled:
            return 0
        hd = row.get("FirstFupHemoDate")
        td = row.get("FirstTxDate")
        if pd.notna(hd) and pd.notna(td):
            return 1 if hd <= td else 0
        return 0

    df_raw["event"] = df_raw.apply(make_event, axis=1)
    if df_raw["event"].sum() == 0 and "NumFupHemorrhages" in df_raw.columns:
        df_raw["event"] = (df_raw["NumFupHemorrhages"].fillna(0) > 0).astype(int)

    df_raw["time"] = df_raw["Fup_Duration_Year"]
    df = df_raw[(df_raw["time"].notna()) & (df_raw["time"] > 0)]
    df = df.copy().reset_index(drop=True)

    df["untreated"] = (
        df.get("TreatedAVMs", pd.Series(index=df.index, dtype=object))
        .astype(str).str.strip() != "1_Treated")

    n_fh = int((df_raw["Follow_hemorrhge"] == 1).sum())
    n_ev = int(df["event"].sum())
    print(f"Longitudinal cohort {len(df)} | events {n_ev} "
          f"({100 * n_ev / len(df):.1f}%) | "
          f"post-treatment bleeds censored {n_fh - n_ev}")
    print(f"Untreated subset {int(df['untreated'].sum())} "
          f"| events {int(df.loc[df['untreated'], 'event'].sum())}")
    return df


def cohort_summary(df, out_csv="cohort_summary.csv"):
    """Follow-up and treatment breakdown for the Methods and Results text."""
    rows = []

    def _block(label, d):
        t = d["time"].astype(float)
        return {
            "Group": label,
            "N": len(d),
            "Events": int(d["event"].sum()),
            "Event_pct": round(100 * d["event"].mean(), 1),
            "FU_mean_yr": round(float(t.mean()), 2),
            "FU_sd_yr": round(float(t.std(ddof=1)), 2),
            "FU_median_yr": round(float(t.median()), 2),
            "FU_q1_yr": round(float(t.quantile(0.25)), 2),
            "FU_q3_yr": round(float(t.quantile(0.75)), 2),
            "Person_years": round(float(t.sum()), 1),
        }

    rows.append(_block("Full longitudinal cohort", df))
    rows.append(_block("Treated", df[~df["untreated"]]))
    rows.append(_block("Untreated", df[df["untreated"]]))
    if "Ever_Ruptured" in df.columns:
        rows.append(_block("Ruptured at presentation",
                           df[df["Ever_Ruptured"] == 1]))
        rows.append(_block("Unruptured at presentation",
                           df[df["Ever_Ruptured"] == 0]))

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print(f"Saved {out_csv}")
    return out


def surv_array(df):
    """Structured array in the (event, time) form scikit-survival expects."""
    return np.array(list(zip(df["event"].astype(bool), df["time"].astype(float))),
                    dtype=[("event", "?"), ("time", "<f8")])
