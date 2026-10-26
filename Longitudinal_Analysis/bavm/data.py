"""Loads the registry and builds the longitudinal cohort, censored at first treatment."""
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
       "Aneurysm_postnidal", "Aneurysm_unrelated", *DEEP, *LOBAR,
       "Follow_hemorrhge"]


def load_and_engineer(data_path):
    """Read the registry and return one row per patient with time and event."""
    df_raw = pd.read_excel(data_path)
    df_raw.columns = df_raw.columns.str.strip()

    er_cols = [c for c in df_raw.columns if "Ever_Ruptured" in c]
    df_raw["Ever_Ruptured"] = pd.to_numeric(df_raw[er_cols[0]], errors="coerce")

    for c in NUM:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    df_raw.replace(99, np.nan, inplace=True)

    # Demographics
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

    # Exclusive deep location (no lobar component)
    for c in DEEP + LOBAR:
        if c not in df_raw.columns:
            df_raw[c] = np.nan
    df_raw["Deep_Loc_excl"] = (
        (df_raw[DEEP].max(axis=1) == 1) &
        (df_raw[LOBAR].max(axis=1).fillna(0) == 0)).astype(float)
    df_raw.loc[df_raw[DEEP].isna().all(axis=1), "Deep_Loc_excl"] = np.nan

    # R2eD components
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

    # Treatment date
    for c in ["FirstFupHemoDate", "FirstSurgeryDate", "FirstSRSDate",
              "FirstEmboDate"]:
        if c in df_raw.columns:
            df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")
    if all(c in df_raw.columns
           for c in ["FirstSurgeryDate", "FirstSRSDate", "FirstEmboDate"]):
        df_raw["FirstTxDate"] = df_raw[
            ["FirstSurgeryDate", "FirstSRSDate", "FirstEmboDate"]].min(axis=1)

    # Event: hemorrhage before treatment
    def make_event(row):
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


def surv_array(df):
    """Structured (event, time) array for scikit-survival."""
    return np.array(list(zip(df["event"].astype(bool), df["time"].astype(float))),
                    dtype=[("event", "?"), ("time", "<f8")])