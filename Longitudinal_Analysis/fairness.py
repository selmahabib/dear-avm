"""
Discrimination parity across subgroups
"""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from survival_helpers import bootstrap_cindex, cindex
N_BOOT = 1000
SEED = 42
MIN_EVENTS = 5

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    folder = Path(sys.argv[1])
    src = folder / "cv_predictions.csv"
    if not src.exists():
        print(f"Not found: {src}")
        return 1
    d = pd.read_csv(src)
    groups = {}
    if "Race_nonwhite" in d.columns:
        groups["White"] = d["Race_nonwhite"] == 0
        groups["Non-white"] = d["Race_nonwhite"] == 1
    if "Gender" in d.columns:
        groups["Male"] = d["Gender"] == 0
        groups["Female"] = d["Gender"] == 1
    if "Ever_Ruptured" in d.columns:
        groups["Prior rupture"] = d["Ever_Ruptured"] == 1
        groups["No prior rupture"] = d["Ever_Ruptured"] == 0
    rows = []
    for name, m in groups.items():
        n_ev = int(d.loc[m, "event"].sum())
        if m.sum() < 20 or n_ev < MIN_EVENTS:
            rows.append({"Subgroup": name, "N": int(m.sum()), "Events": n_ev,
                         "C_index": "-", "CI_low": "-", "CI_high": "-",
                         "Note": "too few events"})
            continue
        c, lo, hi = bootstrap_cindex(d.loc[m, "event"], d.loc[m, "time"],
                                     d.loc[m, "cv_risk"], n_boot=N_BOOT,
                                     seed=SEED)
        rows.append({"Subgroup": name, "N": int(m.sum()), "Events": n_ev,
                     "C_index": round(cindex(d.loc[m, "event"],
                                             d.loc[m, "time"],
                                             d.loc[m, "cv_risk"]), 4),
                     "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                     "Note": ""})
    out = pd.DataFrame(rows)
    out.to_csv(folder / "fairness_subgroup_cindex.csv", index=False)
    print(out.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
