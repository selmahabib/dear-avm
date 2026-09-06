"""
Runs the scripts in order
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DOWNSTREAM = [
    ('figure_main.py', ['CatBoost'], True),
    ('rank_stability.py', [], False),
    ('risk_bands_all.py', [], False),
]

REQUIRED_FROM_MAIN = [
    'nested_oof_predictions.csv',
    'nested_cv_per_fold.csv',
    'summary_table.csv',
]

def run(script, args, label):
    print(f"\n{label}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / script), *args])
    mins = (time.time() - t0) / 60
    ok = r.returncode == 0
    print(f"  {'ok' if ok else 'FAILED'} ({mins:.1f} min)", flush=True)
    return ok

def new_results_dir(root: Path):
    marker = root / '.last_run'
    if not marker.exists():
        return None
    p = Path(marker.read_text().strip())
    return p if p.is_dir() else None
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default=None,
                    help='Skip the main run and use an existing results folder.')
    args = ap.parse_args()

    missing = [s for s, _, _ in DOWNSTREAM if not (HERE / s).exists()]
    if not (HERE / 'Nested_CV.py').exists():
        missing.append('Nested_CV.py')
    if missing:
        print(f"Not found in {HERE}: {', '.join(missing)}")
        return 1

    folder = None

    if args.folder:
        folder = Path(args.folder)
    else:
        ok = run('Nested_CV.py', [], 'main')
        if not ok:
            print("\nMain run failed.")
            return 1
        folder = new_results_dir(HERE)

    if folder is None or not folder.exists():
        print("\nCould not find the results folder. Re-run with:")
        print("  python run_all.py --folder "
          "<path to bavm_cross_sectional_analysis_...>")
        return 1

    absent = [f for f in REQUIRED_FROM_MAIN if not (folder / f).exists()]
    if absent:
        print(f"\n{folder} is missing: {', '.join(absent)}")
        print("The main run didn't finish cleanly. Check output.")
        return 1

    results = {}
    for script, extra, _ in DOWNSTREAM:
        results[script] = run(script, [str(folder), *extra], script)

    n_files = len(list(folder.glob('*.png'))) + len(list(folder.glob('*.csv')))
    print(f"\n{n_files} files in {folder.resolve()}")
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())