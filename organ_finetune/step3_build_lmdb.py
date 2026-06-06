"""Step 3 — build the LMDB datasets (with 3D conformers + k-fold splits).

Wraps the repo's proven builder `experiments/preprocess_data_LNPDB.py`, which:
  * reads the JSON from step 2,
  * generates 11 RDKit 3D conformers per unique molecule (-> mol.lmdb),
  * splits formulations into a held-out test set + k validation folds
    (-> processed_data_dirs/<name>/fold_V{0..k-1}/{train,valid,test}.lmdb).

Run from the repo root:  python organ_finetune/step3_build_lmdb.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

# Inputs/outputs are relative to the experiments/ dir (the builder's cwd).
JSON_PATH = "data_json/LNPDB_organ.json"
OUT_DIR = "processed_data_dirs/lnpdb_organ_gen"
KFOLD_VALID = 5
TEST_RATIO = 0.2
NTHREADS = 8


def main():
    cmd = (
        f"{sys.executable} preprocess_data_LNPDB.py "
        f"--inpath {JSON_PATH} "
        f"--outpath {OUT_DIR} "
        f"--kfold-valid {KFOLD_VALID} "
        f"--test-ratio {TEST_RATIO} "
        f"--nthreads {NTHREADS}"
    )
    print(f"[step3] cwd={EXPERIMENTS}\n[step3] {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=EXPERIMENTS)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\n[step3] LMDB folds written under experiments/{OUT_DIR}/fold_V0 ... fold_V{KFOLD_VALID-1}")


if __name__ == "__main__":
    main()
