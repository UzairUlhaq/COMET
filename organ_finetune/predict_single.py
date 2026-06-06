"""Predict the organ distribution for a SINGLE LNP formulation.

Give it one lipid-nanoparticle recipe (up to 4 components: IL / HL / PEG / CH,
each a SMILES + mol ratio) and it prints the model's predicted probability
distribution over the 9 organs.

It reuses the exact, tested data path: it builds a tiny one-sample LMDB
(`mol.lmdb` with RDKit 3D conformers + a `test.lmdb` with the single recipe)
in the same shape the full pipeline produces, then runs the same
`../unimol/infer_np.py` that step5 uses, and parses the `prob` it dumps.

Run from the repo root (conda env `comet_env`):

    # pick any point from the dataset by its key/index — carries the TRUE label,
    # so you see predicted vs. actual. (default with no args is sample '0')
    python organ_finetune/predict_single.py --sample 42

    # your own recipe on the command line
    python organ_finetune/predict_single.py \
        --il  "CCCC...OC(=O)..." --il-mol 50 \
        --hl  "CCCC...P(=O)..."  --hl-mol 10 \
        --peg "O=C(...)OCCOCC..." --peg-mol 1.5 \
        --chl "C[C@H](CCCC(C)C)..." --chl-mol 38.5

    # or from a JSON file: either the COMET sample shape
    #   {"components": [{"smi": ..., "component_type": "IL", "mol": 50}, ...]}
    # or a flat recipe
    #   {"IL_SMILES": ..., "IL_molratio": 50, "HL_SMILES": ..., ...}
    python organ_finetune/predict_single.py --input my_lnp.json

A component is simply omitted if its SMILES is blank (3-component LNPs are fine).
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import lmdb

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

# Reuse the proven builders so the LMDB bytes are byte-for-byte the same shape
# the full preprocessing produces (conformers + sample dict).
from preprocess_data_LNPDB import inner_smi2coords_onlymol, inner_lnp2data  # noqa: E402

# Frozen class order — MUST match step2_build_json.ORGAN_CLASSES.
ORGAN_CLASSES = [
    "lung_epithelium", "liver", "muscle", "spleen", "bone_marrow",
    "heart", "lung", "kidney", "ear",
]
DATASET_NAME = "lnpdb"

# output component_type -> (cli flag attr for smiles, cli flag attr for mol,
#                           flat-json smiles key, flat-json mol key)
COMPONENTS = [
    ("IL",  "il",  "il_mol",  "IL_SMILES",  "IL_molratio"),
    ("HL",  "hl",  "hl_mol",  "HL_SMILES",  "HL_molratio"),
    ("PEG", "peg", "peg_mol", "PEG_SMILES", "PEG_molratio"),
    ("CH",  "chl", "chl_mol", "CHL_SMILES", "CHL_molratio"),
]

# Inference knobs — kept in lockstep with step5_infer.py.
SCHEMA = "task_schemas/lnpdb_organ_schema.json"
HEAD_NAME = "organ"
NUM_CLASSES = 9
LOSS = "np_finetune_soft_cross_entropy"
DICT_NAME = "dict.txt"
CONF_SIZE = 11
ONLY_POLAR = 0
WEIGHT_PATH = "./save_lnpdb_organ/checkpoint_best.pt"
LNP_LAYERS, LNP_EMBED, LNP_FFN, LNP_HEADS = 8, 256, 256, 8


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict the 9-organ distribution for one LNP formulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", help="JSON file describing one LNP (COMET sample shape or flat recipe).")
    for _, smi_attr, mol_attr, _, _ in COMPONENTS:
        p.add_argument(f"--{smi_attr}", help=f"{smi_attr.upper()} SMILES")
        p.add_argument(f"--{mol_attr.replace('_', '-')}", dest=mol_attr, type=float,
                       help=f"{smi_attr.upper()} mol ratio")
    p.add_argument("--sample", help="pick a data point by its key/index from --dataset-json "
                                    "(carries the true label so you see predicted vs. actual).")
    p.add_argument("--dataset-json", default="experiments/data_json/LNPDB_organ.json",
                   help="(relative to repo root) dataset to pick --sample from.")
    p.add_argument("--weight-path", default=WEIGHT_PATH, help="fine-tuned checkpoint to load.")
    p.add_argument("--work-dir", default="processed_data_dirs/lnpdb_organ_single/fold_V0",
                   help="(relative to experiments/) where the one-sample LMDB is built.")
    p.add_argument("--results-path", default="./infer_results/lnpdb_organ_single",
                   help="(relative to experiments/) where infer_np.py writes its pkl.")
    p.add_argument("--output-json", default="prediction.json",
                   help="where to save the machine-readable prediction (survives cleanup).")
    p.add_argument("--keep-files", action="store_true",
                   help="keep the temporary LMDB/results instead of cleaning up.")
    return p.parse_args()


def _comps_from_sample(obj):
    """COMET sample shape ({"components": [...], "labels": {...}}) -> (components,
    true_label_or_None). The true label is the 'organ' distribution if present."""
    comps = []
    for c in obj["components"]:
        mol = float(c.get("mol", c.get("percent", 0.0)))
        comps.append({"smi": c["smi"], "component_type": c["component_type"], "mol": mol})
    true_label = (obj.get("labels") or {}).get(HEAD_NAME)
    return comps, true_label


def build_components(args):
    """Return (components, true_label) from --sample, --input, or the CLI flags,
    falling back to dataset sample 0 when nothing is provided. true_label is the
    organ distribution (list of 9 probs) when known, else None."""
    # 1) pick a point from the dataset by key/index (has the true label)
    if args.sample is not None:
        path = REPO_ROOT / args.dataset_json
        with open(path) as f:
            data = json.load(f)
        if args.sample not in data:
            sys.exit(f"[predict_single] sample '{args.sample}' not in {path} "
                     f"(keys are '0'..'{len(data) - 1}').")
        print(f"[predict_single] dataset sample '{args.sample}' from {args.dataset_json}")
        return _comps_from_sample(data[args.sample])

    # 2) explicit JSON file
    if args.input:
        with open(args.input) as f:
            obj = json.load(f)
        if "components" in obj:                       # COMET sample shape (may carry label)
            return _comps_from_sample(obj)
        # flat recipe shape
        comps = []
        for ctype, _, _, smi_key, mol_key in COMPONENTS:
            smi = obj.get(smi_key)
            if smi and str(smi).strip():
                comps.append({"smi": smi, "component_type": ctype,
                              "mol": float(obj.get(mol_key) or 0.0)})
        return comps, None

    # 3) CLI flags
    comps = []
    any_cli = False
    for ctype, smi_attr, mol_attr, _, _ in COMPONENTS:
        smi = getattr(args, smi_attr)
        if smi and smi.strip():
            any_cli = True
            comps.append({"smi": smi, "component_type": ctype,
                          "mol": float(getattr(args, mol_attr) or 0.0)})
    if any_cli:
        return comps, None

    # 4) default: dataset sample 0
    print("[predict_single] no input given — using dataset sample '0'. "
          "Use --sample N to pick another point.")
    path = REPO_ROOT / args.dataset_json
    with open(path) as f:
        data = json.load(f)
    return _comps_from_sample(data["0"])


def write_lmdb(path, entries):
    """Write pre-pickled byte entries to an LMDB file, keyed 0..N-1."""
    if os.path.exists(path):
        os.remove(path)
    env = lmdb.open(path, subdir=False, readonly=False, lock=False,
                    readahead=False, meminit=False, max_readers=1, map_size=int(10e9))
    txn = env.begin(write=True)
    for i, e in enumerate(entries):
        txn.put(f"{i}".encode("ascii"), e)
    txn.commit()
    env.close()


def build_single_lmdb(components, work_dir):
    """Build mol.lmdb + lnpdb/test.lmdb for the one recipe, mirroring the shape
    `preprocess_data_LNPDB.py` produces (so the task loader is happy)."""
    work_dir = Path(work_dir)
    subdir = work_dir / DATASET_NAME
    subdir.mkdir(parents=True, exist_ok=True)

    # unique molecules -> mol.lmdb (3D conformers); smi -> mol_id index
    unique_smis = []
    for c in components:
        if c["smi"] not in unique_smis:
            unique_smis.append(c["smi"])
    print(f"[predict_single] generating conformers for {len(unique_smis)} unique molecule(s)...")
    mol_entries = [inner_smi2coords_onlymol(smi) for smi in unique_smis]
    write_lmdb(str(work_dir / "mol.lmdb"), mol_entries)
    smi2mol_id = {smi: i for i, smi in enumerate(unique_smis)}

    # one sample -> test.lmdb. inner_lnp2data reads component['percent'] and a
    # 'labels' dict; the label is unknown at predict time, so use a neutral
    # uniform target purely to satisfy the loss/eval bookkeeping (it does not
    # affect the predicted softmax).
    sample = {
        "components": [
            {"smi": c["smi"], "component_type": c["component_type"],
             "mol": c["mol"], "percent": c["mol"]}
            for c in components
        ],
        "labels": {HEAD_NAME: [1.0 / NUM_CLASSES] * NUM_CLASSES},
        "dataset_name": DATASET_NAME,
        "lnp_id": "single_0",
    }
    write_lmdb(str(subdir / "test.lmdb"), [inner_lnp2data(smi2mol_id, sample)])
    return work_dir


def run_inference(args, task_name):
    cmd = (
        f"{sys.executable} ../unimol/infer_np.py --user-dir ../unimol ./ "
        f"--task-name {task_name} --valid-subset test "
        f"--num-workers 0 --ddp-backend=c10d --batch-size 8 --required-batch-size-multiple 1 "
        f"--task mol_np_finetune --loss {LOSS} --arch np_unimol "
        f"--classification-head-name {HEAD_NAME} --num-classes {NUM_CLASSES} "
        f"--dict-name {DICT_NAME} --conf-size {CONF_SIZE} --only-polar {ONLY_POLAR} "
        f"--path {args.weight_path} "
        f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
        f"--log-interval 50 --log-format simple "
        f"--results-path {args.results_path} "
        f"--lnp-encoder-layers {LNP_LAYERS} --lnp-encoder-embed-dim {LNP_EMBED} "
        f"--lnp-encoder-ffn-embed-dim {LNP_FFN} --lnp-encoder-attention-heads {LNP_HEADS} "
        f"--full-dataset-task-schema-path {SCHEMA} "
        f"--load-full-np-model --concat-datasets --output-cls-rep"
    )
    print(f"[predict_single] cwd={EXPERIMENTS}\n[predict_single] {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=EXPERIMENTS)
    if result.returncode != 0:
        sys.exit(result.returncode)


def report(args, components, true_label):
    fname = args.weight_path.split("/")[-2]      # e.g. save_lnpdb_organ
    pkl = EXPERIMENTS / args.results_path.lstrip("./") / f"{fname}_test.out.pkl"
    with open(pkl, "rb") as f:
        out = pickle.load(f)
    prob = out["prob"][0].tolist()               # [9]
    ranked = sorted(zip(ORGAN_CLASSES, prob), key=lambda kv: kv[1], reverse=True)

    # true label (when known) shown alongside each predicted prob.
    true_map = dict(zip(ORGAN_CLASSES, true_label)) if true_label else None
    true_top = max(true_map, key=true_map.get) if true_map else None

    print("\n=== Predicted organ distribution ===")
    print(f"  {'organ':<16} {'pred':>6}" + (f" {'true':>6}" if true_map else ""))
    for organ, p in ranked:
        bar = "#" * int(round(p * 40))
        suffix = f" {true_map[organ]:6.3f}" if true_map else ""
        print(f"  {organ:<16} {p:6.3f}{suffix}  {bar}")
    print(f"\nTop prediction: {ranked[0][0]}  ({ranked[0][1]:.1%})")
    if true_top is not None:
        match = "✓ match" if true_top == ranked[0][0] else "✗ mismatch"
        print(f"True top organ: {true_top}  ({true_map[true_top]:.1%})   [{match}]")

    # machine-readable prediction, written before cleanup so it persists.
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    payload = {
        "formulation": components,
        "distribution": {organ: p for organ, p in zip(ORGAN_CLASSES, prob)},
        "top_organ": ranked[0][0],
        "top_prob": ranked[0][1],
    }
    if true_map:
        payload["true_distribution"] = true_map
        payload["true_top_organ"] = true_top
        payload["top_match"] = (true_top == ranked[0][0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved prediction -> {out_path}")
    return pkl


def main():
    args = parse_args()
    components, true_label = build_components(args)
    if not components:
        sys.exit("[predict_single] no valid components — provide at least one SMILES.")
    print("[predict_single] formulation:")
    for c in components:
        print(f"    {c['component_type']:<4} mol={c['mol']:<6}  {c['smi'][:60]}{'...' if len(c['smi']) > 60 else ''}")

    work_dir = build_single_lmdb(components, EXPERIMENTS / args.work_dir)
    run_inference(args, args.work_dir)
    report(args, components, true_label)

    if not args.keep_files:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(EXPERIMENTS / args.results_path.lstrip("./"), ignore_errors=True)
    else:
        print(f"\n[predict_single] kept LMDB at {work_dir} and results at {EXPERIMENTS / args.results_path.lstrip('./')}")


if __name__ == "__main__":
    main()
