# organ_finetune — predict organ from an LNP's 4 lipids + composition

Fine-tune COMET to take a lipid-nanoparticle formulation (4 components + their
mol ratios) and output a **probability distribution over 9 organs**
(single 9-way softmax head). Everything you run lives in this folder; it reuses
the framework (`../unimol`), the pretrained encoder (`../ckp`), and the existing
data dirs under `../experiments`.

> Read `../experiments/COMET_TRAINING_GUIDE.md` first for the big picture
> (two encoders, pretraining, the train loop). This README is the runbook.

## The 9 organs (class order, frozen)
```
0 lung_epithelium  1 liver  2 muscle  3 spleen  4 bone_marrow
5 heart  6 lung  7 kidney  8 ear
```

## Prerequisites
- conda env `comet_env` (see repo root README for install).
- Pretrained encoder present: `../ckp/mol_pre_no_h_220816.pt`.
- Your raw data at `../experiments/data_raw/LNPDB.csv` (same columns as LNPDB:
  `IL_SMILES, IL_molratio, HL_SMILES, HL_molratio, PEG_SMILES, PEG_molratio,
  CHL_SMILES, CHL_molratio, Model_target`). To use your own CSV, drop it there or
  edit the input path in `step1_clean_csv.py`.

## Run order (all commands from the **repo root**)
```bash
conda activate comet_env

# 1. clean: drop NaN-target rows, keep 4 lipids + organ  -> experiments/data_processed/LNPDB_clean.csv
python organ_finetune/step1_clean_csv.py

# 2. build JSON + schema: group by composition, softmax-distribution labels
#    -> experiments/data_json/LNPDB_organ.json  +  experiments/task_schemas/lnpdb_organ_schema.json
python organ_finetune/step2_build_json.py

# 3. build LMDB (3D conformers + 5 folds + 20% test)
#    -> experiments/processed_data_dirs/lnpdb_organ_gen/fold_V0 ...
python organ_finetune/step3_build_lmdb.py

# 4. SMOKE TEST (1 epoch) — validate the wiring before committing GPU hours
python organ_finetune/step4_train.py --smoke

# 4b. full fine-tune (loads pretrained encoder, trains LNP encoder + head)
python organ_finetune/step4_train.py

# 5. inference on the held-out test split (softmax probs + CLS embeddings)
python organ_finetune/step5_infer.py
```

## Files
| File | What it does |
|------|--------------|
| `step1_clean_csv.py` | Raw CSV → slim clean CSV (4 lipids + `Model_target`). |
| `step2_build_json.py` | Clean CSV → `LNPDB_organ.json` (one sample per formulation, `labels = {"organ": [9 probs]}`) + the task schema. |
| `step3_build_lmdb.py` | Wraps `experiments/preprocess_data_LNPDB.py` to build the conformer/fold LMDBs. |
| `step4_train.py` | Fine-tune launcher (`--smoke` for a 1-epoch test). |
| `step5_infer.py` | Inference launcher. |
| `../unimol/losses/np_finetune_soft_cross_entropy.py` | The soft-target loss (added to the framework; auto-registered). |

## How the label works (why softmax, not 9 sigmoids)
The same formulation is measured across several organs (biodistribution panels),
and the only reliable per-formulation key is the composition itself. step2 groups
by composition and turns the measured organs into a **distribution**:
single-organ study → one-hot; k-organ panel → uniform 1/k. The model predicts a
distribution that sums to 1; the loss is `-(target · log_softmax(logits)).mean()`.

## Known data caveat (read before trusting heart/kidney)
Heart / kidney / lung appear **only** in 5-organ panels, so their target mass is
small and diluted (≈0.2 each, ~12 total units vs. 1800 for lung_epithelium). The
model will predict them weakly. The lever to sharpen them is **value-weighting**:
re-introduce `Experiment_value` so a panel concentrates mass on the organ it
actually prefers (left as a follow-up; step1 currently drops the value column).

## Troubleshooting / fallback
- **`KeyError: head 'organ'` or a logits/target shape mismatch in the loss.**
  The loss handles both the bare-tensor and the per-task-dict label paths, but if
  the schema path names the head differently, set `--classification-head-name` in
  `step4_train.py` to the head the model registered (printed in the smoke-test
  log), or it will fall back to the single-entry dict automatically.
- **Component-type vocab error at setup.** Ensure `step2` wrote
  `task_schemas/lnpdb_organ_schema.json` (it carries the IL/HL/CH/PEG dictionary).
- **Fallback to hard single-label classification** (no soft targets): switch the
  label in `step2_build_json.py` to an integer organ index and train with the
  repo's existing `--loss np_finetune_cross_entropy`. You lose the panel
  distribution but reuse a fully-proven loss.
- **Out of memory.** Lower `BATCH_SIZE` in `step4_train.py`; keep
  `--freeze-molecule-encoder` on.

## Note on "a fully standalone repo"
This folder is **not** a self-contained copy of the framework — it references
`../unimol` and `../ckp` (the 182 MB pretrained encoder). Copying the whole
framework + checkpoint into a separate repo is possible but heavy and rarely
what you want; keeping it inside COMET reuses the tested code paths.
