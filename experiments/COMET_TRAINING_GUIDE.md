# COMET training — a guide

A from-scratch explanation of how training works in this repo: the big picture,
how the model was pretrained, how fine-tuning is done, what the inputs/outputs
are, and how it all maps to the **organ-prediction** task we've been building.

---

## 0. The mental model in one picture

```
                         ┌─────────────────────────────────────────────┐
   A lipid nanoparticle  │  COMET model (arch = np_unimol)               │
   = 4 components +       │                                               │
     their mol ratios     │   each component ─► UniMol 3D encoder ─► vec  │  ◄── PRETRAINED
                          │                         (frozen or tuned)     │      (loaded from ckp/)
                          │                                               │
                          │   [vec + %ratio + type] per component         │
                          │            │                                  │
                          │            ▼                                  │
                          │   LNP transformer encoder ─► [CLS] vector     │  ◄── TRAINED FRESH
                          │            │                                  │      on your data
                          │            ▼                                  │
                          │   classification/regression head ─► output    │
                          └─────────────────────────────────────────────┘
```

Two encoders stacked:
1. **UniMol** turns each *molecule* (atoms + 3D coordinates) into a vector. It is
   **pretrained elsewhere** and loaded from a checkpoint.
2. The **LNP encoder** turns the *set of components + composition* into one vector
   and predicts the property. It is **trained fresh** on the LNP dataset.

"Fine-tuning" = load the pretrained UniMol, then train (the LNP encoder, and
optionally UniMol too) on your labeled LNP data.

---

## 1. Pretraining (NOT in this repo — it's external UniMol)

There is **no pretraining code here**. The pretrained molecular encoder ships as a
checkpoint:

```
ckp/mol_pre_no_h_220816.pt        # ~182 MB, the UniMol encoder ("no hydrogen" variant)
```

It was pretrained by the UniMol authors (DP Technology) on a very large corpus of
molecular 3D conformers, with self-supervised objectives — masked atom-type
prediction and 3D coordinate/distance denoising (recovering corrupted positions).
The result is an encoder that already "understands" molecular geometry/chemistry,
so we don't need millions of LNP labels to get useful representations.

> Takeaway: you never run pretraining. You **download/use the checkpoint** and
> fine-tune on top of it. The repo README links the `ckp/` download.

---

## 2. The data pipeline: CSV → JSON → LMDB

Training does **not** read CSV directly. Raw data is converted in stages:

```
LNPDB.csv ─►(stage 1)─► clean CSV ─►(stage 2)─► COMET JSON ─►(stage 3)─► LMDB ─► training
```

- **Stage 1** — `dataset_preprocessing.py` → `data_processed/LNPDB_clean.csv`
  Drops NaN-target rows, keeps the 4 lipids + target organ.
- **Stage 2** — `experiments/dataset_preprocessing_organ.py` → `data_json/LNPDB_organ.json` + a task schema
  Groups rows by composition and builds the label per formulation. Each JSON sample:
  ```json
  {
    "components": [ {"smi": "...", "component_type": "IL", "mol": 50.0}, ... ],
    "labels":     { "organ": [0.0, 1.0, 0.0, ...] },   // task -> target
    "dataset_name": "lnpdb"
  }
  ```
- **Stage 3** — the LMDB builder (`experiments/preprocess_data_LNPDB.py`, or the
  pipeline's `preprocess-lmdb`) turns JSON into:
  - `mol.lmdb` — per-molecule: atoms, **11 precomputed 3D conformers** (`conf_size`),
    pairwise distances, edge types. (RDKit generates the conformers here.)
  - `train.lmdb` / `valid.lmdb` / `test.lmdb` — per-formulation: which molecules
    (`mol_id`), their `percent`, `component_type`, and the `target` labels.

The **task schema JSON** (`task_schemas/*.json`) declares which label names are
tasks (→ how many heads), and the component-type vocabulary.

---

## 3. What the model consumes and produces (one training step)

Defined in `unimol/tasks/unimol_np_finetune.py` (data loading) and
`unimol/models/unimol.py` (`NPUniMolModel.forward`).

**Inputs** (`net_input`) per batch:
- `src_tokens`, `src_coord`, `src_distance`, `src_edge_type` — the atoms/geometry
  for each component molecule (fed to UniMol).
- `mol_ids` — which molecules make up each LNP.
- `percents` — composition ratios.
- `component_types` — IL / HL / CH / PEG tokens.

**Target** (`sample["target"]`): the label(s) for each formulation, plus a
per-task **mask** so a formulation only contributes loss on the organs/tasks it
actually has labels for (this is how missing labels are handled cleanly).

**Output**: per task, the head maps the LNP `[CLS]` vector to logits
(`num_classes` wide). For inference, `--output-cls-rep` also dumps the `[CLS]`
embedding.

---

## 4. Fine-tuning: the flags that matter

Entry point: `unimol/train_np.py`. A launcher (e.g.
`experiments/training_script_LNPDB_heartkidney.py`) calls it via subprocess. The
flags that define a fine-tune:

| Flag | Meaning |
|------|---------|
| `--finetune-from-model ckp/mol_pre_no_h_220816.pt` | **Load the pretrained UniMol weights.** On the first run (no `checkpoint_last.pt` yet) the loader takes these weights and resets optimizer/scheduler. This is what makes it fine-tuning vs. from-scratch. |
| `--arch np_unimol` | The two-encoder model. |
| `--task mol_np_finetune` | LNP task: loads the LMDBs + schema, builds heads. |
| `--loss np_finetune_contrastive` | Training objective (see §6). |
| `--num-classes N` | Output width of each head. `1` = regression; `K` = K-way classification. |
| `--classification-head-name <name>` | Which head. |
| `--full-dataset-task-schema-path <schema.json>` | The schema → tasks/heads. |
| `--lnp-encoder-layers/embed-dim/ffn-embed-dim/attention-heads` | The fresh LNP encoder size (defaults 8 / 256 / 256 / 8). |
| `--epoch-to-freeze-molecule-encoder` | Freeze UniMol after epoch N. The pipeline's `freeze_molecule_encoder: true` keeps UniMol frozen the whole time — recommended for small datasets (fewer params to overfit, less GPU). |
| `--conf-size 11`, `--only-polar 0` | 11 conformers/molecule; no explicit hydrogens. |
| `--fp16` | Mixed precision. |

**Optimizer / schedule** (typical): Adam (betas `0.9, 0.99`, eps `1e-6`),
`clip-norm 1.0`, LR `1e-4`, polynomial decay with `warmup-ratio 0.06`.

---

## 5. The training loop (epochs, validation, best checkpoint)

`train_np.py` runs the standard loop:
1. Build task → model → loss; load pretrained weights.
2. For each epoch up to `--max-epoch`: iterate batches, forward → loss → backward → step.
3. Every `--validate-interval` (default 1 epoch) run validation, compute the metric.
4. **Best checkpoint** = the epoch with the best `--best-checkpoint-metric`
   (e.g. `valid_spearmanr_coeff`, with `--maximize-best-checkpoint-metric`). Saved
   as `checkpoint_best.pt`.
5. **Early stopping**: if the metric doesn't improve for `--patience` validations, stop.

Cross-validation: the pipeline splits into K folds (`kfold_valid: 5`) + a held-out
test set (`test_ratio: 0.2`); you train one fold at a time (`--fold`).

---

## 6. The loss (and why it matters for our organ task)

The default `np_finetune_contrastive` is a **pairwise ranking** loss: for two
formulations it pushes the predicted-score *difference* to match the sign of the
true *difference*. Great for "rank LNPs by potency", i.e. regression-style targets.

There's also `np_finetune_cross_entropy` — standard softmax + `nll_loss` for
**single-label classification** (integer class target).

**Our organ task** predicts a **probability distribution over 9 organs** (softmax,
sums to 1), with **soft targets** (one-hot for single-organ studies, spread for
biodistribution panels). Neither existing loss consumes a soft `[N,9]` target, so
the one missing piece is a small **soft-target cross-entropy**:
`loss = -(target * log_softmax(logits)).sum(-1).mean()`.

---

## 7. The easy path: the config-driven pipeline

`pipeline/lnp_pipeline.py` wraps all of the above behind one config file and
subcommands (`prepare-json`, `make-schema`, `preprocess-lmdb`, `train`, `infer`,
`predict`). Example existing config: `pipeline/configs/lnpdb_heartkidney.json`.

```bash
# end-to-end fine-tune of one fold, loading the pretrained encoder:
python pipeline/lnp_pipeline.py train --config pipeline/configs/lnpdb_heartkidney.json --fold 0 --clean
```

The config sets the pretrained encoder, schema, encoder sizes, epochs, lr,
`freeze_molecule_encoder`, metric, patience, etc. — i.e. everything in §4–5.

---

## 8. How to fine-tune for the ORGAN task (what we've built + what's left)

What's done:
- **Stage 1** (`dataset_preprocessing.py`) → `data_processed/LNPDB_clean.csv`.
- **Stage 2** (`experiments/dataset_preprocessing_organ.py`) →
  `data_json/LNPDB_organ.json` + `task_schemas/lnpdb_organ_schema.json`
  (single `organ` task, softmax-distribution targets over 9 organs).

What's left to make it runnable:
1. **Stage 3**: build the LMDBs from `LNPDB_organ.json` (LMDB builder / pipeline `preprocess-lmdb`).
2. **Add a soft-target cross-entropy loss** (`@register_loss`, ~30 lines) — §6.
3. **Make/clone a pipeline config** like `lnpdb_heartkidney.json`, pointing at the
   organ JSON + schema, `--num-classes 9`, the new loss, and
   `pretrained_mol_encoder: ckp/mol_pre_no_h_220816.pt`.
4. Train: `python pipeline/lnp_pipeline.py train --config <organ_config>.json --fold 0`.

Caveat to remember: heart/kidney/lung appear only in 5-organ panels, so their
target mass is small and diluted — see the note printed by stage 2. Value-weighting
(re-adding `Experiment_value`) is the lever to sharpen them.

