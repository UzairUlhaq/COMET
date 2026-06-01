# COMET LNP Pipeline

This folder is the clean, config-driven workflow for preparing data, training,
running inference, and comparing experiments. The old `experiments/*.py` scripts
are still there for provenance, but new work should start here.

## Quick Start

From the repo root:

```bash
python pipeline/lnp_pipeline.py prepare-json --config pipeline/configs/lnpdb_heartkidney.json
python pipeline/lnp_pipeline.py make-schema --config pipeline/configs/lnpdb_heartkidney.json
python pipeline/lnp_pipeline.py inspect --config pipeline/configs/lnpdb_heartkidney.json --fold 0
```

LMDB preprocessing needs RDKit, so use the COMET environment:

```bash
conda activate comet_env
python pipeline/lnp_pipeline.py preprocess-lmdb --config pipeline/configs/lnpdb_heartkidney.json
```

Train and run test inference:

```bash
python pipeline/lnp_pipeline.py train --config pipeline/configs/lnpdb_heartkidney.json --fold 0 --clean
```

Use `--clean` when starting a fresh experiment. UniMol automatically resumes
`checkpoint_last.pt` if it exists in the save directory, so stale checkpoints
from older model/data settings can cause architecture mismatch errors. Use
`--resume` only when you intentionally want to continue the same run.

Preview the exact command without running:

```bash
python pipeline/lnp_pipeline.py train --config pipeline/configs/lnpdb_heartkidney.json --fold 0 --dry-run
```

Run inference only:

```bash
python pipeline/lnp_pipeline.py infer --config pipeline/configs/lnpdb_heartkidney.json --fold 0
```

Train the in-house CACO2/B16F10/DC24 dataset:

```bash
python pipeline/lnp_pipeline.py inspect --config pipeline/configs/in_house_caco2.json --fold 0
python pipeline/lnp_pipeline.py train --config pipeline/configs/in_house_caco2.json --fold 0 --clean
```

Or use the convenience wrapper:

```bash
bash pipeline/train_in_house_caco2.sh
```

Add `--wandb` to stream training loss and validation metrics:

```bash
bash pipeline/train_in_house_caco2.sh --wandb
```

Score new LNP candidates:

```bash
python pipeline/lnp_pipeline.py predict \
  --config pipeline/configs/lnpdb_heartkidney.json \
  --input-json pipeline/examples/new_lnp.json
```

Summarize metrics and losses:

```bash
python pipeline/lnp_pipeline.py summarize-results --config pipeline/configs/lnpdb_heartkidney.json
python pipeline/lnp_pipeline.py summarize-logs --config pipeline/configs/lnpdb_heartkidney.json --contains bs8
```

Log losses and metrics to Weights & Biases:

```bash
python pipeline/lnp_pipeline.py train --config pipeline/configs/lnpdb_heartkidney.json --fold 0 --clean --wandb
```

With `--wandb`, training loss and validation metrics stream live while the
model is running, then the final test metrics and best checkpoint are attached
to the same W&B run.

To upload an already-finished run:

```bash
python pipeline/lnp_pipeline.py wandb-log --config pipeline/configs/lnpdb_heartkidney.json --fold 0
```

Use `--wandb-mode offline` if you want W&B to write a local offline run first.

Run a small hyperparameter sweep:

```bash
python pipeline/lnp_pipeline.py sweep \
  --name lr_bs_search \
  --folds 0 \
  --lr 3e-5 1e-4 3e-4 \
  --batch-size 4 8 \
  --cagrad-c 0 \
  --percent-noise 0 0.1 \
  --contrast-margin-coeff 0.01 \
  --clean \
  --skip-existing
```

The sweep writes resolved configs and a manifest under `pipeline/sweeps/<name>/`.
Summarize completed runs by a metric:

```bash
python pipeline/lnp_pipeline.py summarize-sweep \
  --manifest pipeline/sweeps/lr_bs_search/manifest.json \
  --metric test/heart_test_spearmanr_coeff
```

## Running A New Label Combination

Copy the config:

```bash
cp pipeline/configs/lnpdb_heartkidney.json pipeline/configs/lnpdb_organs.json
```

Edit these fields:

```json
"name": "lnpdb_organs",
"json_path": "experiments/data_json/LNPDB_organs.json",
"processed_root": "experiments/processed_data_dirs/lnpdb_organs_gen",
"schema_path": "experiments/task_schemas/lnpdb_organs_schema.json",
"target_labels": ["liver", "spleen", "lung", "heart", "kidney"],
"outputs": {
  "save_root": "experiments/save_lnpdb_organs",
  "tmp_save_root": "experiments/tmp_save_lnpdb_organs",
  "log_root": "experiments/logs/tmp",
  "infer_root": "experiments/infer_results",
  "new_lnp_lmdb": "experiments/inference_inputs/organs_new_lnp_lmdb",
  "new_lnp_results": "experiments/infer_results/organs_new_lnp_predictions"
}
```

Then run:

```bash
python pipeline/lnp_pipeline.py prepare-json --config pipeline/configs/lnpdb_organs.json
python pipeline/lnp_pipeline.py make-schema --config pipeline/configs/lnpdb_organs.json
python pipeline/lnp_pipeline.py preprocess-lmdb --config pipeline/configs/lnpdb_organs.json
python pipeline/lnp_pipeline.py train --config pipeline/configs/lnpdb_organs.json --fold 0
```

## Notes

- This pipeline fine-tunes by default: it loads `ckp/mol_pre_no_h_220816.pt`.
- `freeze_molecule_encoder: true` keeps GPU memory lower and is recommended for small datasets.
- Set `pretrained_mol_encoder` to `null` and `freeze_molecule_encoder` to `false` only if you intentionally want training from scratch.
- New-LNP predictions are regression scores, not calibrated probabilities.


## Sweep

python pipeline/lnp_pipeline.py sweep \
  --config pipeline/configs/lnpdb_heartkidney.json \
  --name quick_lr_bs_search \
  --folds 0 \
  --lr 3e-5 \
  --batch-size  8 \
  --percent-noise 0.1 \
  --cagrad-c 0 \
  --contrast-margin-coeff 0.01 \
  --clean \
  --skip-existing \
  --wandb
