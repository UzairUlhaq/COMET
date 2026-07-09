# Simple Organ Model

Clean two-stage version of the organ classifier:

```text
LNP JSON/LMDB + RDKit conformers -> frozen UniMol molecule encoder
  -> per-component embeddings -> small LNP transformer -> 9-organ distribution
```

The messy UniMol/RDKit part is isolated in one precompute script. The training
script only sees tensors.

## 1. Precompute frozen UniMol component embeddings

Run after `organ_finetune/step3_build_lmdb.py` has created the folds.

```bash
python organ_simple/precompute_unimol_embeddings.py --folds 0 1 2 3 4
```

This writes:

```text
organ_simple/cache/fold_V0/{train,valid,test}.npz
...
```

Each `.npz` contains:

- `component_embeddings`: `[N, C, D]`
- `percents`: `[N, C]`
- `component_types`: `[N, C]`
- `mask`: `[N, C]`
- `target`: `[N, 9]`
- `lnp_ids`

## 2. Train the simple transformer

```bash
python organ_simple/train.py --fold 1
```

Useful options:

```bash
python organ_simple/train.py \
  --fold 1 \
  --embed-dim 256 \
  --layers 2 \
  --heads 4 \
  --batch-size 64 \
  --epochs 100
```

To remove multi-organ soft-label samples during training/validation/test:

```bash
python organ_simple/train.py --fold 1 --single-organ-only
```

With this flag, the loss switches from soft-target cross entropy to standard
hard-label cross entropy using `target.argmax()` as the class label.

To train only on a subset of organs, pass a comma-separated class list. This
filters out samples with no target mass in those organs, remaps targets to the
selected classes, renormalizes them, and changes the model head width.

```bash
python organ_simple/train.py \
  --fold 0 \
  --organ-classes lung_epithelium,liver,muscle \
  --single-organ-only
```

Outputs:

```text
organ_simple/runs/fold_V1/best.pt
organ_simple/runs/fold_V1/metrics.json
```

Metrics include `loss`, `top1`, and `macro_f1`. Because the organ labels are
soft distributions, both `top1` and `macro_f1` use `argmax(target)` as the class
label.

The loss is soft-target cross entropy:

```python
loss = -(target * log_softmax(logits)).sum(dim=-1).mean()
```

## 3. Run inference on a cached split

```bash
python organ_simple/infer.py --fold 1 --split test
```

For a model trained with `--single-organ-only`, use the same filter at
inference/evaluation time:

```bash
python organ_simple/infer.py --fold 1 --split test --single-organ-only
```

This loads `organ_simple/runs/fold_V1/best.pt` and writes:

```text
organ_simple/runs/fold_V1/test_predictions.csv
organ_simple/runs/fold_V1/test_predictions.metrics.json
```

The CSV contains each `lnp_id`, the target organ, predicted organ, target
distribution, and predicted probability for all 9 organs.

## 4. Plot embeddings

```bash
python organ_simple/plot_embeddings.py --fold 1 --split test
```

By default this plots the trained transformer's `cls_rep` embedding with UMAP,
t-SNE, and PHATE when those packages are installed. Outputs go under:

```text
organ_simple/runs/fold_V1/embedding_plots/
```

Useful variants:

```bash
python organ_simple/plot_embeddings.py --fold 0 --split test --embedding both
python organ_simple/plot_embeddings.py --fold 0 --split test --color-by pred
python organ_simple/plot_embeddings.py --fold 0 --split test --methods umap tsne
```

## 5. Scanpy UMAP and Clustering

```bash
python organ_simple/scanpy_analysis/run_scanpy_umap.py --fold 1 --split test
```

This creates an AnnData object from the LNP embeddings, computes a neighbor
graph, UMAP coordinates, optional Leiden clusters, and writes plots plus CSV
summaries under `organ_simple/runs/fold_V1/scanpy_umap/`.

## 6. Analyze cached data

```bash
python organ_simple/analyze_data.py --fold 1
```

This reports the model inputs and outputs, class counts, target-mass
distribution, and groups of LNPs that share the same target vector. It writes:

```text
organ_simple/runs/fold_V1/data_analysis/summary.json
organ_simple/runs/fold_V1/data_analysis/class_distribution.csv
organ_simple/runs/fold_V1/data_analysis/same_outputs.csv
organ_simple/runs/fold_V1/data_analysis/lnp_inputs_outputs.csv
```

## 7. Analyze raw JSON data

```bash

```

This reads the formulation JSON under
`experiments/processed_data_dirs/lnpdb_organ_gen/fold_V1/lnpdb/` and reports
high-level data heuristics: class imbalance, target-mass distribution,
component-count statistics, composition percentages, unique SMILES counts, and
which LNPs share the same target vector.
