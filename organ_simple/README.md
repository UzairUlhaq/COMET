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

Outputs:

```text
organ_simple/runs/fold_V1/best.pt
organ_simple/runs/fold_V1/metrics.json
```

The loss is soft-target cross entropy:

```python
loss = -(target * log_softmax(logits)).sum(dim=-1).mean()
```
