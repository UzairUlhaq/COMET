# Scanpy Analysis

Scanpy gives us a graph-analysis workflow around the LNP embeddings:

```text
model embeddings -> AnnData -> neighbors -> UMAP -> Leiden clusters -> tables/plots
```

Install optional dependencies in the environment you use for plotting:

```bash
pip install scanpy igraph leidenalg
```

Run on the learned `cls` embeddings:

```bash
python organ_simple/scanpy_analysis/run_scanpy_umap.py --fold 1 --split test
```

Run on single-organ samples only:

```bash
python organ_simple/scanpy_analysis/run_scanpy_umap.py \
  --fold 1 \
  --split test \
  --single-organ-only
```

Compare with the frozen component-mean embedding:

```bash
python organ_simple/scanpy_analysis/run_scanpy_umap.py \
  --fold 1 \
  --split test \
  --embedding mean_components
```

Outputs go to:

```text
organ_simple/runs/fold_V1/scanpy_umap/
```

Files:

- `*_umap.csv`: UMAP coordinates, LNP IDs, targets, predictions, and probabilities.
- `*_leiden_summary.csv`: cluster sizes, dominant target organ, dominant predicted organ, accuracy, and LNP IDs.
- `*_umap_*.png`: UMAP colored by target, prediction, correctness, number of target organs, and Leiden cluster.
- `*.h5ad`: the full AnnData object for later interactive analysis.

