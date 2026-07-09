import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from constants import ORGAN_CLASSES


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return list(data.values())
    return data


def label_vector(sample):
    labels = sample.get("labels", {})
    if "organ" in labels:
        return np.asarray(labels["organ"], dtype=float)
    if labels:
        return np.asarray(next(iter(labels.values())), dtype=float)
    return np.zeros(len(ORGAN_CLASSES), dtype=float)


def target_key(target, decimals):
    return tuple(np.round(target.astype(float), decimals=decimals).tolist())


def target_pattern(target):
    parts = [
        f"{organ}:{value:g}"
        for organ, value in zip(ORGAN_CLASSES, target)
        if value > 0
    ]
    return " + ".join(parts) if parts else "[none]"


def sample_id(sample, fallback_idx):
    return str(sample.get("lnp_id", sample.get("id", fallback_idx)))


def composition_key(sample, decimals):
    parts = []
    for component in sample.get("components", []):
        smi = component.get("smi", "")
        comp_type = component.get("component_type", "")
        percent = component.get("percent", component.get("mol", 0.0))
        parts.append(f"{comp_type}:{smi}:{round(float(percent), decimals)}")
    return "|".join(parts)


def summarize_numeric(values):
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def analyze_samples(samples, source_name, target_decimals, composition_decimals):
    targets = np.asarray([label_vector(sample) for sample in samples], dtype=float)
    top_idx = targets.argmax(axis=1) if len(targets) else np.asarray([], dtype=int)
    target_mass = targets.sum(axis=0) if len(targets) else np.zeros(len(ORGAN_CLASSES))

    component_counts = []
    component_type_counts = Counter()
    percent_by_type = defaultdict(list)
    unique_smiles_by_type = defaultdict(set)
    total_percent_sums = []
    multi_target_count = 0
    target_groups = defaultdict(list)
    composition_groups = defaultdict(list)

    for i, sample in enumerate(samples):
        sid = sample_id(sample, i)
        target = label_vector(sample)
        components = sample.get("components", [])
        component_counts.append(len(components))
        total_percent = 0.0

        for component in components:
            comp_type = str(component.get("component_type", ""))
            smi = str(component.get("smi", ""))
            percent = float(component.get("percent", component.get("mol", 0.0)))
            total_percent += percent
            component_type_counts[comp_type] += 1
            percent_by_type[comp_type].append(percent)
            unique_smiles_by_type[comp_type].add(smi)

        total_percent_sums.append(total_percent)
        if (target > 0).sum() > 1:
            multi_target_count += 1
        target_groups[target_key(target, target_decimals)].append(sid)
        composition_groups[composition_key(sample, composition_decimals)].append(sid)

    argmax_counts = Counter(
        ORGAN_CLASSES[int(i)] for i in top_idx
    )
    target_group_rows = []
    target_by_key = {
        target_key(label_vector(sample), target_decimals): label_vector(sample)
        for sample in samples
    }
    for key, ids in target_groups.items():
        target = target_by_key[key]
        target_group_rows.append({
            "source": source_name,
            "count": len(ids),
            "top_organ": ORGAN_CLASSES[int(target.argmax())],
            "target_pattern": target_pattern(target),
            "lnp_ids": "|".join(ids),
            **{
                f"target_{organ}": float(value)
                for organ, value in zip(ORGAN_CLASSES, target)
            },
        })

    repeated_compositions = {
        key: ids for key, ids in composition_groups.items() if len(ids) > 1
    }

    summary = {
        "source": source_name,
        "num_samples": len(samples),
        "num_defined_classes": len(ORGAN_CLASSES),
        "classes_present_by_argmax": [
            organ for organ in ORGAN_CLASSES if argmax_counts[organ] > 0
        ],
        "classes_present_by_target_mass": [
            organ for organ, mass in zip(ORGAN_CLASSES, target_mass) if mass > 0
        ],
        "argmax_counts": dict(argmax_counts),
        "target_mass": {
            organ: float(mass) for organ, mass in zip(ORGAN_CLASSES, target_mass)
        },
        "multi_target_samples": multi_target_count,
        "multi_target_fraction": multi_target_count / len(samples) if samples else 0.0,
        "components_per_lnp": summarize_numeric(component_counts),
        "percent_sum_per_lnp": summarize_numeric(total_percent_sums),
        "component_type_counts": dict(component_type_counts),
        "unique_smiles_by_type": {
            comp_type: len(smiles)
            for comp_type, smiles in unique_smiles_by_type.items()
        },
        "percent_by_type": {
            comp_type: summarize_numeric(values)
            for comp_type, values in percent_by_type.items()
        },
        "num_unique_target_vectors": len(target_groups),
        "num_repeated_target_vectors": sum(
            1 for ids in target_groups.values() if len(ids) > 1
        ),
        "num_unique_compositions": len(composition_groups),
        "num_repeated_compositions": len(repeated_compositions),
    }
    return summary, sorted(target_group_rows, key=lambda row: (-row["count"], row["target_pattern"]))


def class_rows(summary):
    rows = []
    total = summary["num_samples"]
    for organ in ORGAN_CLASSES:
        count = summary["argmax_counts"].get(organ, 0)
        rows.append({
            "source": summary["source"],
            "organ": organ,
            "argmax_count": count,
            "argmax_percent": count / total if total else 0.0,
            "target_mass": summary["target_mass"][organ],
            "target_mass_percent": summary["target_mass"][organ] / total if total else 0.0,
        })
    return rows


def component_rows(summary):
    rows = []
    for comp_type in sorted(summary["component_type_counts"]):
        stats = summary["percent_by_type"][comp_type]
        rows.append({
            "source": summary["source"],
            "component_type": comp_type,
            "count": summary["component_type_counts"][comp_type],
            "unique_smiles": summary["unique_smiles_by_type"].get(comp_type, 0),
            "percent_min": stats["min"],
            "percent_mean": stats["mean"],
            "percent_median": stats["median"],
            "percent_max": stats["max"],
        })
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary, target_groups, top_groups):
    print(f"[{summary['source']}]")
    print(f"  samples: {summary['num_samples']}")
    print(f"  defined classes: {summary['num_defined_classes']}")
    print(
        "  classes present: "
        f"{len(summary['classes_present_by_target_mass'])} by target mass, "
        f"{len(summary['classes_present_by_argmax'])} by argmax"
    )
    print(
        "  multi-target labels: "
        f"{summary['multi_target_samples']} "
        f"({summary['multi_target_fraction']:.2%})"
    )
    comp = summary["components_per_lnp"]
    print(
        "  components/LNP: "
        f"mean={comp['mean']:.2f}, median={comp['median']:.2f}, "
        f"min={comp['min']:.0f}, max={comp['max']:.0f}"
    )
    psum = summary["percent_sum_per_lnp"]
    print(
        "  percent sum/LNP: "
        f"mean={psum['mean']:.4f}, min={psum['min']:.4f}, max={psum['max']:.4f}"
    )
    print(
        "  unique target vectors: "
        f"{summary['num_unique_target_vectors']} "
        f"({summary['num_repeated_target_vectors']} repeated groups)"
    )
    print(
        "  unique compositions: "
        f"{summary['num_unique_compositions']} "
        f"({summary['num_repeated_compositions']} repeated groups)"
    )
    print("  argmax class distribution:")
    total = summary["num_samples"]
    for organ in ORGAN_CLASSES:
        count = summary["argmax_counts"].get(organ, 0)
        if count:
            print(f"    {organ:<16} {count:>5}  {count / total:>7.2%}")
    print("  target mass distribution:")
    for organ in ORGAN_CLASSES:
        mass = summary["target_mass"][organ]
        if mass:
            print(f"    {organ:<16} {mass:>7.2f}  {mass / total:>7.2%}")

    repeated = [row for row in target_groups if row["count"] > 1]
    print(f"  repeated target groups shown: {min(len(repeated), top_groups)}")
    for row in repeated[:top_groups]:
        ids = row["lnp_ids"].split("|")
        preview = ", ".join(ids[:8])
        if len(ids) > 8:
            preview += ", ..."
        print(f"    n={row['count']:<5} {row['target_pattern']} -> {preview}")
    print("")


def source_paths(args):
    if args.input:
        return [(Path(args.input).stem, Path(args.input))]
    if args.all_folds:
        paths = []
        root = Path(args.fold_root)
        for path in sorted(root.glob("fold_V*/lnpdb/*.json")):
            source = f"{path.parents[1].name}_{path.stem}"
            paths.append((source, path))
        return paths
    root = Path(args.fold_root) / f"fold_V{args.fold}" / "lnpdb"
    return [(split, root / f"{split}.json") for split in args.splits]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        help="analyze one JSON file, e.g. experiments/data_json/LNPDB_organ.json",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--fold-root",
        default="experiments/processed_data_dirs/lnpdb_organ_gen",
    )
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-decimals", type=int, default=6)
    parser.add_argument("--composition-decimals", type=int, default=6)
    parser.add_argument("--include-singletons", action="store_true")
    parser.add_argument("--top-groups", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(
        args.output_dir or f"organ_simple/runs/fold_V{args.fold}/raw_data_analysis"
    )

    summaries = []
    all_class_rows = []
    all_component_rows = []
    all_target_groups = []

    for source, path in source_paths(args):
        samples = load_json(path)
        summary, target_groups = analyze_samples(
            samples, source, args.target_decimals, args.composition_decimals
        )
        summary["path"] = str(path)
        summaries.append(summary)
        all_class_rows.extend(class_rows(summary))
        all_component_rows.extend(component_rows(summary))
        if not args.include_singletons:
            target_groups = [row for row in target_groups if row["count"] > 1]
        all_target_groups.extend(target_groups)
        print_summary(summary, target_groups, args.top_groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"sources": summaries}, handle, indent=2)

    write_csv(
        output_dir / "class_distribution.csv",
        all_class_rows,
        [
            "source",
            "organ",
            "argmax_count",
            "argmax_percent",
            "target_mass",
            "target_mass_percent",
        ],
    )
    write_csv(
        output_dir / "component_statistics.csv",
        all_component_rows,
        [
            "source",
            "component_type",
            "count",
            "unique_smiles",
            "percent_min",
            "percent_mean",
            "percent_median",
            "percent_max",
        ],
    )
    write_csv(
        output_dir / "same_target_vectors.csv",
        sorted(all_target_groups, key=lambda row: (-row["count"], row["source"])),
        [
            "source",
            "count",
            "top_organ",
            "target_pattern",
            "lnp_ids",
            *[f"target_{organ}" for organ in ORGAN_CLASSES],
        ],
    )
    print(f"wrote analysis -> {output_dir}")


if __name__ == "__main__":
    main()
