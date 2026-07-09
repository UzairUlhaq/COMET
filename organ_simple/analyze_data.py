import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from constants import COMPONENT_TYPES, ORGAN_CLASSES


def target_key(target, decimals):
    return tuple(np.round(target.astype(float), decimals=decimals).tolist())


def target_name(target):
    active = [
        f"{organ}:{value:g}"
        for organ, value in zip(ORGAN_CLASSES, target)
        if value > 0
    ]
    return " + ".join(active) if active else "[none]"


def load_split(cache_dir, fold, split):
    path = Path(cache_dir) / f"fold_V{fold}" / f"{split}.npz"
    data = np.load(path, allow_pickle=True)
    return path, {
        "component_embeddings": data["component_embeddings"],
        "percents": data["percents"],
        "component_types": data["component_types"],
        "mask": data["mask"].astype(bool),
        "target": data["target"],
        "lnp_ids": data["lnp_ids"].tolist(),
    }

def summarize_inputs(payload):
    mask = payload["mask"]
    component_counts = mask.sum(axis=1)
    type_counts = Counter()
    for row_types, row_mask in zip(payload["component_types"], mask):
        for type_id in row_types[row_mask]:
            type_name = COMPONENT_TYPES[int(type_id)]
            type_counts[type_name] += 1

    return {
        "component_embeddings_shape": list(payload["component_embeddings"].shape),
        "percents_shape": list(payload["percents"].shape),
        "component_types_shape": list(payload["component_types"].shape),
        "mask_shape": list(payload["mask"].shape),
        "target_shape": list(payload["target"].shape),
        "components_per_lnp": {
            "min": int(component_counts.min()) if len(component_counts) else 0,
            "max": int(component_counts.max()) if len(component_counts) else 0,
            "mean": float(component_counts.mean()) if len(component_counts) else 0.0,
        },
        "component_type_counts": dict(type_counts),
    }


def summarize_classes(target):
    top_idx = target.argmax(axis=1)
    top_counts = Counter(ORGAN_CLASSES[int(i)] for i in top_idx)
    target_mass = target.sum(axis=0)
    present_by_top = [organ for organ in ORGAN_CLASSES if top_counts[organ] > 0]
    present_by_mass = [
        organ for organ, mass in zip(ORGAN_CLASSES, target_mass) if mass > 0
    ]

    return {
        "num_defined_classes": len(ORGAN_CLASSES),
        "num_classes_present_by_argmax": len(present_by_top),
        "num_classes_present_by_target_mass": len(present_by_mass),
        "classes_present_by_argmax": present_by_top,
        "classes_present_by_target_mass": present_by_mass,
        "argmax_counts": dict(top_counts),
        "target_mass": {
            organ: float(mass) for organ, mass in zip(ORGAN_CLASSES, target_mass)
        },
    }


def build_lnp_rows(split, payload):
    rows = []
    for i, lnp_id in enumerate(payload["lnp_ids"]):
        mask = payload["mask"][i]
        target = payload["target"][i]
        top_idx = int(target.argmax())
        component_type_names = [
            COMPONENT_TYPES[int(type_id)]
            for type_id in payload["component_types"][i][mask]
        ]
        percents = payload["percents"][i][mask]
        rows.append({
            "split": split,
            "lnp_id": lnp_id,
            "num_components": int(mask.sum()),
            "component_types": "|".join(component_type_names),
            "percents": "|".join(f"{float(value):.6g}" for value in percents),
            "top_organ": ORGAN_CLASSES[top_idx],
            "target_pattern": target_name(target),
            **{
                f"target_{organ}": float(value)
                for organ, value in zip(ORGAN_CLASSES, target)
            },
        })
    return rows


def group_same_outputs(split, payload, decimals):
    groups = defaultdict(list)
    target_by_key = {}
    for lnp_id, target in zip(payload["lnp_ids"], payload["target"]):
        key = target_key(target, decimals)
        groups[key].append(str(lnp_id))
        target_by_key[key] = target

    rows = []
    for key, lnp_ids in groups.items():
        target = target_by_key[key]
        top_idx = int(target.argmax())
        rows.append({
            "split": split,
            "count": len(lnp_ids),
            "top_organ": ORGAN_CLASSES[top_idx],
            "target_pattern": target_name(target),
            "lnp_ids": "|".join(lnp_ids),
            **{
                f"target_{organ}": float(value)
                for organ, value in zip(ORGAN_CLASSES, target)
            },
        })
    return sorted(rows, key=lambda row: (-row["count"], row["target_pattern"]))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, summary, class_rows, same_output_rows, lnp_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    write_csv(
        output_dir / "class_distribution.csv",
        class_rows,
        ["split", "organ", "argmax_count", "argmax_percent", "target_mass"],
    )
    write_csv(
        output_dir / "same_outputs.csv",
        same_output_rows,
        [
            "split",
            "count",
            "top_organ",
            "target_pattern",
            "lnp_ids",
            *[f"target_{organ}" for organ in ORGAN_CLASSES],
        ],
    )
    write_csv(
        output_dir / "lnp_inputs_outputs.csv",
        lnp_rows,
        [
            "split",
            "lnp_id",
            "num_components",
            "component_types",
            "percents",
            "top_organ",
            "target_pattern",
            *[f"target_{organ}" for organ in ORGAN_CLASSES],
        ],
    )


def print_summary(summary, same_output_rows, top_groups):
    print(f"fold: V{summary['fold']}")
    print(f"splits: {', '.join(summary['splits'])}")
    print(f"defined classes: {summary['num_defined_classes']}")
    print("")

    for split, split_summary in summary["split_summaries"].items():
        print(f"[{split}]")
        print(f"  samples: {split_summary['num_samples']}")
        print(
            "  input: "
            f"component_embeddings={split_summary['input']['component_embeddings_shape']}, "
            f"percents={split_summary['input']['percents_shape']}, "
            f"component_types={split_summary['input']['component_types_shape']}, "
            f"mask={split_summary['input']['mask_shape']}"
        )
        print(f"  output: target={split_summary['input']['target_shape']}")
        print(
            "  classes present: "
            f"{split_summary['classes']['num_classes_present_by_target_mass']} by target mass, "
            f"{split_summary['classes']['num_classes_present_by_argmax']} by argmax"
        )
        print("  argmax class distribution:")
        counts = split_summary["classes"]["argmax_counts"]
        total = split_summary["num_samples"]
        for organ in ORGAN_CLASSES:
            count = counts.get(organ, 0)
            if count:
                print(f"    {organ:<16} {count:>5}  {count / total:>7.2%}")
        print("")

    repeated = [row for row in same_output_rows if row["count"] > 1]
    print(f"same-output groups with at least 2 LNPs: {len(repeated)}")
    for row in repeated[:top_groups]:
        ids = row["lnp_ids"].split("|")
        preview = ", ".join(ids[:8])
        if len(ids) > 8:
            preview += ", ..."
        print(
            f"  [{row['split']}] n={row['count']} "
            f"{row['target_pattern']} -> {preview}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument("--cache-dir", default="organ_simple/cache")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--target-decimals",
        type=int,
        default=6,
        help="round target vectors before grouping same-output LNPs",
    )
    parser.add_argument("--include-singletons", action="store_true")
    parser.add_argument("--top-groups", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(
        args.output_dir
        or f"organ_simple/runs/fold_V{args.fold}/data_analysis"
    )

    summary = {
        "fold": args.fold,
        "splits": args.splits,
        "num_defined_classes": len(ORGAN_CLASSES),
        "class_names": ORGAN_CLASSES,
        "split_summaries": {},
    }
    class_rows = []
    same_output_rows = []
    lnp_rows = []

    for split in args.splits:
        path, payload = load_split(args.cache_dir, args.fold, split)
        input_summary = summarize_inputs(payload)
        class_summary = summarize_classes(payload["target"])
        split_summary = {
            "path": str(path),
            "num_samples": len(payload["lnp_ids"]),
            "input": input_summary,
            "classes": class_summary,
        }
        summary["split_summaries"][split] = split_summary

        total = len(payload["lnp_ids"])
        argmax_counts = class_summary["argmax_counts"]
        for organ in ORGAN_CLASSES:
            count = argmax_counts.get(organ, 0)
            class_rows.append({
                "split": split,
                "organ": organ,
                "argmax_count": count,
                "argmax_percent": count / total if total else 0.0,
                "target_mass": class_summary["target_mass"][organ],
            })

        split_same_outputs = group_same_outputs(
            split, payload, args.target_decimals
        )
        if not args.include_singletons:
            split_same_outputs = [
                row for row in split_same_outputs if row["count"] > 1
            ]
        same_output_rows.extend(split_same_outputs)
        lnp_rows.extend(build_lnp_rows(split, payload))

    same_output_rows.sort(key=lambda row: (-row["count"], row["split"]))
    write_outputs(output_dir, summary, class_rows, same_output_rows, lnp_rows)
    print_summary(summary, same_output_rows, args.top_groups)
    print("")
    print(f"wrote analysis -> {output_dir}")


if __name__ == "__main__":
    main()
