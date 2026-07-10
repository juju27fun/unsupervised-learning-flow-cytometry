#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read dataset.yaml") from exc
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def count_labels(path: Path) -> tuple[int, bool]:
    if not path.exists():
        return 0, False
    n_labels = 0
    has_ambiguous = False
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            n_labels += 1
            try:
                has_ambiguous = has_ambiguous or int(float(parts[0])) == 3
            except ValueError:
                pass
    return n_labels, has_ambiguous


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def rows_from_yolo_dataset(source: Path, repo_root: Path) -> list[dict[str, str]]:
    yaml_path = source / "dataset.yaml"
    data = load_yaml(yaml_path)
    dataset_root = Path(data.get("path", source))
    if not dataset_root.is_absolute():
        dataset_root = (source / dataset_root).resolve()

    rows: list[dict[str, str]] = []
    split_map = {
        "train": data.get("train", "train/signals"),
        "val": data.get("val", "val/signals"),
        "test": data.get("test", "test/signals"),
    }
    for split, split_dir in split_map.items():
        signals_dir = dataset_root / str(split_dir)
        if not signals_dir.exists():
            continue
        for signal_path in sorted(signals_dir.glob("*.npy")):
            label_path = dataset_root / split / "labels" / f"{signal_path.stem}.txt"
            n_labels, has_ambiguous = count_labels(label_path)
            rows.append(
                {
                    "split": split,
                    "id": signal_path.stem,
                    "signal_path": rel_or_abs(signal_path, repo_root),
                    "label_path": rel_or_abs(label_path, repo_root) if label_path.exists() else "",
                    "source_kind": "particle" if n_labels > 0 else "unlabeled_or_background",
                    "n_labels": str(n_labels),
                    "has_ambiguous": "1" if has_ambiguous else "0",
                }
            )
    return rows


def rows_from_p2_manifest(source: Path, repo_root: Path, prefer_source_path: bool = True) -> list[dict[str, str]]:
    manifest_path = source / "manifest.csv"
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chosen = row.get("source_path") if prefer_source_path else row.get("feature_path")
            if not chosen:
                chosen = row.get("feature_path") or row.get("source_path")
            if not chosen:
                continue
            signal_path = Path(chosen)
            if not signal_path.is_absolute():
                signal_path = (repo_root / signal_path).resolve()
            label = row.get("label_path") or ""
            label_path = Path(label) if label else None
            if label_path is not None and not label_path.is_absolute():
                label_path = (source / label_path).resolve()
            rows.append(
                {
                    "split": row.get("split", ""),
                    "id": row.get("id", signal_path.stem),
                    "signal_path": rel_or_abs(signal_path, repo_root),
                    "label_path": rel_or_abs(label_path, repo_root) if label_path and label_path.exists() else "",
                    "source_kind": row.get("source_kind", "unknown"),
                    "n_labels": row.get("n_labels", "0"),
                    "has_ambiguous": row.get("has_ambiguous", "0"),
                }
            )
    return rows


def rows_from_npy_tree(source: Path, repo_root: Path, split_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for signal_path in sorted(source.rglob("*.npy")):
        rows.append(
            {
                "split": split_name,
                "id": signal_path.stem,
                "signal_path": rel_or_abs(signal_path, repo_root),
                "label_path": "",
                "source_kind": signal_path.parent.name,
                "n_labels": "0",
                "has_ambiguous": "0",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an isolated P3_SSL manifest from existing read-only sources.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--prefer-feature-path", action="store_true", help="For P2 manifests, read feature_path instead of source_path.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    source = args.source
    if not source.is_absolute():
        source = (repo_root / source).resolve()

    if (source / "dataset.yaml").exists():
        rows = rows_from_yolo_dataset(source, repo_root)
    elif (source / "manifest.csv").exists():
        rows = rows_from_p2_manifest(source, repo_root, prefer_source_path=not args.prefer_feature_path)
    else:
        rows = rows_from_npy_tree(source, repo_root, split_name=args.split_name)

    if not rows:
        raise SystemExit(f"No .npy rows found under {source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "id", "signal_path", "label_path", "source_kind", "n_labels", "has_ambiguous"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

