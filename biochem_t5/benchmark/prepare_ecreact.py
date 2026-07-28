from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .ecreact import (
    grouped_stratified_split,
    load_ecreact_csv,
    overlap_audit,
    pretraining_overlap_audit,
    sha256_file,
    write_json,
)


def prepare(
    train_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.1,
    split_seed: int = 1234,
    expected_train_rows: int | None = 166_918,
    expected_test_rows: int | None = 18_816,
    pretrain_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    train_csv, test_csv, output_dir = Path(train_csv), Path(test_csv), Path(output_dir)
    train = load_ecreact_csv(train_csv)
    test = load_ecreact_csv(test_csv)
    if expected_train_rows is not None and len(train) != expected_train_rows:
        raise ValueError(f"Expected {expected_train_rows} train rows, found {len(train)}")
    if expected_test_rows is not None and len(test) != expected_test_rows:
        raise ValueError(f"Expected {expected_test_rows} test rows, found {len(test)}")
    train_indices, validation_indices = grouped_stratified_split(train, validation_fraction, split_seed)
    grouped_labels: dict[str, set[str]] = {}
    for record in train:
        grouped_labels.setdefault(record.canonical_key or record.exact_key, set()).add(record.ec3)
    train_groups = {train[index].canonical_key or train[index].exact_key for index in train_indices}
    validation_groups = {train[index].canonical_key or train[index].exact_key for index in validation_indices}
    if train_groups & validation_groups:
        raise AssertionError("Normalized reaction groups crossed the internal split")
    manifest = {
        "version": 1,
        "files": {
            "train": {"path": str(train_csv), "sha256": sha256_file(train_csv), "rows": len(train)},
            "test": {"path": str(test_csv), "sha256": sha256_file(test_csv), "rows": len(test)},
        },
        "split": {
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "train_ec3_counts": dict(Counter(train[index].ec3 for index in train_indices)),
            "validation_ec3_counts": dict(Counter(train[index].ec3 for index in validation_indices)),
        },
        "audit": overlap_audit(train, test),
    }
    manifest["audit"]["conflicting_train_reaction_groups"] = sum(
        len(labels) > 1 for labels in grouped_labels.values()
    )
    if pretrain_jsonl is not None:
        manifest["audit"]["pretraining_test_overlap"] = pretraining_overlap_audit(pretrain_jsonl, test)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_manifest.json", manifest)
    write_json(output_dir / "data_audit.json", manifest["audit"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official ECREACT/CLAIRE retrieval split")
    parser.add_argument("--train-csv", default="benchmark/ECreact_bench/data/train_augmented.csv")
    parser.add_argument("--test-csv", default="benchmark/ECreact_bench/data/test_augmented.csv")
    parser.add_argument("--output-dir", default="outputs/BiochemT5/benchmark/ecreact_t5/data")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--no-count-check", action="store_true")
    parser.add_argument(
        "--pretrain-jsonl",
        help="Optional JSONL for the expensive full pretraining/test overlap audit.",
    )
    args = parser.parse_args()
    manifest = prepare(
        args.train_csv,
        args.test_csv,
        args.output_dir,
        args.validation_fraction,
        args.split_seed,
        None if args.no_count_check else 166_918,
        None if args.no_count_check else 18_816,
        args.pretrain_jsonl,
    )
    print(f"train={manifest['files']['train']['rows']} test={manifest['files']['test']['rows']}")


if __name__ == "__main__":
    main()
