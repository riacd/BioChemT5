from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from biochem_t5.data.smiles_tokenizer import SmilesTokenizer
from biochem_t5.benchmark.common import canonicalize_smiles_set, sha256_file


@dataclass
class CsvLoadResult:
    path: Path
    products: dict[str, list[str]]
    stats: dict[str, int]
def load_retrosynthesis_csv(
    path: str | Path,
    product_column: str = "product_smiles",
    target_column: str = "substrate_smiles",
) -> CsvLoadResult:
    path = Path(path)
    stats = {
        "rows": 0,
        "missing_fields": 0,
        "invalid_products": 0,
        "invalid_targets": 0,
        "valid_rows": 0,
        "duplicate_pairs": 0,
        "unique_products": 0,
        "unique_pairs": 0,
    }
    grouped: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {product_column, target_column}
        if not required.issubset(fieldnames):
            missing = ", ".join(sorted(required - fieldnames))
            raise ValueError(f"CSV {path} is missing required columns: {missing}")
        for row in reader:
            stats["rows"] += 1
            product_raw = (row.get(product_column) or "").strip()
            target_raw = (row.get(target_column) or "").strip()
            if not product_raw or not target_raw:
                stats["missing_fields"] += 1
                continue
            product = canonicalize_smiles_set(product_raw)
            if product is None:
                stats["invalid_products"] += 1
                continue
            target = canonicalize_smiles_set(target_raw)
            if target is None:
                stats["invalid_targets"] += 1
                continue
            targets = grouped.setdefault(product, set())
            if target in targets:
                stats["duplicate_pairs"] += 1
                continue
            targets.add(target)
            stats["valid_rows"] += 1

    products = {product: sorted(targets) for product, targets in sorted(grouped.items())}
    stats["unique_products"] = len(products)
    stats["unique_pairs"] = sum(len(targets) for targets in products.values())
    return CsvLoadResult(path=path, products=products, stats=stats)


def load_retrosynthesis_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    loaded = {
        "train": load_retrosynthesis_csv(train_csv),
        "val": load_retrosynthesis_csv(val_csv),
        "test": load_retrosynthesis_csv(test_csv),
    }
    splits = {name: dict(result.products) for name, result in loaded.items()}
    manifest: dict[str, Any] = {
        "files": {
            name: {
                "path": str(result.path),
                "sha256": sha256_file(result.path),
                **result.stats,
            }
            for name, result in loaded.items()
        },
        "final": {
            name: {
                "products": len(products),
                "pairs": sum(len(targets) for targets in products.values()),
            }
            for name, products in splits.items()
        },
    }
    return splits, manifest


def encode_with_limit(
    tokenizer: SmilesTokenizer,
    text: str,
    max_length: int,
) -> tuple[list[int], dict[str, int]]:
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    tokens = tokenizer.tokenize(text)
    ids = [tokenizer.token_id(token) for token in tokens]
    unk_tokens = sum(token_id == tokenizer.unk_token_id for token_id in ids)
    truncated = int(len(ids) + 1 > max_length)
    ids = ids[: max_length - 1] + [tokenizer.eos_token_id]
    return ids, {
        "tokens": len(tokens),
        "unk_tokens": unk_tokens,
        "sequences": 1,
        "sequences_with_unk": int(unk_tokens > 0),
        "truncated_sequences": truncated,
    }


def tokenizer_audit(
    tokenizer: SmilesTokenizer,
    products: Mapping[str, Sequence[str]],
    source_max_length: int,
    target_max_length: int,
) -> dict[str, dict[str, int | float]]:
    counters = {
        "source": {key: 0 for key in ("tokens", "unk_tokens", "sequences", "sequences_with_unk", "truncated_sequences")},
        "target": {key: 0 for key in ("tokens", "unk_tokens", "sequences", "sequences_with_unk", "truncated_sequences")},
    }
    for product, targets in products.items():
        _, source_stats = encode_with_limit(tokenizer, f"<retro>{product}", source_max_length)
        for key, value in source_stats.items():
            counters["source"][key] += value
        for target in targets:
            _, target_stats = encode_with_limit(tokenizer, target, target_max_length)
            for key, value in target_stats.items():
                counters["target"][key] += value
    result: dict[str, dict[str, int | float]] = {}
    for side, values in counters.items():
        tokens = values["tokens"]
        result[side] = {
            **values,
            "unk_rate": values["unk_tokens"] / tokens if tokens else 0.0,
        }
    return result


class RetrosynthesisDataset(Dataset[tuple[str, str]]):
    def __init__(self, products: Mapping[str, Sequence[str]]):
        self.pairs = [
            (product, target)
            for product in sorted(products)
            for target in sorted(set(products[product]))
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


class RetrosynthesisCollator:
    def __init__(self, tokenizer: SmilesTokenizer, source_max_length: int = 512, target_max_length: int = 512):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __call__(self, batch: Sequence[tuple[str, str]]) -> dict[str, torch.Tensor]:
        source_ids = [
            encode_with_limit(self.tokenizer, f"<retro>{product}", self.source_max_length)[0]
            for product, _target in batch
        ]
        target_ids = [
            encode_with_limit(self.tokenizer, target, self.target_max_length)[0]
            for _product, target in batch
        ]
        source_width = max(len(ids) for ids in source_ids)
        target_width = max(len(ids) for ids in target_ids)
        input_ids = torch.full((len(batch), source_width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), source_width), dtype=torch.long)
        labels = torch.full((len(batch), target_width), -100, dtype=torch.long)
        for row, ids in enumerate(source_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        for row, ids in enumerate(target_ids):
            labels[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
