from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from biochem_t5.benchmark.data import canonicalize_smiles_set
from biochem_t5.data.smiles_tokenizer import SmilesTokenizer


EC_LEVELS = ("ec1", "ec2", "ec3")


def ec_at_level(ec3: str, level: int) -> str:
    value = str(ec3).strip()
    if value == "-":
        return value
    parts = value.split(".")
    if len(parts) < 3 or level not in (1, 2, 3) or any(not part for part in parts[:3]):
        raise ValueError(f"Invalid EC3 label: {ec3!r}")
    return ".".join(parts[:level])


def exact_reaction_key(reaction: str) -> str:
    text = "".join(str(reaction).split())
    sides = text.split(">>")
    if len(sides) != 2:
        raise ValueError(f"Invalid reaction SMILES: {reaction!r}")
    return ">>".join(".".join(sorted(side.split("."))) for side in sides)


@lru_cache(maxsize=400_000)
def _canonicalize_side(side: str) -> str | None:
    return canonicalize_smiles_set(side)


@lru_cache(maxsize=200_000)
def canonical_reaction_key(reaction: str) -> str | None:
    text = "".join(str(reaction).split())
    sides = text.split(">>")
    if len(sides) != 2:
        return None
    canonical = [_canonicalize_side(side) for side in sides]
    if any(side is None for side in canonical):
        return None
    return ">>".join(str(side) for side in canonical)


@dataclass(frozen=True)
class ECReaction:
    sample_id: str
    reaction: str
    ec1: str
    ec2: str
    ec3: str
    exact_key: str
    canonical_key: str | None

    def label(self, level: str) -> str:
        if level not in EC_LEVELS:
            raise ValueError(f"Unknown EC level: {level}")
        return str(getattr(self, level))


def load_ecreact_csv(path: str | Path, require_labels: bool = True) -> list[ECReaction]:
    path = Path(path)
    records: list[ECReaction] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"rxn_smiles"} | ({"ec3"} if require_labels else set())
        if not required.issubset(fields):
            raise ValueError(f"CSV {path} is missing columns: {sorted(required - fields)}")
        for row_index, row in enumerate(reader):
            reaction = (row.get("rxn_smiles") or "").strip()
            ec3 = (row.get("ec3") or "").strip()
            if not reaction or (require_labels and not ec3):
                raise ValueError(f"Missing reaction or EC label at {path}:{row_index + 2}")
            sample_id = (row.get("sample_id") or row.get("id") or str(row_index)).strip()
            if ec3:
                ec1, ec2 = ec_at_level(ec3, 1), ec_at_level(ec3, 2)
            else:
                ec1 = ec2 = ""
            records.append(
                ECReaction(
                    sample_id=sample_id,
                    reaction=reaction,
                    ec1=ec1,
                    ec2=ec2,
                    ec3=ec3,
                    exact_key=exact_reaction_key(reaction),
                    canonical_key=canonical_reaction_key(reaction),
                )
            )
    return records


def grouped_stratified_split(
    records: Sequence[ECReaction], validation_fraction: float = 0.1, seed: int = 1234
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.canonical_key or record.exact_key].append(index)
    by_label: dict[str, list[list[int]]] = defaultdict(list)
    for indices in grouped.values():
        counts = Counter(records[index].ec3 for index in indices)
        stratum = min(counts, key=lambda label: (-counts[label], label))
        by_label[stratum].append(indices)

    rng = random.Random(seed)
    validation: set[int] = set()
    for label in sorted(by_label):
        groups = list(by_label[label])
        rng.shuffle(groups)
        target = round(sum(len(group) for group in groups) * validation_fraction)
        selected = 0
        for group in groups:
            if selected >= target or (len(groups) == 1 and not validation):
                break
            validation.update(group)
            selected += len(group)
    if not validation:
        smallest = min(grouped.values(), key=len)
        validation.update(smallest)
    train = [index for index in range(len(records)) if index not in validation]
    return train, sorted(validation)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap_audit(train: Sequence[ECReaction], test: Sequence[ECReaction]) -> dict[str, Any]:
    train_exact = {record.exact_key for record in train}
    train_canonical = {record.canonical_key for record in train if record.canonical_key}
    exact_ids = [record.sample_id for record in test if record.exact_key in train_exact]
    canonical_ids = [record.sample_id for record in test if record.canonical_key in train_canonical]
    seen_ids = sorted(set(exact_ids) | set(canonical_ids), key=lambda value: int(value) if value.isdigit() else value)
    return {
        "exact_test_rows": len(exact_ids),
        "canonical_test_rows": len(canonical_ids),
        "seen_test_rows": len(seen_ids),
        "unseen_test_rows": len(test) - len(seen_ids),
        "exact_test_sample_ids": exact_ids,
        "canonical_test_sample_ids": canonical_ids,
        "seen_test_sample_ids": seen_ids,
    }


def pretraining_overlap_audit(path: str | Path, test: Sequence[ECReaction]) -> dict[str, Any]:
    test_by_exact: dict[str, set[str]] = defaultdict(set)
    test_by_canonical: dict[str, set[str]] = defaultdict(set)
    for record in test:
        test_by_exact[record.exact_key].add(record.sample_id)
        if record.canonical_key:
            test_by_canonical[record.canonical_key].add(record.sample_id)
    exact_ids: set[str] = set()
    canonical_ids: set[str] = set()
    rows = candidates = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            payload = json.loads(line)
            reactions = [payload.get("rxn"), payload.get("mapped_rxn")]
            views = payload.get("rsmiles_views") or []
            reactions.extend(views if isinstance(views, list) else [])
            for value in reactions:
                reaction = value.get("rxn") if isinstance(value, dict) else value
                if not isinstance(reaction, str) or ">>" not in reaction:
                    continue
                candidates += 1
                try:
                    exact_ids.update(test_by_exact.get(exact_reaction_key(reaction), ()))
                except ValueError:
                    continue
                canonical = canonical_reaction_key(reaction)
                if canonical:
                    canonical_ids.update(test_by_canonical.get(canonical, ()))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "reaction_views": candidates,
        "exact_test_rows": len(exact_ids),
        "canonical_test_rows": len(canonical_ids),
        "seen_test_rows": len(exact_ids | canonical_ids),
        "seen_test_sample_ids": sorted(exact_ids | canonical_ids, key=lambda value: int(value) if value.isdigit() else value),
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def encode_ec_reaction(tokenizer: SmilesTokenizer, reaction: str, max_length: int = 1200) -> list[int]:
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    ids = [tokenizer.token_id(token) for token in tokenizer.tokenize(f"<ec>{reaction}")]
    return ids[: max_length - 1] + [tokenizer.eos_token_id]


class ReactionDataset(Dataset[ECReaction]):
    def __init__(self, records: Sequence[ECReaction]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ECReaction:
        return self.records[index]


class ReactionCollator:
    def __init__(self, tokenizer: SmilesTokenizer, max_length: int = 1200):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, records: Sequence[ECReaction]) -> dict[str, Any]:
        sequences = [encode_ec_reaction(self.tokenizer, record.reaction, self.max_length) for record in records]
        width = max(map(len, sequences))
        input_ids = torch.full((len(records), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(records), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence)
            attention_mask[row, : len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "records": list(records)}


class TripletDataset(Dataset[tuple[ECReaction, ECReaction, ECReaction]]):
    def __init__(self, records: Sequence[ECReaction], level: str, seed: int = 13):
        self.records = list(records)
        self.level = level
        self.seed = seed
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            buckets[record.label(level)].append(index)
        self.buckets = dict(buckets)
        self.labels = sorted(self.buckets)
        self.anchor_indices = [index for indices in self.buckets.values() if len(indices) > 1 for index in indices]
        if len(self.labels) < 2 or not self.anchor_indices:
            raise ValueError(f"Triplet sampling for {level} needs two labels and a positive pair")

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, item: int) -> tuple[ECReaction, ECReaction, ECReaction]:
        anchor_index = self.anchor_indices[item % len(self.anchor_indices)]
        rng = random.Random(self.seed + item)
        anchor = self.records[anchor_index]
        positives = [index for index in self.buckets[anchor.label(self.level)] if index != anchor_index]
        negative_label = rng.choice([label for label in self.labels if label != anchor.label(self.level)])
        return anchor, self.records[rng.choice(positives)], self.records[rng.choice(self.buckets[negative_label])]


class TripletCollator:
    def __init__(self, tokenizer: SmilesTokenizer, max_length: int = 1200):
        self.base = ReactionCollator(tokenizer, max_length)

    def __call__(self, triplets: Sequence[tuple[ECReaction, ECReaction, ECReaction]]) -> dict[str, Any]:
        flat = [record for triplet in triplets for record in triplet]
        batch = self.base(flat)
        batch["triplet_count"] = len(triplets)
        return batch


class HierarchicalBatchSampler(Sampler[list[int]]):
    """Build 2 EC1 x 2 EC2 x 2 EC3 x 2 reaction batches."""

    def __init__(self, records: Sequence[ECReaction], batches_per_epoch: int | None = None, seed: int = 13):
        self.records = list(records)
        self.seed = seed
        self.epoch = 0
        self.fallback_counts: Counter[str] = Counter()
        tree: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, record in enumerate(records):
            tree[record.ec1][record.ec2][record.ec3].append(index)
        self.tree = tree
        self.ec1_labels = sorted(tree)
        if not self.ec1_labels:
            raise ValueError("Hierarchical sampling needs labeled records")
        self.batches_per_epoch = batches_per_epoch or max(1, len(records) // 16)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _choose(self, values: Sequence[str], count: int, rng: random.Random, name: str) -> list[str]:
        values = list(values)
        if len(values) >= count:
            return rng.sample(values, count)
        self.fallback_counts[name] += count - len(values)
        return values + [rng.choice(values) for _ in range(count - len(values))]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            batch: list[int] = []
            for ec1 in self._choose(self.ec1_labels, 2, rng, "ec1"):
                ec2_values = sorted(self.tree[ec1])
                for ec2 in self._choose(ec2_values, 2, rng, "ec2"):
                    ec3_values = sorted(self.tree[ec1][ec2])
                    for ec3 in self._choose(ec3_values, 2, rng, "ec3"):
                        indices = self.tree[ec1][ec2][ec3]
                        if len(indices) >= 2:
                            batch.extend(rng.sample(indices, 2))
                        else:
                            self.fallback_counts["reaction"] += 1
                            batch.extend([indices[0], indices[0]])
            yield batch


def labels_for(records: Iterable[ECReaction], level: str) -> list[str]:
    return [record.label(level) for record in records]
