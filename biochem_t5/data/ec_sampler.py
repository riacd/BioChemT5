from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Sampler


def _record_ec_labels(record: dict[str, Any], level: str) -> list[str]:
    levels = record.get("ec_levels")
    if not isinstance(levels, dict) and level in record:
        levels = record
    if not isinstance(levels, dict):
        return []
    labels = levels.get(level)
    if not isinstance(labels, list):
        return []
    return [str(label) for label in labels if str(label)]


EC_LEVELS = ("ec1", "ec2", "ec3", "ec4")

HIERARCHICAL_RELATIONS = (
    "same_ec4",
    "same_ec3_diff_ec4",
    "same_ec2_diff_ec3",
    "same_ec1_diff_ec2",
    "diff_ec1",
)


@dataclass
class ECBalancedBatchSampler(Sampler[list[int]]):
    records: Sequence[dict[str, Any]]
    level: str = "ec4"
    ec_keys_per_batch: int = 64
    samples_per_ec: int = 2
    batches_per_epoch: int | None = None
    seed: int = 13

    def __post_init__(self) -> None:
        if self.ec_keys_per_batch <= 0:
            raise ValueError("ec_keys_per_batch must be positive")
        if self.samples_per_ec <= 0:
            raise ValueError("samples_per_ec must be positive")

        buckets: dict[str, set[int]] = defaultdict(set)
        for idx, record in enumerate(self.records):
            for label in _record_ec_labels(record, self.level):
                buckets[label].add(idx)
        self.buckets = {label: sorted(indices) for label, indices in buckets.items() if len(indices) >= self.samples_per_ec}
        self.labels = sorted(self.buckets)
        if len(self.labels) < self.ec_keys_per_batch:
            raise ValueError(
                f"Not enough {self.level} buckets with >= {self.samples_per_ec} records: "
                f"need {self.ec_keys_per_batch}, found {len(self.labels)}"
            )
        self.batch_size = self.ec_keys_per_batch * self.samples_per_ec
        if self.batches_per_epoch is None:
            self.batches_per_epoch = max(1, len(self.records) // self.batch_size)
        self._epoch = 0

    def __len__(self) -> int:
        return int(self.batches_per_epoch or 0)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(len(self)):
            used: set[int] = set()
            batch: list[int] = []
            labels = list(self.labels)
            rng.shuffle(labels)

            for label in labels:
                available = [idx for idx in self.buckets[label] if idx not in used]
                if len(available) < self.samples_per_ec:
                    continue
                selected = rng.sample(available, self.samples_per_ec)
                batch.extend(selected)
                used.update(selected)
                if len(batch) >= self.batch_size:
                    break

            if len(batch) != self.batch_size:
                raise RuntimeError(
                    f"Could not build a full EC-balanced batch for {self.level}: "
                    f"wanted {self.batch_size}, got {len(batch)}"
                )
            yield batch


@dataclass
class ECHierarchicalBatchSampler(Sampler[list[int]]):
    records: Sequence[dict[str, Any]]
    groups_per_batch: int = 21
    relation_names: Sequence[str] = HIERARCHICAL_RELATIONS
    seq2seq_only_samples_per_batch: int = 0
    batches_per_epoch: int | None = None
    seed: int = 13
    max_anchor_attempts: int = 1000
    max_sample_attempts: int = 200

    def __post_init__(self) -> None:
        if self.groups_per_batch <= 0:
            raise ValueError("groups_per_batch must be positive")
        if self.seq2seq_only_samples_per_batch < 0:
            raise ValueError("seq2seq_only_samples_per_batch must be non-negative")
        unknown = sorted(set(self.relation_names).difference(HIERARCHICAL_RELATIONS))
        if unknown:
            raise ValueError(f"Unknown EC hierarchical relations: {unknown}")

        self.relation_names = tuple(self.relation_names)
        self.level_sets: list[dict[str, frozenset[str]]] = []
        buckets: dict[str, dict[str, list[int]]] = {level: defaultdict(list) for level in EC_LEVELS}
        for idx, record in enumerate(self.records):
            sets = {level: frozenset(_record_ec_labels(record, level)) for level in EC_LEVELS}
            self.level_sets.append(sets)
            for level, labels in sets.items():
                for label in labels:
                    buckets[level][label].append(idx)
        self.buckets = {level: dict(label_buckets) for level, label_buckets in buckets.items()}
        self.anchor_indices = [
            idx for idx, levels in enumerate(self.level_sets) if levels["ec1"] and levels["ec2"] and levels["ec3"] and levels["ec4"]
        ]
        self.seq2seq_only_indices = [
            idx
            for idx, levels in enumerate(self.level_sets)
            if not (levels["ec1"] and levels["ec2"] and levels["ec3"] and levels["ec4"])
        ]
        if not self.anchor_indices:
            raise ValueError("No records with complete EC1/EC2/EC3/EC4 labels are available for hierarchical sampling")
        if self.seq2seq_only_samples_per_batch > 0 and not self.seq2seq_only_indices:
            raise ValueError("seq2seq_only_samples_per_batch was requested, but no incomplete-EC records are available")
        self.all_indices = list(self.anchor_indices)
        self.sample_index_set = set(self.all_indices)

        self.group_size = 1 + len(self.relation_names)
        self.batch_size = self.groups_per_batch * self.group_size + self.seq2seq_only_samples_per_batch
        if self.batches_per_epoch is None:
            self.batches_per_epoch = max(1, len(self.anchor_indices) // self.groups_per_batch)
        self._epoch = 0

    def __len__(self) -> int:
        return int(self.batches_per_epoch or 0)

    def _shares(self, idx: int, anchor_idx: int, level: str) -> bool:
        return bool(self.level_sets[idx][level].intersection(self.level_sets[anchor_idx][level]))

    def _matches_relation(self, idx: int, anchor_idx: int, relation: str) -> bool:
        if idx == anchor_idx:
            return False
        if relation == "same_ec4":
            return self._shares(idx, anchor_idx, "ec4")
        if relation == "same_ec3_diff_ec4":
            return self._shares(idx, anchor_idx, "ec3") and not self._shares(idx, anchor_idx, "ec4")
        if relation == "same_ec2_diff_ec3":
            return self._shares(idx, anchor_idx, "ec2") and not self._shares(idx, anchor_idx, "ec3")
        if relation == "same_ec1_diff_ec2":
            return self._shares(idx, anchor_idx, "ec1") and not self._shares(idx, anchor_idx, "ec2")
        if relation == "diff_ec1":
            return not self._shares(idx, anchor_idx, "ec1")
        raise ValueError(f"Unknown EC hierarchical relation: {relation}")

    def _candidate_level(self, relation: str) -> str | None:
        if relation == "same_ec4":
            return "ec4"
        if relation == "same_ec3_diff_ec4":
            return "ec3"
        if relation == "same_ec2_diff_ec3":
            return "ec2"
        if relation == "same_ec1_diff_ec2":
            return "ec1"
        return None

    def _sample_relation(self, anchor_idx: int, relation: str, used: set[int], rng: random.Random) -> int | None:
        level = self._candidate_level(relation)
        anchor_labels = self.level_sets[anchor_idx][level] if level is not None else ()

        for _ in range(self.max_sample_attempts):
            if level is None:
                idx = rng.choice(self.all_indices)
            else:
                label = rng.choice(tuple(anchor_labels))
                bucket = self.buckets[level].get(label) or []
                if not bucket:
                    continue
                idx = rng.choice(bucket)
            if idx in self.sample_index_set and idx not in used and self._matches_relation(idx, anchor_idx, relation):
                return idx
        return None

    def _sample_group(self, used: set[int], rng: random.Random) -> list[int] | None:
        for _ in range(self.max_anchor_attempts):
            anchor_idx = rng.choice(self.anchor_indices)
            if anchor_idx in used:
                continue
            group = [anchor_idx]
            group_used = set(used)
            group_used.add(anchor_idx)
            for relation in self.relation_names:
                idx = self._sample_relation(anchor_idx, relation, group_used, rng)
                if idx is None:
                    break
                group.append(idx)
                group_used.add(idx)
            if len(group) == self.group_size:
                return group
        return None

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(len(self)):
            used: set[int] = set()
            batch: list[int] = []
            for _group_idx in range(self.groups_per_batch):
                group = self._sample_group(used, rng)
                if group is None:
                    raise RuntimeError(
                        "Could not build a full EC hierarchical batch. "
                        "Try reducing groups_per_batch, reducing relation_names, or using a larger EC-labeled corpus slice."
                    )
                batch.extend(group)
                used.update(group)
            if self.seq2seq_only_samples_per_batch:
                available = [idx for idx in self.seq2seq_only_indices if idx not in used]
                if len(available) >= self.seq2seq_only_samples_per_batch:
                    selected = rng.sample(available, self.seq2seq_only_samples_per_batch)
                else:
                    selected = list(available)
                    while len(selected) < self.seq2seq_only_samples_per_batch:
                        selected.append(rng.choice(self.seq2seq_only_indices))
                batch.extend(selected)
            yield batch


@dataclass
class DistributedBatchSampler(Sampler[list[int]]):
    batch_sampler: Sampler[list[int]]
    num_replicas: int
    rank: int

    def __post_init__(self) -> None:
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

    def __len__(self) -> int:
        total = len(self.batch_sampler)  # type: ignore[arg-type]
        return (total + self.num_replicas - 1 - self.rank) // self.num_replicas

    def __iter__(self) -> Iterator[list[int]]:
        for batch_idx, batch in enumerate(self.batch_sampler):
            if batch_idx % self.num_replicas == self.rank:
                yield batch
