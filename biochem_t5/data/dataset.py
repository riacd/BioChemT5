from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset, IterableDataset

from .task_views import iter_vocab_texts


def template_keys_from_record(record: dict[str, Any], line_no: int | None = None) -> list[str]:
    keys: list[str] = []
    template_id = record.get("primary_template_id")
    if template_id is not None:
        keys.append(f"template:{template_id}")
    template_ids = record.get("template_ids")
    if isinstance(template_ids, list):
        keys.extend(f"template:{item}" for item in template_ids if item is not None)
    if template_id is None:
        template_id = record.get("template_id")
    if template_id is not None:
        keys.append(f"template:{template_id}")
    keys = sorted(set(keys))
    if keys:
        return keys

    rxn_id = record.get("rxn_id")
    if rxn_id is not None:
        return [f"no_template_rxn:{rxn_id}"]
    if line_no is not None:
        return [f"no_template_line:{line_no}"]
    return ["no_template_unknown"]


def split_key_from_record(record: dict[str, Any], line_no: int | None = None) -> str:
    return "|".join(template_keys_from_record(record, line_no=line_no))


class JsonlReactionDataset(Dataset):
    def __init__(self, path: str | Path, max_records: int | None = None):
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        self.ec_level_sets: list[dict[str, list[str]]] = []
        self.split_keys: list[str] = []
        self.template_key_sets: list[list[str]] = []
        with self.path.open("r", encoding="utf-8") as reader:
            for line_no, line in enumerate(reader, start=1):
                if line.strip():
                    record = json.loads(line)
                    levels = record.get("ec_levels") if isinstance(record.get("ec_levels"), dict) else {}
                    template_keys = template_keys_from_record(record, line_no=line_no)
                    self.records.append(record)
                    self.ec_level_sets.append({key: list(levels.get(key) or []) for key in ("ec1", "ec2", "ec3", "ec4")})
                    self.split_keys.append("|".join(template_keys))
                    self.template_key_sets.append(template_keys)
                    if max_records is not None and len(self.records) >= max_records:
                        break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


class IndexedJsonlReactionDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        max_records: int | None = None,
        index_path: str | Path | None = None,
        rebuild_index: bool = False,
    ):
        self.path = Path(path)
        self.max_records = max_records
        self.index_path = Path(index_path) if index_path is not None else self.path.with_suffix(self.path.suffix + ".biochem_t5_index.pkl")
        self._reader = None
        self.offsets: list[int]
        self.ec_level_sets: list[dict[str, list[str]]]
        self.split_keys: list[str]
        self.template_key_sets: list[list[str]]
        if self.index_path.exists() and not rebuild_index:
            self._load_index()
        else:
            self._build_index()

        if max_records is not None:
            self.offsets = self.offsets[:max_records]
            self.ec_level_sets = self.ec_level_sets[:max_records]
            self.split_keys = self.split_keys[:max_records]
            self.template_key_sets = self.template_key_sets[:max_records]

    def _load_index(self) -> None:
        with self.index_path.open("rb") as reader:
            payload = pickle.load(reader)
        self.offsets = list(payload["offsets"])
        self.ec_level_sets = list(payload["ec_level_sets"])
        self.split_keys = list(payload.get("split_keys") or [])
        self.template_key_sets = [list(keys) for keys in payload.get("template_key_sets") or []]
        if len(self.split_keys) != len(self.offsets):
            self.split_keys = [""] * len(self.offsets)
        if len(self.template_key_sets) != len(self.offsets):
            self.template_key_sets = [[key] if key else [] for key in self.split_keys]

    def _build_index(self) -> None:
        offsets: list[int] = []
        ec_level_sets: list[dict[str, list[str]]] = []
        split_keys: list[str] = []
        template_key_sets: list[list[str]] = []
        with self.path.open("rb") as reader:
            line_no = 0
            while True:
                offset = reader.tell()
                line = reader.readline()
                if not line:
                    break
                line_no += 1
                if not line.strip():
                    continue
                record = json.loads(line)
                levels = record.get("ec_levels") if isinstance(record.get("ec_levels"), dict) else {}
                template_keys = template_keys_from_record(record, line_no=line_no)
                offsets.append(offset)
                ec_level_sets.append({key: list(levels.get(key) or []) for key in ("ec1", "ec2", "ec3", "ec4")})
                split_keys.append("|".join(template_keys))
                template_key_sets.append(template_keys)
                if self.max_records is not None and len(offsets) >= self.max_records:
                    break
        self.offsets = offsets
        self.ec_level_sets = ec_level_sets
        self.split_keys = split_keys
        self.template_key_sets = template_key_sets
        if self.max_records is None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
            with tmp_path.open("wb") as writer:
                pickle.dump(
                    {
                        "offsets": self.offsets,
                        "ec_level_sets": self.ec_level_sets,
                        "split_keys": self.split_keys,
                        "template_key_sets": self.template_key_sets,
                    },
                    writer,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp_path.replace(self.index_path)

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_reader(self):
        if self._reader is None or self._reader.closed:
            self._reader = self.path.open("rb")
        return self._reader

    def __getitem__(self, idx: int) -> dict[str, Any]:
        reader = self._get_reader()
        reader.seek(self.offsets[idx])
        return json.loads(reader.readline())


class ReactionDatasetSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)
        base_ec = getattr(dataset, "ec_level_sets", None)
        base_split_keys = getattr(dataset, "split_keys", None)
        base_template_key_sets = getattr(dataset, "template_key_sets", None)
        self.ec_level_sets = [base_ec[idx] for idx in self.indices] if base_ec is not None else []
        self.split_keys = [base_split_keys[idx] for idx in self.indices] if base_split_keys is not None else []
        self.template_key_sets = [base_template_key_sets[idx] for idx in self.indices] if base_template_key_sets is not None else []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[self.indices[idx]]


class JsonlReactionIterableDataset(IterableDataset):
    def __init__(self, path: str | Path, max_records: int | None = None):
        self.path = Path(path)
        self.max_records = max_records

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yielded = 0
        with self.path.open("r", encoding="utf-8") as reader:
            for line in reader:
                if not line.strip():
                    continue
                yield json.loads(line)
                yielded += 1
                if self.max_records is not None and yielded >= self.max_records:
                    return


def iter_texts_for_vocab(path: str | Path, max_records: int | None = None) -> Iterator[str]:
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            record = json.loads(line)
            for text in iter_vocab_texts(record):
                if text:
                    yield text
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return
