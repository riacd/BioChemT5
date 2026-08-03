from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from .smiles_tokenizer import SmilesTokenizer, pad_sequences, strip_atom_maps, strip_atom_maps_from_tokens
from .span_masking import make_t5_span_corruption, token_weights_from_center_maps
from .task_views import build_ec_reaction_view, build_ec_reaction_views, build_forward_sample, build_mlm_reaction, build_retro_sample


def _sample_task(task_probs: dict[str, float], rng: random.Random) -> str:
    total = sum(task_probs.values())
    pick = rng.random() * total
    cumulative = 0.0
    for task, prob in task_probs.items():
        cumulative += prob
        if cumulative >= pick:
            return task
    return next(iter(task_probs))


@dataclass
class PretrainCollator:
    tokenizer: SmilesTokenizer
    task_probs: dict[str, float]
    max_source_length: int = 512
    max_target_length: int = 256
    mask_fraction: float = 0.15
    mean_span_len: float = 3.0
    center_weight: float = 4.0
    neighbor_weight: float = 2.0
    base_weight: float = 1.0
    seed: int = 13
    mlm_use_mapped_rxn: bool = True
    ec_views_per_record: int = 1
    seq2seq_enabled: bool = True
    include_ec: bool = True

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if self.ec_views_per_record not in (1, 2):
            raise ValueError("ec_views_per_record must be 1 or 2")

    def _truncate_with_eos(self, ids: list[int], max_length: int) -> list[int]:
        if len(ids) <= max_length:
            return ids
        ids = ids[:max_length]
        ids[-1] = self.tokenizer.eos_token_id
        return ids

    def _encode_source(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(strip_atom_maps(text), add_eos=True)
        return self._truncate_with_eos(ids, self.max_source_length)

    def _encode_target(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(strip_atom_maps(text), add_eos=True)
        return self._truncate_with_eos(ids, self.max_target_length)

    def _build_seq2seq(self, record: dict[str, Any]) -> tuple[str, list[int], list[int]]:
        task = _sample_task(self.task_probs, self.rng)
        if task == "forward":
            source, target = build_forward_sample(record, self.rng)
            return task, self._encode_source(source), self._encode_target(target)
        if task == "retro":
            source, target = build_retro_sample(record, self.rng)
            return task, self._encode_source(source), self._encode_target(target)

        reaction = build_mlm_reaction(record, self.rng, use_mapped_rxn=self.mlm_use_mapped_rxn)
        tokens_with_maps = self.tokenizer.tokenize(reaction)[: self.max_source_length - 1]
        weights = token_weights_from_center_maps(
            tokens_with_maps,
            record,
            base=self.base_weight,
            center=self.center_weight,
            neighbor=self.neighbor_weight,
        )
        tokens = strip_atom_maps_from_tokens(tokens_with_maps)
        corrupted, target, _meta = make_t5_span_corruption(
            tokens=tokens,
            token_weights=weights,
            rng=self.rng,
            mask_fraction=self.mask_fraction,
            mean_span_len=self.mean_span_len,
        )
        mlm_source = [self.tokenizer.token_id(tok) for tok in corrupted] + [self.tokenizer.eos_token_id]
        mlm_target = [self.tokenizer.token_id(tok) for tok in target]
        return "mlm", self._truncate_with_eos(mlm_source, self.max_source_length), self._truncate_with_eos(
            mlm_target, self.max_target_length
        )

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        tasks: list[str] = []
        source_ids: list[list[int]] = []
        target_ids: list[list[int]] = []
        ec_ids: list[list[int]] = []
        ec_pair_ids: list[int] = []
        ec_level_sets: list[dict[str, list[str]]] = []

        for pair_id, record in enumerate(records):
            if self.seq2seq_enabled:
                task, src, tgt = self._build_seq2seq(record)
                tasks.append(task)
                source_ids.append(src)
                target_ids.append(tgt)

            if self.include_ec:
                if self.ec_views_per_record == 1:
                    ec_texts = [build_ec_reaction_view(record, self.rng)]
                else:
                    ec_texts = list(build_ec_reaction_views(record, self.rng))
                levels = record.get("ec_levels") if isinstance(record.get("ec_levels"), dict) else {}
                for text in ec_texts:
                    ec_ids.append(self._encode_source(text))
                    ec_pair_ids.append(pair_id if self.ec_views_per_record == 2 else len(ec_pair_ids))
                    ec_level_sets.append({key: list(levels.get(key) or []) for key in ("ec1", "ec2", "ec3", "ec4")})

        batch = {"tasks": tasks}
        if self.include_ec:
            ec_input_ids, ec_attention_mask = pad_sequences(ec_ids, self.tokenizer.pad_token_id)
            batch.update(
                {
                    "ec_input_ids": torch.tensor(ec_input_ids, dtype=torch.long),
                    "ec_attention_mask": torch.tensor(ec_attention_mask, dtype=torch.long),
                    "ec_pair_ids": torch.tensor(ec_pair_ids, dtype=torch.long),
                    "ec_level_sets": ec_level_sets,
                }
            )
        if self.seq2seq_enabled:
            input_ids, attention_mask = pad_sequences(source_ids, self.tokenizer.pad_token_id)
            labels, _ = pad_sequences(target_ids, self.tokenizer.pad_token_id)
            label_tensor = torch.tensor(labels, dtype=torch.long)
            label_tensor[label_tensor == self.tokenizer.pad_token_id] = -100
            batch.update(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": label_tensor,
                }
            )
        return batch
