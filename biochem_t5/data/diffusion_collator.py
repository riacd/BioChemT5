from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch

from .collator import _sample_task
from .smiles_tokenizer import SmilesTokenizer, pad_sequences, strip_atom_maps, strip_atom_maps_from_tokens
from .span_masking import token_weights_from_center_maps
from .task_views import (
    build_ec_reaction_view,
    build_ec_reaction_views,
    build_forward_sample,
    build_mlm_reaction,
    build_retro_sample,
    split_reaction,
)


DEFAULT_LENGTH_BUCKETS = (64, 128, 256, 512, 768)


def scaled_mask_probabilities(
    timestep: float,
    weights: list[float],
) -> list[float]:
    """Scale and clip weights while preserving the requested mean probability."""
    if not weights:
        return []
    timestep = min(max(float(timestep), 0.0), 1.0)
    if timestep in (0.0, 1.0):
        return [timestep] * len(weights)
    positive = [float(weight) for weight in weights]
    if any(weight <= 0.0 for weight in positive):
        raise ValueError("Masking weights must be positive")
    low, high = 0.0, 1.0 / min(weight for weight in positive if weight > 0.0)
    while sum(min(1.0, high * weight) for weight in positive) / len(positive) < timestep:
        high *= 2.0
    for _ in range(64):
        middle = (low + high) / 2.0
        mean = sum(min(1.0, middle * weight) for weight in positive) / len(positive)
        if mean < timestep:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2.0
    return [min(1.0, scale * weight) for weight in positive]


def choose_length_bucket(length: int, buckets: tuple[int, ...] = DEFAULT_LENGTH_BUCKETS) -> tuple[int, bool]:
    for index, bucket in enumerate(buckets):
        if length <= bucket:
            return index, False
    return len(buckets) - 1, True


@dataclass
class DiffusionPretrainCollator:
    tokenizer: SmilesTokenizer
    task_probs: dict[str, float]
    max_sequence_length: int = 2048
    length_buckets: tuple[int, ...] = DEFAULT_LENGTH_BUCKETS
    center_weight: float = 4.0
    neighbor_weight: float = 2.0
    base_weight: float = 1.0
    weighted_masking: bool = True
    seed: int = 13
    mlm_use_mapped_rxn: bool = True
    timestep_min: float = 1e-3
    ec_views_per_record: int = 1
    include_ec: bool = True

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.mask_token_id = self.tokenizer.ensure_mask_token()
        if not self.length_buckets or tuple(sorted(self.length_buckets)) != self.length_buckets:
            raise ValueError("length_buckets must be a non-empty increasing tuple")
        if self.ec_views_per_record not in (1, 2):
            raise ValueError("ec_views_per_record must be 1 or 2")

    def _truncate_prompt(self, ids: list[int], canvas_length: int = 0) -> tuple[list[int], bool]:
        available = self.max_sequence_length - canvas_length
        if available < 1:
            raise ValueError("Target canvas is larger than max_sequence_length")
        return ids[:available], len(ids) > available

    def _conditional(self, record: dict[str, Any], task: str) -> dict[str, Any]:
        if task == "forward":
            source, target = build_forward_sample(record, self.rng)
        else:
            source, target = build_retro_sample(record, self.rng)
            if "<mask_reactants>" not in source:
                source = f"{source}<mask_reactants>"
        prompt = self.tokenizer.encode(strip_atom_maps(source), add_eos=False)
        target_tokens = self.tokenizer.tokenize(strip_atom_maps(target))
        target_weights = [self.base_weight] * len(target_tokens)
        center_weights_applied = False
        mapped_reaction = record.get("mapped_rxn")
        if self.weighted_masking and isinstance(mapped_reaction, str) and mapped_reaction:
            try:
                mapped_reactants, mapped_products = split_reaction(mapped_reaction)
                mapped_tokens = self.tokenizer.tokenize(mapped_products if task == "forward" else mapped_reactants)
                if strip_atom_maps_from_tokens(mapped_tokens) == target_tokens:
                    target_weights = token_weights_from_center_maps(
                        mapped_tokens,
                        record,
                        base=self.base_weight,
                        center=self.center_weight,
                        neighbor=self.neighbor_weight,
                    )
                    center_weights_applied = any(
                        not math.isclose(weight, self.base_weight) for weight in target_weights
                    )
            except ValueError:
                pass
        target_ids = [self.tokenizer.token_id(token) for token in target_tokens] + [self.tokenizer.eos_token_id]
        target_weights = target_weights + [self.base_weight]
        bucket_index, truncated = choose_length_bucket(len(target_ids), self.length_buckets)
        bucket_length = self.length_buckets[bucket_index]
        if truncated:
            target_ids = target_ids[:bucket_length]
            target_ids[-1] = self.tokenizer.eos_token_id
            target_weights = target_weights[:bucket_length]
            target_weights[-1] = self.base_weight
        canvas = target_ids + [self.tokenizer.eos_token_id] * (bucket_length - len(target_ids))
        canvas_weights = target_weights + [self.base_weight] * (bucket_length - len(target_weights))
        prompt, prompt_truncated = self._truncate_prompt(prompt, bucket_length)
        timestep = self.rng.uniform(self.timestep_min, 1.0)
        probabilities = (
            scaled_mask_probabilities(timestep, canvas_weights)
            if self.weighted_masking
            else [timestep] * bucket_length
        )
        return {
            "task": task,
            "prompt": prompt,
            "clean": prompt + canvas,
            "noiseable": [False] * len(prompt) + [True] * bucket_length,
            "labels": [-100] * len(prompt) + canvas,
            "probabilities": [0.0] * len(prompt) + probabilities,
            "weights": [0.0] * len(prompt) + canvas_weights,
            "length_bucket": bucket_index,
            "target_truncated": truncated,
            "prompt_truncated": prompt_truncated,
            "mlm_truncated": False,
            "center_weights_applied": center_weights_applied,
        }

    def _mlm(self, record: dict[str, Any]) -> dict[str, Any]:
        reaction = build_mlm_reaction(record, self.rng, use_mapped_rxn=self.mlm_use_mapped_rxn)
        all_mapped_tokens = self.tokenizer.tokenize(reaction)
        mapped_tokens = all_mapped_tokens[: self.max_sequence_length - 1]
        weights = token_weights_from_center_maps(
            mapped_tokens,
            record,
            base=self.base_weight,
            center=self.center_weight,
            neighbor=self.neighbor_weight,
        )
        tokens = strip_atom_maps_from_tokens(mapped_tokens)
        clean = [self.tokenizer.token_id(token) for token in tokens] + [self.tokenizer.eos_token_id]
        noiseable = [index > 0 for index in range(len(tokens))] + [False]
        timestep = self.rng.uniform(self.timestep_min, 1.0)
        active_weights = [weights[index] for index, active in enumerate(noiseable[:-1]) if active]
        active_probs = (
            scaled_mask_probabilities(timestep, active_weights)
            if self.weighted_masking
            else [timestep] * len(active_weights)
        )
        probabilities: list[float] = []
        active_index = 0
        for active in noiseable:
            if active:
                probabilities.append(active_probs[active_index])
                active_index += 1
            else:
                probabilities.append(0.0)
        return {
            "task": "mlm",
            "prompt": clean[:1],
            "clean": clean,
            "noiseable": noiseable,
            "labels": [token if active else -100 for token, active in zip(clean, noiseable)],
            "probabilities": probabilities,
            "weights": [weight if active else 0.0 for weight, active in zip(weights + [0.0], noiseable)],
            "length_bucket": -100,
            "target_truncated": False,
            "prompt_truncated": False,
            "mlm_truncated": len(all_mapped_tokens) > self.max_sequence_length - 1,
            "center_weights_applied": self.weighted_masking
            and any(
                not math.isclose(weight, self.base_weight)
                for weight, active in zip(weights, noiseable[:-1])
                if active
            ),
        }

    def _weight_class(self, weight: float) -> str:
        if not self.weighted_masking:
            return "base"
        if not math.isclose(self.center_weight, self.base_weight) and math.isclose(weight, self.center_weight):
            return "center"
        if not math.isclose(self.neighbor_weight, self.base_weight) and math.isclose(weight, self.neighbor_weight):
            return "neighbor"
        return "base"

    def _monitoring(self, rows: list[dict[str, Any]], loss_rows: list[list[bool]]) -> dict[str, float | int]:
        stats: dict[str, float | int] = {
            "target_truncated_samples": 0,
            "prompt_truncated_samples": 0,
            "mlm_truncated_samples": 0,
            "center_weighted_samples": 0,
            "noiseable_tokens": 0,
            "masked_tokens": 0,
            "mask_probability_sum": 0.0,
        }
        for task in ("forward", "retro", "mlm"):
            stats[f"task_{task}_samples"] = 0
        for bucket in self.length_buckets:
            stats[f"bucket_{bucket}_samples"] = 0
        for weight_class in ("base", "neighbor", "center"):
            stats[f"weight_{weight_class}_tokens"] = 0
            stats[f"weight_{weight_class}_masked_tokens"] = 0
            stats[f"weight_{weight_class}_probability_sum"] = 0.0

        for row, selected in zip(rows, loss_rows):
            stats[f"task_{row['task']}_samples"] += 1
            if row["length_bucket"] != -100:
                bucket = self.length_buckets[int(row["length_bucket"])]
                stats[f"bucket_{bucket}_samples"] += 1
            stats["target_truncated_samples"] += int(row["target_truncated"])
            stats["prompt_truncated_samples"] += int(row["prompt_truncated"])
            stats["mlm_truncated_samples"] += int(row["mlm_truncated"])
            stats["center_weighted_samples"] += int(row["center_weights_applied"])
            for active, masked, probability, weight in zip(
                row["noiseable"], selected, row["probabilities"], row["weights"]
            ):
                if not active:
                    continue
                weight_class = self._weight_class(float(weight))
                stats["noiseable_tokens"] += 1
                stats["masked_tokens"] += int(masked)
                stats["mask_probability_sum"] += float(probability)
                stats[f"weight_{weight_class}_tokens"] += 1
                stats[f"weight_{weight_class}_masked_tokens"] += int(masked)
                stats[f"weight_{weight_class}_probability_sum"] += float(probability)
        return stats

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for record in records:
            task = _sample_task(self.task_probs, self.rng)
            rows.append(self._mlm(record) if task == "mlm" else self._conditional(record, task))

        noisy_rows: list[list[int]] = []
        loss_rows: list[list[bool]] = []
        for row in rows:
            noisy: list[int] = []
            selected: list[bool] = []
            for token, active, probability in zip(row["clean"], row["noiseable"], row["probabilities"]):
                masked = bool(active and self.rng.random() < probability)
                noisy.append(self.mask_token_id if masked else token)
                selected.append(masked)
            noisy_rows.append(noisy)
            loss_rows.append(selected)

        input_ids, attention_mask = pad_sequences(noisy_rows, self.tokenizer.pad_token_id)
        labels, _ = pad_sequences([row["labels"] for row in rows], -100)
        probabilities, _ = pad_sequences([row["probabilities"] for row in rows], 0.0)
        loss_mask, _ = pad_sequences(loss_rows, False)
        prompt_ids, prompt_attention_mask = pad_sequences([row["prompt"] for row in rows], self.tokenizer.pad_token_id)
        batch = {
            "tasks": [row["task"] for row in rows],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "noise_probabilities": torch.tensor(probabilities, dtype=torch.float32),
            "prompt_input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(prompt_attention_mask, dtype=torch.long),
            "length_bucket_labels": torch.tensor([row["length_bucket"] for row in rows], dtype=torch.long),
            "truncated_targets": sum(bool(row["target_truncated"]) for row in rows),
            "monitoring": self._monitoring(rows, loss_rows),
        }
        if self.include_ec:
            ec_ids: list[list[int]] = []
            ec_pair_ids: list[int] = []
            ec_level_sets: list[dict[str, list[str]]] = []
            for pair_id, record in enumerate(records):
                texts = (
                    [build_ec_reaction_view(record, self.rng)]
                    if self.ec_views_per_record == 1
                    else list(build_ec_reaction_views(record, self.rng))
                )
                levels = record.get("ec_levels") if isinstance(record.get("ec_levels"), dict) else {}
                for text in texts:
                    encoded = self.tokenizer.encode(strip_atom_maps(text), add_eos=True)[: self.max_sequence_length]
                    if encoded:
                        encoded[-1] = self.tokenizer.eos_token_id
                    ec_ids.append(encoded)
                    ec_pair_ids.append(pair_id if self.ec_views_per_record == 2 else len(ec_pair_ids))
                    ec_level_sets.append(
                        {key: list(levels.get(key) or []) for key in ("ec1", "ec2", "ec3", "ec4")}
                    )
            ec_input_ids, ec_attention_mask = pad_sequences(ec_ids, self.tokenizer.pad_token_id)
            batch.update(
                {
                    "ec_input_ids": torch.tensor(ec_input_ids, dtype=torch.long),
                    "ec_attention_mask": torch.tensor(ec_attention_mask, dtype=torch.long),
                    "ec_pair_ids": torch.tensor(ec_pair_ids, dtype=torch.long),
                    "ec_level_sets": ec_level_sets,
                }
            )
        return batch
