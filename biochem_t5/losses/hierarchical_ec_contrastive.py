from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
import torch.nn.functional as F


def _positive_mask_for_level(level_sets: Sequence[dict[str, list[str]]], level: str, pair_ids: torch.Tensor) -> torch.Tensor:
    n = len(level_sets)
    mask = pair_ids.reshape(-1, 1).eq(pair_ids.reshape(1, -1))
    label_indices: dict[str, list[int]] = defaultdict(list)
    for idx, level_set in enumerate(level_sets):
        for label in set(level_set.get(level) or []):
            label_indices[str(label)].append(idx)
    for indices in label_indices.values():
        index = torch.tensor(indices, dtype=torch.long, device=pair_ids.device)
        mask[index[:, None], index[None, :]] = True
    mask.fill_diagonal_(False)
    return mask


def supervised_contrastive_loss(representations: torch.Tensor, positive_mask: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if representations.numel() == 0:
        return representations.new_tensor(0.0)
    reps = F.normalize(representations, dim=-1)
    logits = reps @ reps.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, -1e9)
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1).clamp_min(1e-12)
    log_prob = logits - torch.log(denom).unsqueeze(1)
    positives = positive_mask & ~self_mask
    positive_counts = positives.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return representations.new_tensor(0.0)
    per_anchor = -(log_prob * positives.float()).sum(dim=1) / positive_counts.clamp_min(1)
    return per_anchor[valid].mean()


def hierarchical_ec_contrastive_loss(
    representations: torch.Tensor,
    level_sets: Sequence[dict[str, list[str]]],
    pair_ids: torch.Tensor,
    temperature: float = 0.07,
    level_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = level_weights or {"ec1": 0.1, "ec2": 0.2, "ec3": 0.3, "ec4": 0.4}
    total = representations.new_tensor(0.0)
    weight_sum = 0.0
    for level, weight in weights.items():
        mask = _positive_mask_for_level(level_sets, level, pair_ids)
        total = total + float(weight) * supervised_contrastive_loss(representations, mask, temperature=temperature)
        weight_sum += float(weight)
    return total / max(weight_sum, 1e-12)
