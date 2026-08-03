from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    noise_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Importance-corrected masked-token CE, normalized per original target."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have matching batch and sequence dimensions")
    if labels.shape != loss_mask.shape or labels.shape != noise_probabilities.shape:
        raise ValueError("labels, loss_mask, and noise_probabilities must have equal shapes")

    predictable = labels.ne(-100)
    selected = predictable & loss_mask.bool()
    safe_labels = labels.masked_fill(~predictable, 0)
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    corrected = token_loss * selected / noise_probabilities.clamp_min(1e-6)
    denominators = predictable.sum(dim=1)
    per_sample = corrected.sum(dim=1) / denominators.clamp_min(1)
    valid = denominators.gt(0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    return per_sample[valid].mean()
