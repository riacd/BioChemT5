from __future__ import annotations

import torch
import torch.distributed as dist


class _GatherWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            ctx.rank = 0
            ctx.world_size = 1
            return tensor
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        gathered = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        if ctx.world_size == 1:
            return gradient
        chunks = gradient.chunk(ctx.world_size, dim=0)
        local = chunks[ctx.rank].contiguous()
        dist.all_reduce(local)
        return local


def gather_with_gradient(tensor: torch.Tensor) -> torch.Tensor:
    return _GatherWithGradient.apply(tensor)


def positive_mask(labels: list[str], device: torch.device) -> torch.Tensor:
    encoded: dict[str, int] = {}
    ids = [encoded.setdefault(label, len(encoded)) for label in labels]
    values = torch.tensor(ids, device=device)
    mask = values[:, None].eq(values[None, :])
    mask.fill_diagonal_(False)
    return mask


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: list[str], temperature: float = 0.07
) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.t() / temperature
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits = logits.masked_fill(self_mask, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = positive_mask(labels, embeddings.device)
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    return (-(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / counts.clamp_min(1))[valid]).mean()


def hierarchical_loss(
    embeddings: torch.Tensor,
    labels_by_level: dict[str, list[str]],
    temperature: float = 0.07,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or {"ec1": 0.1, "ec2": 0.2, "ec3": 0.3}
    total = embeddings.sum() * 0.0
    for level, weight in weights.items():
        total = total + weight * supervised_contrastive_loss(embeddings, labels_by_level[level], temperature)
    return total / sum(weights.values())
