from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch


def exact_topk_euclidean(
    queries: torch.Tensor, library: torch.Tensor, k: int = 4, query_chunk: int = 512, library_chunk: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    if k < 1 or k > len(library):
        raise ValueError("k must be in [1, library size]")
    all_distances: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for start in range(0, len(queries), query_chunk):
        query = queries[start : start + query_chunk].float()
        best_dist = torch.full((len(query), k), torch.inf, device=query.device)
        best_idx = torch.full((len(query), k), -1, dtype=torch.long, device=query.device)
        for library_start in range(0, len(library), library_chunk):
            block = library[library_start : library_start + library_chunk].to(query.device).float()
            distances = torch.cdist(query, block)
            indices = torch.arange(library_start, library_start + len(block), device=query.device).expand(len(query), -1)
            combined_dist = torch.cat([best_dist, distances], dim=1)
            combined_idx = torch.cat([best_idx, indices], dim=1)
            best_dist, positions = combined_dist.topk(k, largest=False, sorted=True)
            best_idx = combined_idx.gather(1, positions)
        all_distances.append(best_dist.cpu())
        all_indices.append(best_idx.cpu())
    return torch.cat(all_distances), torch.cat(all_indices)


def classification_metrics(truth: Sequence[str], prediction: Sequence[str]) -> dict[str, float]:
    labels = sorted(set(truth) | set(prediction))
    support = Counter(truth)
    f1s: list[float] = []
    recalls: list[float] = []
    weighted = 0.0
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        recalls.append(recall)
        weighted += support[label] * f1
    return {
        "accuracy": sum(t == p for t, p in zip(truth, prediction)) / max(len(truth), 1),
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "macro_recall": sum(recalls) / max(len(recalls), 1),
        "weighted_f1": weighted / max(len(truth), 1),
    }


def recall_at_k(truth: Sequence[str], neighbor_labels: Sequence[Sequence[str]]) -> dict[str, float]:
    widths = sorted({1, *(len(row) for row in neighbor_labels)})
    return {
        f"recall_at_{k}": sum(target in row[:k] for target, row in zip(truth, neighbor_labels)) / max(len(truth), 1)
        for k in widths
    }
