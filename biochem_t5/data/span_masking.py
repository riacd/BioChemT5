from __future__ import annotations

import math
import random


def token_weights_from_center_maps(tokens: list[str], record: dict, base: float = 1.0, center: float = 4.0, neighbor: float = 2.0) -> list[float]:
    changed = {int(x) for x in record.get("reaction_center_changed_atom_maps") or []}
    neighbors = {int(x) for x in record.get("reaction_center_neighbor_atom_maps") or []}
    weights: list[float] = []
    for token in tokens:
        value = base
        if ":" in token and token.endswith("]"):
            try:
                amap = int(token.rsplit(":", 1)[1].rstrip("]"))
            except ValueError:
                amap = None
            if amap in changed:
                value = center
            elif amap in neighbors:
                value = neighbor
        weights.append(value)
    return weights


def _weighted_choice(indices: list[int], weights: list[float], rng: random.Random) -> int:
    total = sum(max(weights[idx], 0.0) for idx in indices)
    if total <= 0:
        return rng.choice(indices)
    pick = rng.random() * total
    cumulative = 0.0
    for idx in indices:
        cumulative += max(weights[idx], 0.0)
        if cumulative >= pick:
            return idx
    return indices[-1]


def make_t5_span_corruption(
    tokens: list[str],
    token_weights: list[float],
    rng: random.Random,
    mask_fraction: float = 0.15,
    mean_span_len: float = 3.0,
    max_sentinels: int = 100,
) -> tuple[list[str], list[str], dict]:
    if len(tokens) != len(token_weights):
        raise ValueError("tokens and token_weights must have the same length")
    maskable = [idx for idx, token in enumerate(tokens) if not (token.startswith("<") and token.endswith(">"))]
    if not maskable:
        return tokens, ["</s>"], {"masked_token_count": 0, "span_count": 0}

    target_count = max(1, int(math.ceil(len(maskable) * mask_fraction)))
    selected: set[int] = set()
    attempts = 0
    while len(selected) < target_count and attempts < len(tokens) * 10 and len(selected) < len(maskable):
        attempts += 1
        start = _weighted_choice(maskable, token_weights, rng)
        span_len = max(1, int(round(rng.expovariate(1.0 / max(mean_span_len, 1e-6)))))
        for idx in range(start, min(len(tokens), start + span_len)):
            if idx in maskable:
                selected.add(idx)
            if len(selected) >= target_count:
                break

    spans: list[tuple[int, int]] = []
    for idx in sorted(selected):
        if not spans or idx > spans[-1][1]:
            spans.append((idx, idx + 1))
        else:
            spans[-1] = (spans[-1][0], idx + 1)
    spans = spans[:max_sentinels]

    corrupted: list[str] = []
    target: list[str] = []
    cursor = 0
    for sent_idx, (start, end) in enumerate(spans):
        sentinel = f"<extra_id_{sent_idx}>"
        corrupted.extend(tokens[cursor:start])
        corrupted.append(sentinel)
        target.append(sentinel)
        target.extend(tokens[start:end])
        cursor = end
    corrupted.extend(tokens[cursor:])
    target.append("</s>")
    return corrupted, target, {"masked_token_count": sum(end - start for start, end in spans), "span_count": len(spans)}
