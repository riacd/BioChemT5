from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from rdkit import Chem

from biochem_t5.data.smiles_tokenizer import SmilesTokenizer
from biochem_t5.models.factory import generate_conditional, load_conditional_model

from .data import canonicalize_smiles_set, encode_with_limit, load_retrosynthesis_csv


def randomized_product_smiles(product: str, count: int, seed: int) -> list[str]:
    if count < 1:
        raise ValueError("augmentation must be at least 1")
    canonical = canonicalize_smiles_set(product)
    if canonical is None:
        raise ValueError(f"Invalid product SMILES: {product}")
    if count == 1:
        return [canonical]
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"Invalid canonical product SMILES: {canonical}")
    generated: list[str]
    if hasattr(Chem, "MolToRandomSmilesVect"):
        generated = list(
            Chem.MolToRandomSmilesVect(
                mol,
                count - 1,
                randomSeed=int(seed),
                isomericSmiles=True,
            )
        )
    else:
        from rdkit import rdBase

        rdBase.SeedRandomNumberGenerator(int(seed))
        generated = [Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True) for _ in range(count - 1)]
    return [canonical, *generated]


def fuse_candidates(augmentation_candidates: Iterable[Iterable[dict[str, Any]]], top_k: int = 10) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for augmentation_index, candidates in enumerate(augmentation_candidates):
        seen_in_augmentation: set[str] = set()
        for candidate in candidates:
            canonical = candidate.get("canonical")
            if not canonical or canonical in seen_in_augmentation:
                continue
            seen_in_augmentation.add(canonical)
            rank = int(candidate["rank"])
            score = float(candidate.get("sequence_score", float("-inf")))
            item = aggregated.setdefault(
                canonical,
                {
                    "smiles": canonical,
                    "reciprocal_rank": 0.0,
                    "best_rank": rank,
                    "best_sequence_score": score,
                    "augmentation_hits": 0,
                    "ranks": {},
                },
            )
            item["reciprocal_rank"] += 1.0 / (60 + rank)
            item["best_rank"] = min(int(item["best_rank"]), rank)
            item["best_sequence_score"] = max(float(item["best_sequence_score"]), score)
            item["augmentation_hits"] += 1
            item["ranks"][str(augmentation_index)] = rank
    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            -float(item["reciprocal_rank"]),
            int(item["best_rank"]),
            -float(item["best_sequence_score"]),
            str(item["smiles"]),
        ),
    )
    return ranked[:top_k]


def _decode_sequence(tokenizer: SmilesTokenizer, ids: Iterable[int]) -> str:
    tokens: list[str] = []
    for value in ids:
        token_id = int(value)
        if token_id == tokenizer.eos_token_id:
            break
        if token_id == tokenizer.pad_token_id:
            continue
        tokens.append(tokenizer.id_to_token.get(token_id, "<unk>"))
    return "".join(tokens)


def _encode_sources(
    tokenizer: SmilesTokenizer,
    sources: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [encode_with_limit(tokenizer, f"<retro>{source}", max_length)[0] for source in sources]
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[index, : len(row)] = 1
    return input_ids, attention_mask


def _read_completed_products(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid existing JSONL at {path}:{line_number}") from exc
            product = payload.get("product")
            if not isinstance(product, str) or product in completed:
                raise ValueError(f"Invalid or duplicate product in existing predictions: {product!r}")
            completed.add(product)
    return completed


@torch.no_grad()
def predict(
    checkpoint: str | Path,
    test_csv: str | Path,
    output: str | Path,
    *,
    augmentation: int = 1,
    seed: int = 13,
    batch_size: int = 8,
    num_beams: int = 10,
    num_return_sequences: int = 10,
    max_new_tokens: int = 512,
    length_penalty: float = 1.0,
    top_k: int = 10,
    resume: bool = True,
    device: str | None = None,
) -> None:
    checkpoint = Path(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = SmilesTokenizer.load(checkpoint / "smiles_vocab.json")
    model = load_conditional_model(checkpoint)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    source_max_length = 512
    config_path = checkpoint / "resolved_config.yaml"
    if config_path.is_file():
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        source_max_length = int(resolved.get("data", {}).get("source_max_length", source_max_length))

    products = sorted(load_retrosynthesis_csv(test_csv).products)
    completed = _read_completed_products(output) if resume else set()
    mode = "a" if resume and output.exists() else "w"
    pending = [product for product in products if product not in completed]
    with output.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            product_batch = pending[start : start + batch_size]
            augmented = []
            for product in product_batch:
                digest = hashlib.sha1(f"{seed}:{product}".encode("utf-8")).digest()
                product_seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
                augmented.append(randomized_product_smiles(product, augmentation, product_seed))
            flat_sources = [source for sources in augmented for source in sources]
            input_ids, attention_mask = _encode_sources(tokenizer, flat_sources, source_max_length, target_device)
            generation = generate_conditional(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                diffusion_steps=64,
                temperature=0.5,
                candidate_validator=lambda ids: canonicalize_smiles_set(
                    _decode_sequence(tokenizer, ids)
                ) is not None,
            )
            sequences = generation.sequences.detach().cpu()
            if generation.sequences_scores is None:
                sequence_scores = [float("nan")] * len(sequences)
            else:
                sequence_scores = generation.sequences_scores.detach().float().cpu().tolist()

            rows_per_product = augmentation * num_return_sequences
            for product_index, product in enumerate(product_batch):
                raw_augmentations: list[dict[str, Any]] = []
                candidates_for_fusion: list[list[dict[str, Any]]] = []
                invalid_count = 0
                for augmentation_index, source in enumerate(augmented[product_index]):
                    base = product_index * rows_per_product + augmentation_index * num_return_sequences
                    candidates: list[dict[str, Any]] = []
                    for rank in range(1, num_return_sequences + 1):
                        result_index = base + rank - 1
                        decoded = _decode_sequence(tokenizer, sequences[result_index].tolist())
                        canonical = canonicalize_smiles_set(decoded)
                        invalid_count += int(canonical is None)
                        candidates.append(
                            {
                                "rank": rank,
                                "smiles": decoded,
                                "sequence_score": float(sequence_scores[result_index]),
                                "canonical": canonical,
                                "valid": canonical is not None,
                            }
                        )
                    candidates.sort(key=lambda item: float(item["sequence_score"]), reverse=True)
                    for candidate_rank, candidate in enumerate(candidates, start=1):
                        candidate["rank"] = candidate_rank
                    raw_augmentations.append({"input": source, "candidates": candidates})
                    candidates_for_fusion.append(candidates)
                fused = fuse_candidates(candidates_for_fusion, top_k=top_k)
                payload = {
                    "product": product,
                    "augmentation": augmentation,
                    "augmentation_inputs": augmented[product_index],
                    "raw_augmentations": raw_augmentations,
                    "fused_candidates": fused,
                    "statistics": {
                        "raw_candidates": augmentation * num_return_sequences,
                        "invalid_candidates": invalid_count,
                        "valid_candidates": augmentation * num_return_sequences - invalid_count,
                        "unique_fused_candidates": len(fused),
                    },
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrosynthesis predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--augmentation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-return-sequences", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    predict(
        args.checkpoint,
        args.test_csv,
        args.output,
        augmentation=args.augmentation,
        seed=args.seed,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        top_k=args.top_k,
        resume=not args.no_resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
