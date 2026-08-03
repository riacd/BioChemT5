from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from rdkit import Chem

from biochem_t5.benchmark.common import canonicalize_smiles_set
from biochem_t5.benchmark.retrosynthesis.predict import fuse_candidates
from biochem_t5.data.smiles_tokenizer import SmilesTokenizer
from biochem_t5.models.factory import generate_conditional, load_conditional_model

from .data import DEFAULT_TASK_TOKEN, encode_with_limit, forward_source, load_forward_prediction_csv


def randomized_substrate_smiles(substrate: str, count: int, seed: int) -> list[str]:
    if count < 1:
        raise ValueError("augmentation must be at least 1")
    canonical = canonicalize_smiles_set(substrate)
    if canonical is None:
        raise ValueError(f"Invalid substrate SMILES: {substrate}")
    if count == 1:
        return [canonical]

    parts = canonical.split(".")
    generated = [canonical]
    for augmentation_index in range(1, count):
        randomized_parts: list[str] = []
        for part_index, part in enumerate(parts):
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                raise ValueError(f"Invalid canonical substrate fragment: {part}")
            digest = hashlib.sha1(
                f"{seed}:{augmentation_index}:{part_index}:{part}".encode("utf-8")
            ).digest()
            part_seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            if hasattr(Chem, "MolToRandomSmilesVect"):
                randomized = Chem.MolToRandomSmilesVect(
                    mol,
                    1,
                    randomSeed=part_seed,
                    isomericSmiles=True,
                )[0]
            else:
                randomized = Chem.MolToSmiles(
                    mol, canonical=False, doRandom=True, isomericSmiles=True
                )
            randomized_parts.append(randomized)
        order_rng = random.Random(f"{seed}:{augmentation_index}:{canonical}")
        order_rng.shuffle(randomized_parts)
        generated.append(".".join(randomized_parts))
    return generated


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
    task_token: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        encode_with_limit(tokenizer, forward_source(source, task_token), max_length)[0]
        for source in sources
    ]
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[index, : len(row)] = 1
    return input_ids, attention_mask


def _read_completed_substrates(path: Path) -> set[str]:
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
            substrate = payload.get("substrate")
            if not isinstance(substrate, str) or substrate in completed:
                raise ValueError(f"Invalid or duplicate substrate in existing predictions: {substrate!r}")
            completed.add(substrate)
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
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} != {model.config.vocab_size}")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    source_max_length = 512
    task_token = DEFAULT_TASK_TOKEN
    config_path = checkpoint / "resolved_config.yaml"
    if config_path.is_file():
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        source_max_length = int(resolved.get("data", {}).get("source_max_length", source_max_length))
        task_token = str(resolved.get("vocabulary", {}).get("task_token", task_token))

    substrates = sorted(load_forward_prediction_csv(test_csv).substrates)
    completed = _read_completed_substrates(output) if resume else set()
    mode = "a" if resume and output.exists() else "w"
    pending = [substrate for substrate in substrates if substrate not in completed]
    with output.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            substrate_batch = pending[start : start + batch_size]
            augmented: list[list[str]] = []
            for substrate in substrate_batch:
                digest = hashlib.sha1(f"{seed}:{substrate}".encode("utf-8")).digest()
                substrate_seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
                augmented.append(randomized_substrate_smiles(substrate, augmentation, substrate_seed))
            flat_sources = [source for sources in augmented for source in sources]
            input_ids, attention_mask = _encode_sources(
                tokenizer, flat_sources, source_max_length, task_token, target_device
            )
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

            rows_per_substrate = augmentation * num_return_sequences
            for substrate_index, substrate in enumerate(substrate_batch):
                raw_augmentations: list[dict[str, Any]] = []
                candidates_for_fusion: list[list[dict[str, Any]]] = []
                invalid_count = 0
                for augmentation_index, source in enumerate(augmented[substrate_index]):
                    base = substrate_index * rows_per_substrate + augmentation_index * num_return_sequences
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
                    "substrate": substrate,
                    "augmentation": augmentation,
                    "augmentation_inputs": augmented[substrate_index],
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
    parser = argparse.ArgumentParser(description="Generate single-product forward predictions")
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
