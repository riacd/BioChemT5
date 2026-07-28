#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


MATCH_TIERS = (
    "canonical_isomeric",
    "canonical_no_stereo",
    "uncharged_no_stereo",
    "uncharged_no_stereo_drop_small_inorganic",
    "uncharged_no_stereo_drop_small_inorganic_set",
)
EC_REQUIRED_TIERS = set(MATCH_TIERS[-2:])
_UNCHARGER = rdMolStandardize.Uncharger()
_INDEX: dict[str, dict[str, tuple["BenchmarkReference", ...]]] = {}


@dataclass(frozen=True)
class BenchmarkReference:
    split: str
    ec: str
    rxn_text: str
    substrate_smiles: str
    product_smiles: str


def _normalize_ec(value: Any) -> str:
    return str(value or "").strip().removeprefix("EC ")


def _canonical_molecule(smiles: str) -> tuple[str, str, str, bool] | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    isomeric = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    no_stereo_molecule = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(no_stereo_molecule)
    no_stereo = Chem.MolToSmiles(no_stereo_molecule, canonical=True, isomericSmiles=False)
    try:
        uncharged_molecule = _UNCHARGER.uncharge(Chem.Mol(molecule))
    except (RuntimeError, ValueError):
        uncharged_molecule = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(uncharged_molecule)
    uncharged = Chem.MolToSmiles(uncharged_molecule, canonical=True, isomericSmiles=False)
    heavy_atoms = uncharged_molecule.GetNumHeavyAtoms()
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in uncharged_molecule.GetAtoms())
    small_inorganic = heavy_atoms <= 1 or (carbon_atoms == 0 and heavy_atoms <= 5)
    return isomeric, no_stereo, uncharged, small_inorganic


def _side_variants(side: str) -> dict[str, tuple[str, ...]] | None:
    output: dict[str, list[str]] = {tier: [] for tier in MATCH_TIERS}
    for component in str(side).split("."):
        if not component:
            return None
        canonical = _canonical_molecule(component)
        if canonical is None:
            return None
        isomeric, no_stereo, uncharged, small_inorganic = canonical
        output["canonical_isomeric"].append(isomeric)
        output["canonical_no_stereo"].append(no_stereo)
        output["uncharged_no_stereo"].append(uncharged)
        if not small_inorganic:
            output["uncharged_no_stereo_drop_small_inorganic"].append(uncharged)
            output["uncharged_no_stereo_drop_small_inorganic_set"].append(uncharged)
    variants: dict[str, tuple[str, ...]] = {}
    for tier, components in output.items():
        components = sorted(set(components)) if tier.endswith("_set") else sorted(components)
        if components:
            variants[tier] = tuple(components)
    return variants


def reaction_variants(reaction: str) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None:
    compact = "".join(str(reaction).split())
    if ">>" in compact:
        left, right = compact.split(">>", 1)
    else:
        parts = compact.split(">")
        if len(parts) != 3:
            return None
        left, right = parts[0], parts[2]
    if not left or not right:
        return None
    left_variants = _side_variants(left)
    right_variants = _side_variants(right)
    if left_variants is None or right_variants is None:
        return None
    return {
        tier: (left_variants[tier], right_variants[tier])
        for tier in MATCH_TIERS
        if tier in left_variants and tier in right_variants
    }


def _partial_key(substrates: tuple[str, ...], product: str) -> str:
    return f"{'.'.join(substrates)}>>@{product}"


def load_benchmark_index(
    benchmark_files: Iterable[tuple[str, str | Path]],
) -> tuple[dict[str, dict[str, tuple[BenchmarkReference, ...]]], dict[str, Any]]:
    mutable: dict[str, dict[str, set[BenchmarkReference]]] = {tier: defaultdict(set) for tier in MATCH_TIERS}
    stats: dict[str, Any] = {"files": [], "rows": 0, "invalid_rows": 0, "unique_rxn_texts": set()}
    for split, path_value in benchmark_files:
        path = Path(path_value)
        rows = 0
        invalid_rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"EC_NUM", "RXN_TEXT", "substrate_smiles", "product_smiles"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV {path} is missing required columns: {', '.join(sorted(missing))}")
            for row in reader:
                rows += 1
                substrate = str(row.get("substrate_smiles") or "").strip()
                product = str(row.get("product_smiles") or "").strip()
                variants = reaction_variants(f"{substrate}>>{product}")
                if variants is None:
                    invalid_rows += 1
                    continue
                reference = BenchmarkReference(
                    split=split,
                    ec=_normalize_ec(row.get("EC_NUM")),
                    rxn_text=str(row.get("RXN_TEXT") or "").strip(),
                    substrate_smiles=substrate,
                    product_smiles=product,
                )
                stats["unique_rxn_texts"].add(reference.rxn_text)
                for tier, (substrates, products) in variants.items():
                    if len(products) != 1:
                        invalid_rows += 1
                        continue
                    mutable[tier][_partial_key(substrates, products[0])].add(reference)
        stats["rows"] += rows
        stats["invalid_rows"] += invalid_rows
        stats["files"].append({"split": split, "path": str(path), "rows": rows, "invalid_rows": invalid_rows})
    index = {
        tier: {
            key: tuple(sorted(refs, key=lambda item: (item.split, item.ec, item.rxn_text, item.product_smiles)))
            for key, refs in keys.items()
        }
        for tier, keys in mutable.items()
    }
    stats["unique_rxn_texts"] = len(stats["unique_rxn_texts"])
    stats["unique_keys_by_tier"] = {tier: len(keys) for tier, keys in index.items()}
    return index, stats


def match_reaction(
    reaction: str,
    ecs: Iterable[Any],
    index: dict[str, dict[str, tuple[BenchmarkReference, ...]]],
) -> tuple[dict[str, Any] | None, bool]:
    variants = reaction_variants(reaction)
    if variants is None:
        return None, True
    ec_values = {_normalize_ec(value) for value in ecs if _normalize_ec(value)}
    for tier in MATCH_TIERS:
        if tier not in variants:
            continue
        left, right = variants[tier]
        for reversed_direction, substrates, products in ((False, left, right), (True, right, left)):
            references: list[BenchmarkReference] = []
            for product in products:
                references.extend(index[tier].get(_partial_key(substrates, product), ()))
            if tier in EC_REQUIRED_TIERS:
                references = [reference for reference in references if reference.ec in ec_values]
            if references:
                return {
                    "tier": tier,
                    "reversed": reversed_direction,
                    "references": tuple(sorted(set(references), key=lambda item: (item.split, item.ec, item.rxn_text))),
                }, False
    return None, False


def _init_worker(index: dict[str, dict[str, tuple[BenchmarkReference, ...]]]) -> None:
    global _INDEX
    _INDEX = index


def _match_worker(payload: tuple[str, list[Any]]) -> tuple[dict[str, Any] | None, bool]:
    return match_reaction(payload[0], payload[1], _INDEX)


def _parallel_matches(pool, payloads: list[tuple[str, list[Any]]], workers: int):
    if pool is None:
        return [_match_worker(payload) for payload in payloads]
    return pool.map(_match_worker, payloads, chunksize=max(1, len(payloads) // (workers * 4)))


def filter_corpus(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    matches_jsonl: str | Path,
    index: dict[str, dict[str, tuple[BenchmarkReference, ...]]],
    workers: int,
    chunk_size: int,
    progress_every: int,
) -> dict[str, Any]:
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    matches_path = Path(matches_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(output_path.name + ".tmp")
    matches_tmp = matches_path.with_name(matches_path.name + ".tmp")
    counters: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    started = time.time()
    context = mp.get_context("fork")
    pool = context.Pool(workers, initializer=_init_worker, initargs=(index,)) if workers > 1 else None
    if pool is None:
        _init_worker(index)
    try:
        with input_path.open("r", encoding="utf-8") as reader, output_tmp.open("w", encoding="utf-8") as writer, matches_tmp.open("w", encoding="utf-8") as match_writer:
            while True:
                lines: list[str] = []
                records: list[dict[str, Any]] = []
                while len(lines) < chunk_size:
                    line = reader.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    lines.append(line)
                    records.append(json.loads(line))
                if not lines:
                    break
                payloads = [(str(record.get("rxn") or ""), list(record.get("ecs") or [])) for record in records]
                results = _parallel_matches(pool, payloads, workers)
                previous = counters["input_records"]
                for line, record, (match, invalid) in zip(lines, records, results):
                    counters["input_records"] += 1
                    counters["invalid_pretrain_reactions"] += int(invalid)
                    if match is None:
                        writer.write(line if line.endswith("\n") else line + "\n")
                        counters["kept_records"] += 1
                        continue
                    counters["removed_records"] += 1
                    counters[f"tier:{match['tier']}"] += 1
                    counters[f"direction:{'reversed' if match['reversed'] else 'forward'}"] += 1
                    references = match["references"]
                    for split in {reference.split for reference in references}:
                        split_counts[split] += 1
                    match_writer.write(json.dumps({
                        "rxn_id": record.get("rxn_id"),
                        "source_rxn_ids": record.get("source_rxn_ids"),
                        "ecs": record.get("ecs"),
                        "rxn": record.get("rxn"),
                        "match_tier": match["tier"],
                        "reversed": match["reversed"],
                        "benchmark_references": [reference.__dict__ for reference in references],
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                if progress_every and counters["input_records"] // progress_every != previous // progress_every:
                    print(f"[biochem-bench-filter] scanned={counters['input_records']:,} removed={counters['removed_records']:,} elapsed={time.time() - started:.1f}s", flush=True)
        output_tmp.replace(output_path)
        matches_tmp.replace(matches_path)
    except BaseException:
        output_tmp.unlink(missing_ok=True)
        matches_tmp.unlink(missing_ok=True)
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {
        "input_records": counters["input_records"],
        "kept_records": counters["kept_records"],
        "removed_records": counters["removed_records"],
        "invalid_pretrain_reactions": counters["invalid_pretrain_reactions"],
        "removed_by_tier": {tier: counters[f"tier:{tier}"] for tier in MATCH_TIERS},
        "removed_by_direction": {"forward": counters["direction:forward"], "reversed": counters["direction:reversed"]},
        "removed_by_benchmark_split": dict(sorted(split_counts.items())),
        "elapsed_s": time.time() - started,
    }


def audit_corpus(input_jsonl: str | Path, index, workers: int, chunk_size: int, progress_every: int) -> dict[str, Any]:
    path = Path(input_jsonl)
    counters: Counter[str] = Counter()
    started = time.time()
    context = mp.get_context("fork")
    pool = context.Pool(workers, initializer=_init_worker, initargs=(index,)) if workers > 1 else None
    if pool is None:
        _init_worker(index)
    try:
        with path.open("r", encoding="utf-8") as reader:
            while True:
                records = []
                while len(records) < chunk_size:
                    line = reader.readline()
                    if not line:
                        break
                    if line.strip():
                        records.append(json.loads(line))
                if not records:
                    break
                previous = counters["records"]
                results = _parallel_matches(pool, [(str(item.get("rxn") or ""), list(item.get("ecs") or [])) for item in records], workers)
                for match, invalid in results:
                    counters["records"] += 1
                    counters["invalid_reactions"] += int(invalid)
                    if match is not None:
                        counters["overlap_records"] += 1
                        counters[f"tier:{match['tier']}"] += 1
                if progress_every and counters["records"] // progress_every != previous // progress_every:
                    print(f"[biochem-bench-audit] scanned={counters['records']:,} overlaps={counters['overlap_records']:,} elapsed={time.time() - started:.1f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {"records": counters["records"], "invalid_reactions": counters["invalid_reactions"], "overlap_records": counters["overlap_records"], "overlaps_by_tier": {tier: counters[f"tier:{tier}"] for tier in MATCH_TIERS}, "elapsed_s": time.time() - started}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove BioChem Bench val/test structural overlaps from pretraining data.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-matches", required=True)
    parser.add_argument("--out-stats", required=True)
    parser.add_argument("--test-csv", default="benchmark/biochem_bench/test.csv")
    parser.add_argument("--val-csv", default="benchmark/biochem_bench/val.csv")
    parser.add_argument("--workers", type=int, default=max(1, min(16, mp.cpu_count())))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    index, benchmark = load_benchmark_index([("test", args.test_csv), ("val", args.val_csv)])
    corpus = filter_corpus(args.input_jsonl, args.out_jsonl, args.out_matches, index, args.workers, args.chunk_size, args.progress_every)
    _write_json(args.out_stats, {"benchmark": benchmark, "corpus": corpus, "method": {"tiers": MATCH_TIERS, "ec_required_tiers": sorted(EC_REQUIRED_TIERS), "partial_product_match": True, "reverse_match": True}, "elapsed_s": time.time() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
