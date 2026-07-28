#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter, deque
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


REACTION_CENTER_VERSION = "mapped_rxn_environment_v1"


def _open_text(path: str | Path, mode: str = "rt") -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2, sort_keys=True)
        writer.write("\n")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _split_reaction(rxn: str) -> tuple[str, str] | None:
    if ">>" in rxn and rxn.count(">>") == 1:
        left, right = rxn.split(">>")
        return (left, right) if left and right else None
    parts = rxn.split(">")
    if len(parts) == 3 and parts[0] and parts[2]:
        return parts[0], parts[2]
    return None


def _default_center_fields(status: str, source: str = "none") -> dict[str, Any]:
    return {
        "reaction_center_version": REACTION_CENTER_VERSION,
        "reaction_center_source": source,
        "reaction_center_status": status,
        "reaction_center_changed_atom_maps": [],
        "reaction_center_neighbor_atom_maps": [],
        "reaction_center_atom_maps": [],
        "reaction_center_changed_atom_count": 0,
        "reaction_center_neighbor_atom_count": 0,
        "reaction_center_atom_count": 0,
    }


def _init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _bond_signature(bond: Any) -> tuple[Any, ...]:
    return (
        str(bond.GetBondType()),
        bool(bond.GetIsAromatic()),
        bool(bond.GetIsConjugated()),
        str(bond.GetStereo()),
    )


def _atom_base_signature(atom: Any) -> tuple[Any, ...]:
    return (
        int(atom.GetAtomicNum()),
        int(atom.GetIsotope()),
        int(atom.GetFormalCharge()),
        int(atom.GetNumRadicalElectrons()),
        int(atom.GetChiralTag()),
        bool(atom.GetIsAromatic()),
        bool(atom.GetNoImplicit()),
        int(atom.GetDegree()),
        int(atom.GetTotalNumHs()),
        str(atom.GetHybridization()),
    )


def _side_profile(side_smiles: str) -> tuple[str, dict[int, tuple[Any, ...]], dict[int, set[int]]]:
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(side_smiles)
    except Exception:
        mol = None
    if mol is None:
        return "rdkit_failed", {}, {}

    profiles: dict[int, tuple[Any, ...]] = {}
    neighbors: dict[int, set[int]] = {}
    duplicate_maps: set[int] = set()

    for atom in mol.GetAtoms():
        amap = int(atom.GetAtomMapNum())
        if amap <= 0:
            continue
        if amap in profiles:
            duplicate_maps.add(amap)
        mapped_bonds: list[tuple[int, tuple[Any, ...]]] = []
        mapped_neighbors: set[int] = set()
        for neighbor in atom.GetNeighbors():
            nmap = int(neighbor.GetAtomMapNum())
            if nmap <= 0:
                continue
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if bond is None:
                continue
            mapped_bonds.append((nmap, _bond_signature(bond)))
            mapped_neighbors.add(nmap)
        profiles[amap] = (_atom_base_signature(atom), tuple(sorted(mapped_bonds)))
        neighbors[amap] = mapped_neighbors

    if duplicate_maps:
        return "duplicate_atom_map", profiles, neighbors
    if not profiles:
        return "missing_atom_map", profiles, neighbors
    return "ok", profiles, neighbors


def infer_reaction_center(mapped_rxn: str | None) -> dict[str, Any]:
    if not isinstance(mapped_rxn, str) or not mapped_rxn:
        return _default_center_fields("no_mapped_rxn")

    split = _split_reaction(mapped_rxn)
    if split is None:
        return _default_center_fields("invalid_reaction")

    reactants, products = split
    r_status, r_profiles, r_neighbors = _side_profile(reactants)
    p_status, p_profiles, p_neighbors = _side_profile(products)
    if r_status != "ok":
        return _default_center_fields(f"reactant_{r_status}")
    if p_status != "ok":
        return _default_center_fields(f"product_{p_status}")

    all_maps = set(r_profiles) | set(p_profiles)
    changed = {
        amap
        for amap in all_maps
        if r_profiles.get(amap) != p_profiles.get(amap)
    }
    neighbor_maps: set[int] = set()
    for amap in changed:
        neighbor_maps.update(r_neighbors.get(amap, set()))
        neighbor_maps.update(p_neighbors.get(amap, set()))
    neighbor_maps.difference_update(changed)
    center_maps = changed | neighbor_maps
    status = "ok" if changed else "no_changed_atoms"

    return {
        "reaction_center_version": REACTION_CENTER_VERSION,
        "reaction_center_source": "mapped_rxn_environment",
        "reaction_center_status": status,
        "reaction_center_changed_atom_maps": sorted(changed),
        "reaction_center_neighbor_atom_maps": sorted(neighbor_maps),
        "reaction_center_atom_maps": sorted(center_maps),
        "reaction_center_changed_atom_count": len(changed),
        "reaction_center_neighbor_atom_count": len(neighbor_maps),
        "reaction_center_atom_count": len(center_maps),
    }


def _process_center_chunk(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [infer_reaction_center(item.get("mapped_rxn")) for item in items]


def _iter_record_chunks(
    path: str | Path,
    skip_ids: set[int],
    chunk_size: int,
    limit: int | None,
) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    yielded = 0
    with _open_text(path) as reader:
        for line in reader:
            if not line.strip():
                continue
            record = json.loads(line)
            rxn_id = record.get("rxn_id")
            if isinstance(rxn_id, int) and rxn_id in skip_ids:
                continue
            chunk.append(record)
            yielded += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if limit is not None and yielded >= limit:
                break
    if chunk:
        yield chunk


def _minimal_center_inputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"mapped_rxn": record.get("mapped_rxn")} for record in records]


def _load_existing_ids(path: Path) -> set[int]:
    existing: set[int] = set()
    if not path.exists():
        return existing
    with _open_text(path) as reader:
        for line in reader:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rxn_id = record.get("rxn_id")
            if isinstance(rxn_id, int):
                existing.add(rxn_id)
    return existing


def _ensure_ec_fields(record: dict[str, Any]) -> bool:
    filled = False
    ecs = record.get("ecs")
    if ecs is None:
        record["ecs"] = []
        filled = True
    elif not isinstance(ecs, list):
        record["ecs"] = [str(ecs)]
        filled = True
    ec_levels = record.get("ec_levels")
    if not isinstance(ec_levels, dict):
        record["ec_levels"] = {"ec1": [], "ec2": [], "ec3": [], "ec4": []}
        filled = True
    return filled


def _hist_key(value: int) -> str:
    if value <= 20:
        return str(value)
    if value <= 50:
        return "21-50"
    if value <= 100:
        return "51-100"
    return ">100"


def cmd_enrich(args: argparse.Namespace) -> int:
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[int] = set()
    mode = "w"
    if args.resume and out_jsonl.exists():
        existing_ids = _load_existing_ids(out_jsonl)
        mode = "a"
        _progress(f"[resume] existing records={len(existing_ids):,}")

    started = time.time()
    records_written = 0
    records_without_ec_filled = 0
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rsmiles_status_counts: Counter[str] = Counter()
    center_count_histogram: Counter[str] = Counter()
    total_center_atom_maps = 0
    total_changed_atom_maps = 0
    total_neighbor_atom_maps = 0

    record_chunks = _iter_record_chunks(args.input_jsonl, existing_ids, args.chunk_size, args.limit)

    with out_jsonl.open(mode, encoding="utf-8") as writer:
        if args.workers <= 1:
            _init_worker()
            for records in record_chunks:
                center_infos = _process_center_chunk(_minimal_center_inputs(records))
                records_written, records_without_ec_filled = _write_enriched_records(
                    writer,
                    records,
                    center_infos,
                    records_written,
                    records_without_ec_filled,
                    status_counts,
                    source_counts,
                    rsmiles_status_counts,
                    center_count_histogram,
                    args.progress_every,
                    started,
                )
                total_center_atom_maps += sum(int(info["reaction_center_atom_count"]) for info in center_infos)
                total_changed_atom_maps += sum(int(info["reaction_center_changed_atom_count"]) for info in center_infos)
                total_neighbor_atom_maps += sum(int(info["reaction_center_neighbor_atom_count"]) for info in center_infos)
        else:
            with Pool(processes=args.workers, initializer=_init_worker) as pool:
                pending_records: deque[list[dict[str, Any]]] = deque()

                def work_chunks() -> Iterator[list[dict[str, Any]]]:
                    for records in record_chunks:
                        pending_records.append(records)
                        yield _minimal_center_inputs(records)

                work_iter = (
                    center_infos
                    for center_infos in pool.imap(_process_center_chunk, work_chunks(), chunksize=1)
                )
                for center_infos in work_iter:
                    records = pending_records.popleft()
                    records_written, records_without_ec_filled = _write_enriched_records(
                        writer,
                        records,
                        center_infos,
                        records_written,
                        records_without_ec_filled,
                        status_counts,
                        source_counts,
                        rsmiles_status_counts,
                        center_count_histogram,
                        args.progress_every,
                        started,
                    )
                    total_center_atom_maps += sum(int(info["reaction_center_atom_count"]) for info in center_infos)
                    total_changed_atom_maps += sum(int(info["reaction_center_changed_atom_count"]) for info in center_infos)
                    total_neighbor_atom_maps += sum(int(info["reaction_center_neighbor_atom_count"]) for info in center_infos)

    stats = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(out_jsonl),
        "reaction_center_version": REACTION_CENTER_VERSION,
        "records_written_this_run": records_written,
        "skipped_existing_records": len(existing_ids),
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "rsmiles_status_counts": dict(rsmiles_status_counts),
        "center_atom_count_histogram": dict(sorted(center_count_histogram.items())),
        "records_without_ec_filled": records_without_ec_filled,
        "total_center_atom_maps": total_center_atom_maps,
        "total_changed_atom_maps": total_changed_atom_maps,
        "total_neighbor_atom_maps": total_neighbor_atom_maps,
        "elapsed_s": time.time() - started,
    }
    _write_json(args.out_stats, stats)
    _progress(f"[reaction-center done] written={records_written:,} stats={args.out_stats}")
    return 0


def _write_enriched_records(
    writer: TextIO,
    records: list[dict[str, Any]],
    center_infos: list[dict[str, Any]],
    records_written: int,
    records_without_ec_filled: int,
    status_counts: Counter[str],
    source_counts: Counter[str],
    rsmiles_status_counts: Counter[str],
    center_count_histogram: Counter[str],
    progress_every: int,
    started: float,
) -> tuple[int, int]:
    for record, center_info in zip(records, center_infos):
        if _ensure_ec_fields(record):
            records_without_ec_filled += 1
        record.update(center_info)
        writer.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        records_written += 1
        status = str(center_info.get("reaction_center_status"))
        source = str(center_info.get("reaction_center_source"))
        rsmiles_status = str(record.get("rsmiles_status"))
        center_count = int(center_info.get("reaction_center_atom_count") or 0)
        status_counts[status] += 1
        source_counts[source] += 1
        rsmiles_status_counts[rsmiles_status] += 1
        center_count_histogram[_hist_key(center_count)] += 1

    if progress_every and records_written % progress_every < len(records):
        _progress(
            f"[reaction-center] written={records_written:,} "
            f"statuses={dict(status_counts)} elapsed={time.time() - started:.1f}s"
        )
    return records_written, records_without_ec_filled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add reaction-center atom-map fields to R-SMILES corpus.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-stats", required=True)
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return cmd_enrich(args)


if __name__ == "__main__":
    raise SystemExit(main())
