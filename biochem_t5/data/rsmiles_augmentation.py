#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import random
import re
import sys
import time
import types
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


ATOM_MAP_RE = re.compile(r":(\d+)]")
DEFAULT_BASE_DIR = (
    "data/Reaction_gen/results/screen_then_dedup_merged/"
    "r2_non_pubchem_plus_r2_incremental_20260513_143420/"
    "stage01_finalbio_dgbyg_api_equivalent_enzymemap_merged"
)
DEFAULT_INPUT_JSONL = (
    f"{DEFAULT_BASE_DIR}/"
    "reactions_finalbio.dedup_by_rxn_ec.dgbyg_api_equivalent."
    "enzymemap_merged.dgr_negative.jsonl"
)
DEFAULT_ASSIGNMENTS = (
    f"{DEFAULT_BASE_DIR}/drfp_template_clusters/"
    "full_sim095_r3_rings_official_complete_combined_fast5_h2002/"
    "final_assignments.sim095_r3_rings_official_complete_combined_fast5.tsv.gz"
)

_RXN_MAPPER: Any = None
_RSMILES: Any = None
_WORKER_CONFIG: dict[str, Any] = {}


def _open_text(path: str | Path, mode: str = "rt") -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8")


def _loads(line: str) -> dict[str, Any]:
    return json.loads(line)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2, sort_keys=True)
        writer.write("\n")


def _safe_text(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _split_reaction(rxn: str) -> tuple[str, str] | None:
    if ">>" in rxn and rxn.count(">>") == 1:
        left, right = rxn.split(">>")
        if left and right:
            return left, right
        return None
    parts = rxn.split(">")
    if len(parts) == 3 and parts[0] and parts[2]:
        return parts[0], parts[2]
    return None


def _to_two_part_reaction(rxn: str) -> str | None:
    split = _split_reaction(rxn)
    if split is None:
        return None
    return f"{split[0]}>>{split[1]}"


def _atom_maps(side: str) -> list[str]:
    return ATOM_MAP_RE.findall(side)


def _mapping_status(mapped_rxn: str) -> str:
    split = _split_reaction(mapped_rxn)
    if split is None:
        return "invalid_reaction"
    reactants, products = split
    rids = _atom_maps(reactants)
    pids = _atom_maps(products)
    if not rids or not pids:
        return "missing_atom_map"
    if len(rids) != len(set(rids)) or len(pids) != len(set(pids)):
        return "duplicate_atom_map"
    if sorted(rids) != sorted(pids):
        return "unbalanced_atom_map"
    return "ok"


def _ec_levels(ecs: list[str]) -> dict[str, list[str]]:
    levels = {"ec1": set(), "ec2": set(), "ec3": set(), "ec4": set()}
    for ec in ecs:
        parts = str(ec).split(".")
        if len(parts) >= 1 and parts[0]:
            levels["ec1"].add(parts[0])
        if len(parts) >= 2:
            levels["ec2"].add(".".join(parts[:2]))
        if len(parts) >= 3:
            levels["ec3"].add(".".join(parts[:3]))
        if len(parts) >= 4:
            levels["ec4"].add(".".join(parts[:4]))
    return {key: sorted(value) for key, value in levels.items()}


def _primary_template_id(record: dict[str, Any]) -> Any:
    template_ids = record.get("template_ids") or []
    if template_ids:
        return template_ids[0]
    template_id = record.get("template_id")
    return template_id


def _normalize_center_record(
    line_no: int,
    record: dict[str, Any],
    center_meta: dict[str, Any],
) -> dict[str, Any]:
    ecs = [str(ec) for ec in (record.get("ecs") or [])]
    out = dict(record)
    out["rxn_id"] = line_no
    out["cluster_id"] = center_meta["cluster_id"]
    out["cluster_size"] = center_meta["cluster_size"]
    out["is_cluster_center"] = True
    out["primary_template_id"] = _primary_template_id(record)
    out["primary_ec"] = ecs[0] if ecs else None
    out["ec_levels"] = _ec_levels(ecs)
    return out


def _iter_jsonl_with_line_no(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with _open_text(path) as reader:
        for line_no, line in enumerate(reader, start=1):
            if line.strip():
                yield line_no, _loads(line)


def _load_cluster_centers(
    assignments_path: str | Path,
    max_centers: int | None = None,
    progress_every: int = 5_000_000,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    started = time.time()
    centers: dict[int, dict[str, Any]] = {}
    rows = 0
    partial = False

    with _open_text(assignments_path) as reader:
        header = reader.readline().rstrip("\n").split("\t")
        try:
            cluster_idx = header.index("cluster_id")
            center_idx = header.index("center_record_id")
        except ValueError as exc:
            raise SystemExit(f"Assignments missing cluster_id/center_record_id: {assignments_path}") from exc

        for line in reader:
            if not line.strip():
                continue
            rows += 1
            parts = line.rstrip("\n").split("\t")
            center_record_id = int(parts[center_idx])
            cluster_id = parts[cluster_idx]
            meta = centers.get(center_record_id)
            if meta is None:
                centers[center_record_id] = {"cluster_id": cluster_id, "cluster_size": 1}
                if max_centers is not None and len(centers) >= max_centers:
                    partial = True
                    break
            else:
                meta["cluster_size"] += 1

            if progress_every and rows % progress_every == 0:
                _progress(
                    f"[assignments] rows={rows:,} centers={len(centers):,} "
                    f"elapsed={time.time() - started:.1f}s"
                )

    stats = {
        "assignment_rows_scanned": rows,
        "centers_loaded": len(centers),
        "cluster_sizes_are_partial": partial,
        "elapsed_s": time.time() - started,
    }
    return centers, stats


def cmd_build_centers(args: argparse.Namespace) -> int:
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    centers, load_stats = _load_cluster_centers(
        args.assignments_tsv_gz,
        max_centers=args.max_centers,
        progress_every=args.progress_every,
    )

    started = time.time()
    written = 0
    missing_after_scan = 0
    with out_jsonl.open("w", encoding="utf-8") as writer:
        for line_no, record in _iter_jsonl_with_line_no(args.input_jsonl):
            meta = centers.pop(line_no, None)
            if meta is None:
                continue
            out = _normalize_center_record(line_no, record, meta)
            writer.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            if args.max_centers is not None and written >= args.max_centers:
                break
            if args.progress_every and written % args.progress_every == 0:
                _progress(
                    f"[centers] written={written:,} line_no={line_no:,} "
                    f"remaining={len(centers):,}"
                )
        missing_after_scan = len(centers)

    stats = {
        "input_jsonl": str(args.input_jsonl),
        "assignments_tsv_gz": str(args.assignments_tsv_gz),
        "out_jsonl": str(out_jsonl),
        "written": written,
        "missing_center_records_after_scan": missing_after_scan,
        "build_elapsed_s": time.time() - started,
        **load_stats,
    }
    _write_json(args.out_stats, stats)
    _progress(f"[build-centers done] written={written:,} stats={args.out_stats}")
    return 0


def _load_existing_ids(path: Path) -> set[int]:
    existing: set[int] = set()
    if not path.exists():
        return existing
    with _open_text(path) as reader:
        for line in reader:
            if not line.strip():
                continue
            try:
                record = _loads(line)
            except json.JSONDecodeError:
                continue
            rxn_id = record.get("rxn_id")
            if isinstance(rxn_id, int):
                existing.add(rxn_id)
    return existing


def _iter_records_for_augment(
    path: str | Path,
    skip_ids: set[int],
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    yielded = 0
    with _open_text(path) as reader:
        for line in reader:
            if not line.strip():
                continue
            record = _loads(line)
            rxn_id = record.get("rxn_id")
            if isinstance(rxn_id, int) and rxn_id in skip_ids:
                continue
            yield record
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _chunks(iterable: Iterable[dict[str, Any]], chunk_size: int) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _load_rsmiles_module(repo_root: str | Path) -> Any:
    if "textdistance" not in sys.modules and importlib.util.find_spec("textdistance") is None:
        levenshtein = types.SimpleNamespace(distance=_levenshtein_distance)
        sys.modules["textdistance"] = types.SimpleNamespace(levenshtein=levenshtein)

    path = Path(repo_root) / "preprocessing" / "get_R-SMILES.py"
    if not path.exists():
        alt_path = Path(repo_root) / "Rsmiles-main" / "preprocessing" / "get_R-SMILES.py"
        path = alt_path if alt_path.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"R-SMILES source not found: {path}")
    spec = importlib.util.spec_from_file_location("rsmiles_get", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load R-SMILES module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _levenshtein_distance(left: Any, right: Any) -> int:
    a = list(left)
    b = list(right)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _init_worker(config: dict[str, Any]) -> None:
    global _RXN_MAPPER, _RSMILES, _WORKER_CONFIG
    if config.get("cpu_only", True):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _WORKER_CONFIG = dict(config)
    _RSMILES = _load_rsmiles_module(config["rsmiles_repo"])

    from rxnmapper import BatchedMapper
    import torch

    torch.set_num_threads(1)
    _RXN_MAPPER = BatchedMapper(
        batch_size=config["map_batch_size"],
        canonicalize=config["canonicalize_rxns"],
        placeholder_for_invalid=">>",
    )


def _strip_token_spaces(value: str) -> str:
    return "".join(value.split())


def _map_reactions(rxns: list[str]) -> list[dict[str, Any]]:
    assert _RXN_MAPPER is not None
    results: list[dict[str, Any]] = []
    for result in _RXN_MAPPER.map_reactions_with_info(rxns):
        if not result:
            results.append({"mapped_rxn": None, "confidence": None, "status": "rxnmapper_failed"})
        else:
            results.append(
                {
                    "mapped_rxn": result.get("mapped_rxn"),
                    "confidence": result.get("confidence"),
                    "status": "ok",
                }
            )
    return results


def _generate_rsmiles_views(
    mapped_rxn: str,
    augmentation: int,
    forward_mode: str,
    seed: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    assert _RSMILES is not None
    mapping_status = _mapping_status(mapped_rxn)
    if mapping_status != "ok":
        return mapping_status, [], {}

    split = _split_reaction(mapped_rxn)
    if split is None:
        return "invalid_reaction", [], {}
    reactants, products = split

    random.seed(seed)
    data = {
        "reactant": reactants,
        "product": products,
        "augmentation": min(int(augmentation), 20),
        "separated": forward_mode == "separated",
    }
    try:
        retro = _RSMILES.get_retro_rsmiles(data)
        random.seed(seed)
        forward = _RSMILES.get_forward_rsmiles(data)
    except Exception as exc:
        return "rsmiles_failed", [], {"error": f"{exc.__class__.__name__}: {_safe_text(exc)}"}

    if retro.get("status") not in (0, "0"):
        return str(retro.get("status")), [], {}
    if forward.get("status") not in (0, "0"):
        return str(forward.get("status")), [], {}

    retro_src = retro.get("src_data") or []
    retro_tgt = retro.get("tgt_data") or []
    forward_src = forward.get("src_data") or []
    forward_tgt = forward.get("tgt_data") or []
    count = min(len(retro_src), len(retro_tgt), len(forward_src), len(forward_tgt), data["augmentation"])
    views: list[dict[str, Any]] = []
    for idx in range(count):
        view = {
            "aug_id": idx,
            "forward_input": _strip_token_spaces(forward_src[idx]),
            "forward_target": _strip_token_spaces(forward_tgt[idx]),
            "retro_input": _strip_token_spaces(retro_src[idx]),
            "retro_target": _strip_token_spaces(retro_tgt[idx]),
        }
        views.append(view)

    meta = {
        "edit_distance_forward": forward.get("edit_distance"),
        "edit_distance_retro": retro.get("edit_distance"),
    }
    if not views:
        return "empty_rsmiles", [], meta
    return "ok", views, meta


def _process_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rxns_for_mapping: list[str] = []
    mapping_positions: list[int] = []
    mapped_by_position: dict[int, dict[str, Any]] = {}

    for idx, record in enumerate(chunk):
        mapped_rxn = record.get("mapped_rxn")
        if isinstance(mapped_rxn, str) and _mapping_status(mapped_rxn) == "ok":
            mapped_by_position[idx] = {
                "mapped_rxn": mapped_rxn,
                "confidence": record.get("rxnmapper_confidence"),
                "status": "ok",
            }
            continue
        rxn = _to_two_part_reaction(str(record.get("rxn", "")))
        if rxn is None:
            mapped_by_position[idx] = {"mapped_rxn": None, "confidence": None, "status": "invalid_reaction"}
            continue
        rxns_for_mapping.append(rxn)
        mapping_positions.append(idx)

    if rxns_for_mapping:
        mapped_results = _map_reactions(rxns_for_mapping)
        for idx, result in zip(mapping_positions, mapped_results):
            mapped_by_position[idx] = result

    out_records: list[dict[str, Any]] = []
    seed_base = int(_WORKER_CONFIG["seed"])
    augmentation = int(_WORKER_CONFIG["augmentation"])
    forward_mode = str(_WORKER_CONFIG["forward_mode"])

    for idx, record in enumerate(chunk):
        out = dict(record)
        mapped = mapped_by_position[idx]
        mapped_rxn = mapped.get("mapped_rxn")
        out["mapped_rxn"] = mapped_rxn
        out["rxnmapper_confidence"] = mapped.get("confidence")
        out["rsmiles_views"] = []
        out["rsmiles_view_count"] = 0
        out["unique_rsmiles_view_count"] = 0

        if not isinstance(mapped_rxn, str) or mapped.get("status") != "ok":
            out["rsmiles_status"] = mapped.get("status", "rxnmapper_failed")
            out_records.append(out)
            continue

        rxn_id = out.get("rxn_id")
        seed = seed_base + int(rxn_id if isinstance(rxn_id, int) else idx)
        status, views, meta = _generate_rsmiles_views(
            mapped_rxn=mapped_rxn,
            augmentation=augmentation,
            forward_mode=forward_mode,
            seed=seed,
        )
        out["rsmiles_status"] = status
        out["rsmiles_views"] = views
        out["rsmiles_view_count"] = len(views)
        out["unique_rsmiles_view_count"] = len(
            {
                (
                    view["forward_input"],
                    view["forward_target"],
                    view["retro_input"],
                    view["retro_target"],
                )
                for view in views
            }
        )
        out.update(meta)
        out_records.append(out)
    return out_records


def cmd_augment(args: argparse.Namespace) -> int:
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.augmentation > 20:
        _progress(f"[augment] clipping augmentation {args.augmentation} to 20")
        args.augmentation = 20

    existing_ids: set[int] = set()
    mode = "w"
    if args.resume and out_jsonl.exists():
        existing_ids = _load_existing_ids(out_jsonl)
        mode = "a"
        _progress(f"[resume] existing records={len(existing_ids):,}")

    records = _iter_records_for_augment(args.input_jsonl, existing_ids, args.limit)
    chunks = _chunks(records, args.chunk_size)
    worker_config = {
        "rsmiles_repo": args.rsmiles_repo,
        "augmentation": args.augmentation,
        "forward_mode": args.forward_mode,
        "seed": args.seed,
        "map_batch_size": args.map_batch_size,
        "canonicalize_rxns": args.canonicalize_rxns,
        "cpu_only": not args.allow_cuda,
    }

    started = time.time()
    written = 0
    status_counts: Counter[str] = Counter()
    view_count_sum = 0
    unique_view_count_sum = 0

    with out_jsonl.open(mode, encoding="utf-8") as writer:
        if args.workers <= 1:
            _init_worker(worker_config)
            iterator = map(_process_chunk, chunks)
            for result_chunk in iterator:
                for record in result_chunk:
                    writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                    status_counts[str(record.get("rsmiles_status"))] += 1
                    view_count_sum += int(record.get("rsmiles_view_count") or 0)
                    unique_view_count_sum += int(record.get("unique_rsmiles_view_count") or 0)
                if args.progress_every and written % args.progress_every < len(result_chunk):
                    _progress(
                        f"[augment] written={written:,} statuses={dict(status_counts)} "
                        f"elapsed={time.time() - started:.1f}s"
                    )
        else:
            with Pool(processes=args.workers, initializer=_init_worker, initargs=(worker_config,)) as pool:
                for result_chunk in pool.imap_unordered(_process_chunk, chunks, chunksize=1):
                    for record in result_chunk:
                        writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        written += 1
                        status_counts[str(record.get("rsmiles_status"))] += 1
                        view_count_sum += int(record.get("rsmiles_view_count") or 0)
                        unique_view_count_sum += int(record.get("unique_rsmiles_view_count") or 0)
                    if args.progress_every and written % args.progress_every < len(result_chunk):
                        _progress(
                            f"[augment] written={written:,} statuses={dict(status_counts)} "
                            f"elapsed={time.time() - started:.1f}s"
                        )

    stats = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(out_jsonl),
        "rsmiles_repo": str(args.rsmiles_repo),
        "augmentation": args.augmentation,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "map_batch_size": args.map_batch_size,
        "records_written_this_run": written,
        "skipped_existing_records": len(existing_ids),
        "status_counts": dict(status_counts),
        "total_rsmiles_views": view_count_sum,
        "total_unique_rsmiles_views": unique_view_count_sum,
        "elapsed_s": time.time() - started,
    }
    _write_json(args.out_stats, stats)
    _progress(f"[augment done] written={written:,} stats={args.out_stats}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cluster-center R-SMILES augmented corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-centers", help="Extract cluster-center records.")
    build.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    build.add_argument("--assignments-tsv-gz", default=DEFAULT_ASSIGNMENTS)
    build.add_argument("--out-jsonl", required=True)
    build.add_argument("--out-stats", required=True)
    build.add_argument("--max-centers", type=int)
    build.add_argument("--progress-every", type=int, default=1_000_000)
    build.set_defaults(func=cmd_build_centers)

    augment = subparsers.add_parser("augment", help="Atom-map and generate R-SMILES views.")
    augment.add_argument("--input-jsonl", required=True)
    augment.add_argument("--out-jsonl", required=True)
    augment.add_argument("--out-stats", required=True)
    augment.add_argument("--rsmiles-repo", default="Rsmiles-main")
    augment.add_argument("--augmentation", type=int, default=20)
    augment.add_argument("--forward-mode", choices=["mixed", "separated"], default="mixed")
    augment.add_argument("--workers", type=int, default=120)
    augment.add_argument("--chunk-size", type=int, default=32)
    augment.add_argument("--map-batch-size", type=int, default=16)
    augment.add_argument("--seed", type=int, default=20260615)
    augment.add_argument("--limit", type=int)
    augment.add_argument("--resume", action="store_true")
    augment.add_argument("--canonicalize-rxns", action="store_true")
    augment.add_argument("--allow-cuda", action="store_true")
    augment.add_argument("--progress-every", type=int, default=1000)
    augment.set_defaults(func=cmd_augment)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
