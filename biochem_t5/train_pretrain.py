from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data.collator import PretrainCollator
from .data.dataset import IndexedJsonlReactionDataset, JsonlReactionDataset, ReactionDatasetSubset, iter_texts_for_vocab
from .data.ec_sampler import DistributedBatchSampler, ECBalancedBatchSampler, ECHierarchicalBatchSampler
from .data.smiles_tokenizer import SmilesTokenizer
from .losses.hierarchical_ec_contrastive import hierarchical_ec_contrastive_loss
from .models.biochem_t5 import BiochemT5ForPretraining, build_t5_config

SEQ2SEQ_TASKS = ("forward", "retro", "mlm")


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _requires_ec_training(cfg: dict[str, Any]) -> bool:
    loss_cfg = cfg.get("loss", {})
    ec_cfg = cfg.get("ec", {})
    return (
        float(loss_cfg.get("ec_contrastive", 0.0) or 0.0) > 0.0
        or bool(ec_cfg.get("contrastive", {}).get("enabled", False))
        or bool(ec_cfg.get("text_prediction", {}).get("enabled", False))
    )


def _assert_implemented_experiment(cfg: dict[str, Any]) -> None:
    ec_cfg = cfg.get("ec", {})
    if bool(ec_cfg.get("text_prediction", {}).get("enabled", False)):
        raise NotImplementedError("EC text prediction T5 is not implemented")
    if str(cfg.get("experiment", {}).get("type", "")).lower() == "ec_interface_placeholder":
        raise NotImplementedError("EC interface placeholder is not a trainable experiment")


def _build_or_load_tokenizer(cfg: dict[str, Any]) -> SmilesTokenizer:
    tokenizer_cfg = cfg.get("tokenizer", {})
    path = Path(tokenizer_cfg.get("path", "data/BiochemT5/tokenizer/smiles_wordlevel_final_vocab.json"))
    if path.exists() and not tokenizer_cfg.get("rebuild", False):
        return SmilesTokenizer.load(path)
    texts = iter_texts_for_vocab(cfg["data"]["center_corpus"], max_records=tokenizer_cfg.get("max_records"))
    tokenizer = SmilesTokenizer.build(texts, vocab_size=int(tokenizer_cfg.get("vocab_size", 4096)))
    tokenizer.save(path)
    return tokenizer


def _save_checkpoint(
    model: BiochemT5ForPretraining,
    tokenizer: SmilesTokenizer,
    out_dir: Path,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t5_dir = out_dir / "t5"
    t5_dir.mkdir(parents=True, exist_ok=True)
    model.t5.config.save_pretrained(t5_dir)
    if getattr(model.t5, "generation_config", None) is not None:
        model.t5.generation_config.save_pretrained(t5_dir)
    weights_path = t5_dir / "pytorch_model.bin"
    weights_tmp = weights_path.with_suffix(weights_path.suffix + ".tmp")
    torch.save(model.t5.state_dict(), weights_tmp)
    weights_tmp.replace(weights_path)
    state = {"model": model.state_dict(), "step": step, "metrics": metrics}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    state_path = out_dir / "biochem_t5_pretraining_state.pt"
    state_tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    torch.save(state, state_tmp)
    state_tmp.replace(state_path)
    tokenizer.save(out_dir / "smiles_vocab.json")
    metrics_path = out_dir / "train_metrics.json"
    metrics_tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    metrics_tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_tmp.replace(metrics_path)


def _resolve_resume_checkpoint(train_cfg: dict[str, Any], out_dir: Path) -> Path | None:
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        path = Path(str(resume_from))
        if path.is_dir():
            path = path / "biochem_t5_pretraining_state.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path
    if not bool(train_cfg.get("auto_resume", False)):
        return None
    for path in (
        out_dir / "biochem_t5_pretraining_state.pt",
        out_dir / "latest" / "biochem_t5_pretraining_state.pt",
    ):
        if path.is_file():
            return path
    return None


def _load_checkpoint(
    path: Path,
    model: BiochemT5ForPretraining,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0)), dict(state.get("metrics") or {})


def _distributed_state() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def _init_distributed() -> tuple[bool, int, int, int]:
    is_distributed, rank, local_rank, world_size = _distributed_state()
    if is_distributed and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
    return is_distributed, rank, local_rank, world_size


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _maybe_barrier() -> None:
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        else:
            dist.barrier()


def _distributed_mean(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _distributed_sum(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _distributed_max(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _seq2seq_task_loss_metrics(logits: torch.Tensor, labels: torch.Tensor, tasks: list[str]) -> dict[str, Any]:
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape={tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits/labels shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}")
    if len(tasks) != labels.size(0):
        raise ValueError(f"tasks length must match batch size: tasks={len(tasks)}, batch={labels.size(0)}")

    with torch.no_grad():
        vocab_size = logits.size(-1)
        valid_tokens = labels.ne(-100)
        token_losses = F.cross_entropy(
            logits.detach().float().reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        token_losses = token_losses * valid_tokens

        stats: list[torch.Tensor] = []
        for task in SEQ2SEQ_TASKS:
            row_mask = torch.tensor([item == task for item in tasks], dtype=torch.bool, device=labels.device)
            if torch.any(row_mask):
                loss_sum = token_losses[row_mask].sum(dtype=torch.float64)
                token_count = valid_tokens[row_mask].sum(dtype=torch.float64)
                sample_count = row_mask.sum(dtype=torch.float64)
            else:
                loss_sum = torch.zeros((), dtype=torch.float64, device=labels.device)
                token_count = torch.zeros((), dtype=torch.float64, device=labels.device)
                sample_count = torch.zeros((), dtype=torch.float64, device=labels.device)
            stats.extend((loss_sum, token_count, sample_count))

        packed = torch.stack(stats)
        if dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        metrics: dict[str, Any] = {}
        for idx, task in enumerate(SEQ2SEQ_TASKS):
            loss_sum = float(packed[idx * 3].item())
            token_count = int(packed[idx * 3 + 1].item())
            sample_count = int(packed[idx * 3 + 2].item())
            metrics[f"{task}_loss"] = loss_sum / token_count if token_count > 0 else None
            metrics[f"{task}_tokens"] = token_count
            metrics[f"{task}_samples"] = sample_count
        return metrics


def _has_complete_ec(levels: dict[str, list[str]]) -> bool:
    return all(bool(levels.get(level)) for level in ("ec1", "ec2", "ec3", "ec4"))


def _ec_complete_count(level_sets: list[dict[str, list[str]]]) -> int:
    return sum(1 for levels in level_sets if _has_complete_ec(levels))


def _complete_ec_contrastive_loss(
    representations: torch.Tensor,
    level_sets: list[dict[str, list[str]]],
    pair_ids: torch.Tensor,
    temperature: float,
    level_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    complete_indices = [idx for idx, levels in enumerate(level_sets) if _has_complete_ec(levels)]
    if len(complete_indices) < 2:
        return representations.new_tensor(0.0)
    index_tensor = torch.tensor(complete_indices, dtype=torch.long, device=representations.device)
    return hierarchical_ec_contrastive_loss(
        representations.index_select(0, index_tensor),
        [level_sets[idx] for idx in complete_indices],
        pair_ids.index_select(0, index_tensor),
        temperature=temperature,
        level_weights=level_weights,
    )


def _split_score(split_key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{split_key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _split_dataset_by_template(
    dataset: JsonlReactionDataset | IndexedJsonlReactionDataset,
    val_fraction: float,
    split_seed: int,
) -> tuple[ReactionDatasetSubset, ReactionDatasetSubset, dict[str, Any]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("validation.val_fraction must be in (0, 1)")
    split_keys = list(getattr(dataset, "split_keys", []))
    if len(split_keys) != len(dataset) or not any(split_keys):
        raise ValueError("Dataset index does not contain split_keys; rebuild the indexed dataset before template split")

    template_key_sets = getattr(dataset, "template_key_sets", None)
    if not template_key_sets or len(template_key_sets) != len(dataset):
        template_key_sets = [[key] for key in split_keys]

    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for keys in template_key_sets:
        if not keys:
            continue
        first = keys[0]
        find(first)
        for key in keys[1:]:
            union(first, key)

    component_keys = ["|".join(sorted({find(key) for key in keys})) if keys else split_keys[idx] for idx, keys in enumerate(template_key_sets)]
    component_to_is_val = {key: _split_score(key, split_seed) < val_fraction for key in set(component_keys)}
    if not any(component_to_is_val.values()):
        chosen = min(component_to_is_val, key=lambda key: _split_score(key, split_seed))
        component_to_is_val[chosen] = True
    if all(component_to_is_val.values()):
        chosen = max(component_to_is_val, key=lambda key: _split_score(key, split_seed))
        component_to_is_val[chosen] = False

    train_indices: list[int] = []
    val_indices: list[int] = []
    for idx, component_key in enumerate(component_keys):
        if component_to_is_val[component_key]:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    train_keys = {key for idx in train_indices for key in template_key_sets[idx]}
    val_keys = {key for idx in val_indices for key in template_key_sets[idx]}
    overlap = train_keys.intersection(val_keys)
    if overlap:
        raise RuntimeError(f"Template split leakage detected: {len(overlap)} split keys appear in both train and validation")

    stats = {
        "event": "data_split",
        "split": "template_component_hash",
        "split_seed": split_seed,
        "val_fraction_requested": val_fraction,
        "train_records": len(train_indices),
        "val_records": len(val_indices),
        "train_template_keys": len(train_keys),
        "val_template_keys": len(val_keys),
        "template_components": len(set(component_keys)),
        "overlap_template_keys": len(overlap),
    }
    return ReactionDatasetSubset(dataset, train_indices), ReactionDatasetSubset(dataset, val_indices), stats


def _build_dataset(cfg: dict[str, Any], rank: int) -> JsonlReactionDataset | IndexedJsonlReactionDataset:
    data_cfg = cfg["data"]
    max_records = data_cfg.get("max_records")
    dataset_type = str(data_cfg.get("dataset_type", "indexed" if max_records is None else "memory"))
    validation_enabled = bool(cfg.get("validation", {}).get("enabled", False))
    if dataset_type == "indexed":
        if dist.is_initialized() and rank != 0:
            _maybe_barrier()
        dataset = IndexedJsonlReactionDataset(
            data_cfg["center_corpus"],
            max_records=max_records,
            index_path=data_cfg.get("index_path"),
            rebuild_index=bool(data_cfg.get("rebuild_index", False)),
        )
        if validation_enabled and not any(getattr(dataset, "split_keys", [])):
            if rank == 0:
                dataset = IndexedJsonlReactionDataset(
                    data_cfg["center_corpus"],
                    max_records=max_records,
                    index_path=data_cfg.get("index_path"),
                    rebuild_index=True,
                )
        if dist.is_initialized() and rank == 0:
            _maybe_barrier()
        return dataset
    if dataset_type == "memory":
        return JsonlReactionDataset(data_cfg["center_corpus"], max_records=max_records)
    raise ValueError(f"Unknown data.dataset_type: {dataset_type}")


def _build_loader(
    cfg: dict[str, Any],
    dataset: JsonlReactionDataset | IndexedJsonlReactionDataset | ReactionDatasetSubset,
    collator: PretrainCollator,
    seed: int,
    rank: int,
    world_size: int,
) -> DataLoader:
    sampler_cfg = cfg["train"].get("sampler", {})
    sampler_type = str(sampler_cfg.get("type", "random"))
    sampler_records = getattr(dataset, "ec_level_sets", getattr(dataset, "records", None))

    if sampler_type == "ec_balanced":
        batch_sampler = ECBalancedBatchSampler(
            sampler_records,
            level=str(sampler_cfg.get("level", "ec4")),
            ec_keys_per_batch=int(sampler_cfg.get("ec_keys_per_batch", 64)),
            samples_per_ec=int(sampler_cfg.get("samples_per_ec", 2)),
            batches_per_epoch=sampler_cfg.get("batches_per_epoch"),
            seed=seed,
        )
        if world_size > 1:
            batch_sampler = DistributedBatchSampler(batch_sampler, num_replicas=world_size, rank=rank)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collator,
            pin_memory=bool(cfg["train"].get("pin_memory", torch.cuda.is_available())),
        )
    if sampler_type == "ec_hierarchical":
        batch_sampler = ECHierarchicalBatchSampler(
            sampler_records,
            groups_per_batch=int(sampler_cfg.get("groups_per_batch", 21)),
            relation_names=sampler_cfg.get("relation_names")
            or (
                "same_ec4",
                "same_ec3_diff_ec4",
                "same_ec2_diff_ec3",
                "same_ec1_diff_ec2",
                "diff_ec1",
            ),
            batches_per_epoch=sampler_cfg.get("batches_per_epoch"),
            seed=seed,
            seq2seq_only_samples_per_batch=int(sampler_cfg.get("seq2seq_only_samples_per_batch", 0)),
            max_anchor_attempts=int(sampler_cfg.get("max_anchor_attempts", 1000)),
            max_sample_attempts=int(sampler_cfg.get("max_sample_attempts", 200)),
        )
        if world_size > 1:
            batch_sampler = DistributedBatchSampler(batch_sampler, num_replicas=world_size, rank=rank)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collator,
            pin_memory=bool(cfg["train"].get("pin_memory", torch.cuda.is_available())),
        )
    if sampler_type == "random":
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if world_size > 1 else None
        return DataLoader(
            dataset,
            batch_size=int(cfg["train"].get("batch_size", 8)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collator,
            pin_memory=bool(cfg["train"].get("pin_memory", torch.cuda.is_available())),
        )
    raise ValueError(f"Unknown train.sampler.type: {sampler_type}")


def _build_validation_loader(
    cfg: dict[str, Any],
    dataset: ReactionDatasetSubset,
    collator: PretrainCollator,
    seed: int,
    rank: int,
    world_size: int,
) -> DataLoader:
    validation_cfg = cfg.get("validation", {})
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed) if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=int(validation_cfg.get("batch_size", cfg["train"].get("batch_size", 8))),
        shuffle=False,
        sampler=sampler,
        num_workers=int(validation_cfg.get("num_workers", cfg["train"].get("num_workers", 0))),
        collate_fn=collator,
        pin_memory=bool(cfg["train"].get("pin_memory", torch.cuda.is_available())),
    )


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision in ("no", "none", "fp32"):
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown train.mixed_precision: {precision}")


def _init_wandb(cfg: dict[str, Any], rank: int):
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)) or rank != 0:
        os.environ.setdefault("WANDB_DISABLED", "true")
        return None
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = str(wandb_cfg.get("mode", "offline"))
    if wandb_cfg.get("dir"):
        os.environ["WANDB_DIR"] = str(wandb_cfg["dir"])
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("logging.wandb.enabled is true, but wandb is not installed") from exc
    return wandb.init(
        project=str(wandb_cfg.get("project", "BioChemT5")),
        name=wandb_cfg.get("name"),
        dir=wandb_cfg.get("dir"),
        tags=wandb_cfg.get("tags"),
        config=cfg,
        mode=str(wandb_cfg.get("mode", "offline")),
    )


@torch.no_grad()
def _run_validation(
    model,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    mixed_precision: str,
) -> dict[str, Any]:
    model.eval()
    max_batches = cfg.get("validation", {}).get("max_batches")

    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with _autocast_context(device, mixed_precision):
            seq_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        tokens = labels.ne(-100).sum().item()
        sums += torch.tensor(
            [float(seq_out.loss.detach().cpu()), 1.0, float(input_ids.size(0)), float(tokens)],
            dtype=torch.float64,
            device=device,
        )

    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    model.train()

    batch_count = max(float(sums[1].item()), 1.0)
    return {
        "val_loss": float((sums[0] / batch_count).item()),
        "val_seq2seq_loss": float((sums[0] / batch_count).item()),
        "val_batches": int(sums[1].item()),
        "val_samples": int(sums[2].item()),
        "val_target_tokens": int(sums[3].item()),
    }


def train(config_path: str | Path) -> dict[str, Any]:
    is_distributed, rank, local_rank, world_size = _init_distributed()
    cfg = _load_config(config_path)
    try:
        _assert_implemented_experiment(cfg)
        seed = int(cfg.get("seed", 13))
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)

        wandb_run = _init_wandb(cfg, rank)
        tokenizer = _build_or_load_tokenizer(cfg)
        dataset = _build_dataset(cfg, rank=rank)
        validation_cfg = cfg.get("validation", {})
        val_loader = None
        split_stats: dict[str, Any] | None = None
        if bool(validation_cfg.get("enabled", False)):
            dataset, val_dataset, split_stats = _split_dataset_by_template(
                dataset,
                val_fraction=float(validation_cfg.get("val_fraction", 0.01)),
                split_seed=int(validation_cfg.get("split_seed", seed)),
            )
            if _is_main_process(rank):
                print(json.dumps(split_stats, sort_keys=True), flush=True)

        collator = PretrainCollator(
            tokenizer=tokenizer,
            task_probs=cfg["data"]["task_probs"],
            max_source_length=int(cfg["data"].get("max_source_length", 512)),
            max_target_length=int(cfg["data"].get("max_target_length", 256)),
            mask_fraction=float(cfg["masking"].get("mask_fraction", 0.15)),
            mean_span_len=float(cfg["masking"].get("mean_span_len", 3.0)),
            center_weight=float(cfg["masking"].get("center_weight", 4.0)),
            neighbor_weight=float(cfg["masking"].get("neighbor_weight", 2.0)),
            base_weight=float(cfg["masking"].get("base_weight", 1.0)),
            seed=seed + rank,
            mlm_use_mapped_rxn=bool(cfg["masking"].get("mlm_use_mapped_rxn", True)),
            ec_views_per_record=int(cfg["data"].get("ec_views_per_record", 1)),
            seq2seq_enabled=bool(cfg["train"].get("seq2seq_enabled", True)),
        )
        loader = _build_loader(cfg, dataset, collator, seed=seed, rank=rank, world_size=world_size)
        if bool(validation_cfg.get("enabled", False)):
            val_loader = _build_validation_loader(cfg, val_dataset, collator, seed=seed, rank=rank, world_size=world_size)

        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu"
        )
        model_cfg = cfg.get("model", {})
        raw_model = BiochemT5ForPretraining(
            build_t5_config(len(tokenizer), model_cfg),
            projection_dim=int(model_cfg.get("projection_dim", 128)),
        ).to(device)
        seq2seq_enabled = bool(cfg["train"].get("seq2seq_enabled", True))
        if not seq2seq_enabled:
            for parameter in raw_model.t5.decoder.parameters():
                parameter.requires_grad = False
            raw_model.t5.shared.weight.requires_grad = True
        if bool(cfg["train"].get("gradient_checkpointing", False)):
            try:
                raw_model.t5.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                raw_model.t5.gradient_checkpointing_enable()
            raw_model.t5.config.use_cache = False

        if is_distributed and device.type == "cuda":
            model = DistributedDataParallel(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=bool(cfg["train"].get("ddp_static_graph", False)),
                find_unused_parameters=bool(cfg["train"].get("ddp_find_unused_parameters", True)),
            )
        elif is_distributed:
            model = DistributedDataParallel(
                raw_model,
                static_graph=bool(cfg["train"].get("ddp_static_graph", False)),
                find_unused_parameters=bool(cfg["train"].get("ddp_find_unused_parameters", True)),
            )
        else:
            model = raw_model

        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=float(cfg["train"].get("learning_rate", 3e-4)))
        ec_cfg = cfg.get("ec", {})
        ec_contrastive_cfg = ec_cfg.get("contrastive", {})
        ec_loss_weight = float(cfg.get("loss", {}).get("ec_contrastive", 0.0) or 0.0)
        ec_contrastive_enabled = bool(ec_contrastive_cfg.get("enabled", False)) or ec_loss_weight > 0.0
        if not seq2seq_enabled and not ec_contrastive_enabled:
            raise ValueError("EC-only training requires EC contrastive training to be enabled")
        if ec_contrastive_enabled and ec_loss_weight <= 0.0:
            raise ValueError("loss.ec_contrastive must be positive when EC contrastive training is enabled")
        ec_temperature = float(cfg.get("loss", {}).get("temperature", 0.07))
        ec_level_weights = ec_contrastive_cfg.get("level_weights")
        max_steps = int(cfg["train"].get("max_steps", 100))
        save_every = int(cfg["train"].get("save_every", 0))
        gradient_accumulation_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
        mixed_precision = str(cfg["train"].get("mixed_precision", "no")).lower()
        val_every = int(validation_cfg.get("every_steps", 0)) if bool(validation_cfg.get("enabled", False)) else 0
        out_dir = Path(cfg["train"].get("output_dir", "outputs/BiochemT5/ablation_smoke"))
        metrics: dict[str, Any] = {
            "steps": 0,
            "loss": None,
            "seq2seq_loss": None,
            "ec_loss": None,
            "rank": rank,
            "world_size": world_size,
        }
        if split_stats is not None:
            metrics.update({f"split_{key}": value for key, value in split_stats.items() if key != "event"})

        resume_path = _resolve_resume_checkpoint(cfg["train"], out_dir)
        step = 0
        if resume_path is not None:
            step, resumed_metrics = _load_checkpoint(resume_path, raw_model, optimizer)
            metrics.update(resumed_metrics)
            metrics["steps"] = step
            if _is_main_process(rank):
                print(json.dumps({"event": "resume", "path": str(resume_path), "step": step}, sort_keys=True), flush=True)

        model.train()
        micro_step = 0
        last_log_time = time.time()
        total_samples = int(metrics.get("samples") or 0)
        total_tokens = int(metrics.get("target_tokens") or 0)
        last_log_samples = total_samples
        last_log_tokens = total_tokens
        optimizer.zero_grad(set_to_none=True)

        while step < max_steps:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(step)
            for batch in loader:
                labels = batch["labels"].to(device) if seq2seq_enabled else None
                batch_samples = len(batch["ec_level_sets"])
                batch_tokens = int(labels.ne(-100).sum().item()) if labels is not None else 0
                sync_gradients = (micro_step + 1) % gradient_accumulation_steps == 0
                sync_context = model.no_sync() if is_distributed and not sync_gradients else contextlib.nullcontext()
                with sync_context:
                    with _autocast_context(device, mixed_precision):
                        if ec_contrastive_enabled and seq2seq_enabled:
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            seq_out, ec_representations = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                ec_input_ids=batch["ec_input_ids"].to(device),
                                ec_attention_mask=batch["ec_attention_mask"].to(device),
                                return_ec_representations=True,
                            )
                            ec_loss = _complete_ec_contrastive_loss(
                                ec_representations,
                                batch["ec_level_sets"],
                                batch["ec_pair_ids"].to(device),
                                temperature=ec_temperature,
                                level_weights=ec_level_weights,
                            )
                            loss = seq_out.loss + ec_loss_weight * ec_loss
                        elif ec_contrastive_enabled:
                            seq_out = None
                            ec_representations = model(
                                ec_input_ids=batch["ec_input_ids"].to(device),
                                ec_attention_mask=batch["ec_attention_mask"].to(device),
                                ec_only=True,
                            )
                            ec_loss = _complete_ec_contrastive_loss(
                                ec_representations,
                                batch["ec_level_sets"],
                                batch["ec_pair_ids"].to(device),
                                temperature=ec_temperature,
                                level_weights=ec_level_weights,
                            )
                            loss = ec_loss_weight * ec_loss
                        else:
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            seq_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            ec_loss = None
                            loss = seq_out.loss
                        loss_for_backward = loss / gradient_accumulation_steps
                    loss_for_backward.backward()

                micro_step += 1
                total_samples += batch_samples
                total_tokens += batch_tokens
                if not sync_gradients:
                    continue

                step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    float(cfg["train"].get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                now = time.time()
                elapsed = max(now - last_log_time, 1e-9)
                global_samples = _distributed_sum(float(total_samples - last_log_samples), device)
                global_tokens = _distributed_sum(float(total_tokens - last_log_tokens), device)
                last_log_time = now
                last_log_samples = total_samples
                last_log_tokens = total_tokens

                loss_value = _distributed_mean(float(loss.detach().cpu()), device)
                seq2seq_loss_value = (
                    _distributed_mean(float(seq_out.loss.detach().cpu()), device) if seq_out is not None else None
                )
                ec_loss_value = (
                    _distributed_mean(float(ec_loss.detach().cpu()), device) if ec_loss is not None else None
                )
                grad_norm_value = _distributed_mean(float(grad_norm.detach().cpu()), device)
                metrics = {
                    "steps": step,
                    "loss": loss_value,
                    "seq2seq_loss": seq2seq_loss_value,
                    "ec_loss": ec_loss_value,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm_value,
                    "samples_per_s": global_samples / elapsed,
                    "tokens_per_s": global_tokens / elapsed,
                    "samples": int(_distributed_sum(float(total_samples), device)),
                    "target_tokens": int(_distributed_sum(float(total_tokens), device)),
                    "rank": rank,
                    "world_size": world_size,
                }
                if seq_out is not None and labels is not None:
                    metrics.update(_seq2seq_task_loss_metrics(seq_out.logits, labels, batch["tasks"]))
                if device.type == "cuda":
                    max_allocated = _distributed_max(torch.cuda.max_memory_allocated(device) / (1024**3), device)
                    max_reserved = _distributed_max(torch.cuda.max_memory_reserved(device) / (1024**3), device)
                    metrics["cuda_max_memory_allocated_gb"] = round(max_allocated, 3)
                    metrics["cuda_max_memory_reserved_gb"] = round(max_reserved, 3)
                should_validate = val_loader is not None and val_every > 0 and step % val_every == 0
                if should_validate:
                    metrics.update(_run_validation(model, val_loader, cfg, device, mixed_precision))
                if _is_main_process(rank) and (
                    step % int(cfg["train"].get("log_every", 10)) == 0 or step == 1 or should_validate
                ):
                    print(json.dumps(metrics, sort_keys=True), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)
                if save_every > 0 and step % save_every == 0 and _is_main_process(rank):
                    _save_checkpoint(raw_model, tokenizer, out_dir / "latest", metrics, optimizer=optimizer, step=step)
                if step >= max_steps:
                    break

        if _is_main_process(rank):
            _save_checkpoint(raw_model, tokenizer, out_dir, metrics, optimizer=optimizer, step=step)
            if wandb_run is not None:
                wandb_run.finish()
        _maybe_barrier()
        return metrics
    finally:
        if is_distributed:
            _cleanup_distributed()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
