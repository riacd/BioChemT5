from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup

from biochem_t5.benchmark.common import write_json
from biochem_t5.data.smiles_tokenizer import SmilesTokenizer
from biochem_t5.models.factory import (
    conditional_forward,
    enable_gradient_checkpointing,
    load_conditional_model,
    resize_token_embeddings,
    save_conditional_model,
)

from .data import (
    DEFAULT_TASK_TOKEN,
    ForwardPredictionCollator,
    ForwardPredictionDataset,
    extend_tokenizer,
    extension_tokens,
    load_forward_prediction_splits,
    tokenizer_audit,
)


STATE_FILENAME = "forward_prediction_state.pt"


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _distributed_state() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def _init_distributed() -> tuple[bool, int, int, int]:
    distributed, rank, local_rank, world_size = _distributed_state()
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    return distributed, rank, local_rank, world_size


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model  # type: ignore[return-value]


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    directory: Path,
    model: torch.nn.Module,
    tokenizer: SmilesTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: dict[str, Any],
    config: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    unwrapped = _unwrap(model)
    save_conditional_model(unwrapped, directory)
    tokenizer.save(directory / "smiles_vocab.json")
    checkpoint = {
        **state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _rng_state(),
    }
    state_path = directory / STATE_FILENAME
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(state_path)
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_json(directory / "train_metrics.json", state.get("metrics", {}))


def _resolve_resume_checkpoint(train_cfg: dict[str, Any], output_dir: Path) -> Path | None:
    value = train_cfg.get("resume_from")
    if value:
        path = Path(str(value))
        if path.is_file():
            path = path.parent
        if not (path / STATE_FILENAME).is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path
    latest = output_dir / "latest"
    if bool(train_cfg.get("auto_resume", False)) and (latest / STATE_FILENAME).is_file():
        return latest
    return None


def _load_training_state(
    directory: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    checkpoint = torch.load(directory / STATE_FILENAME, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    _restore_rng_state(checkpoint.get("rng", {}))
    return checkpoint


def _resize_and_initialize(
    model: torch.nn.Module,
    tokenizer: SmilesTokenizer,
    old_vocab_size: int,
    task_token: str,
    initialize_from: str,
) -> None:
    if len(tokenizer) == old_vocab_size:
        return
    try:
        resize_token_embeddings(model, len(tokenizer))
    except TypeError:
        resize_token_embeddings(model, len(tokenizer))
    if task_token in tokenizer.token_to_id and initialize_from in tokenizer.token_to_id:
        task_id = tokenizer.token_to_id[task_token]
        source_id = tokenizer.token_to_id[initialize_from]
        if task_id >= old_vocab_size:
            with torch.no_grad():
                model.get_input_embeddings().weight[task_id].copy_(
                    model.get_input_embeddings().weight[source_id]
                )


@torch.no_grad()
def _validation_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, bf16: bool) -> float:
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    token_count = torch.zeros((), dtype=torch.float64, device=device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        tokens = batch["labels"].ne(-100).sum()
        with autocast:
            output = conditional_forward(model, batch)
        loss_sum += output.loss.detach().double() * tokens
        token_count += tokens
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    model.train()
    return float((loss_sum / token_count.clamp_min(1)).item())


def train(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    distributed, rank, local_rank, world_size = _init_distributed()
    train_cfg = config["training"]
    data_cfg = config["data"]
    vocab_cfg = config.get("vocabulary", {})
    output_dir = Path(config["output_dir"])
    seed = int(train_cfg.get("seed", 13))
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    splits, manifest = load_forward_prediction_splits(
        data_cfg["train_csv"], data_cfg["val_csv"], data_cfg["test_csv"]
    )
    pretrained = Path(config["pretrained_checkpoint"])
    task_token = str(vocab_cfg.get("task_token", DEFAULT_TASK_TOKEN))
    initialize_from = str(vocab_cfg.get("initialize_task_from", "<forward>"))
    source_max_length = int(data_cfg.get("source_max_length", 512))
    target_max_length = int(data_cfg.get("target_max_length", 512))
    resume_dir = _resolve_resume_checkpoint(train_cfg, output_dir)
    base_tokenizer = SmilesTokenizer.load(pretrained / "smiles_vocab.json")
    extension_split_names = [str(name) for name in vocab_cfg.get("extend_from_splits", ["train"])]
    unknown_extension_splits = sorted(set(extension_split_names) - set(splits))
    if unknown_extension_splits:
        raise ValueError(f"Unknown vocabulary extension splits: {unknown_extension_splits}")
    if resume_dir is None:
        missing_tokens: set[str] = set()
        for split_name in extension_split_names:
            missing_tokens.update(extension_tokens(base_tokenizer, splits[split_name], task_token))
        added_tokens = [task_token] if task_token in missing_tokens else []
        added_tokens.extend(sorted(missing_tokens - {task_token}))
        tokenizer = extend_tokenizer(base_tokenizer, added_tokens)
    else:
        tokenizer = SmilesTokenizer.load(resume_dir / "smiles_vocab.json")
        added_tokens = [
            token
            for token, token_id in sorted(tokenizer.token_to_id.items(), key=lambda item: item[1])
            if token_id >= len(base_tokenizer)
        ]

    manifest["tokenizer"] = {
        "base_vocab_size": len(base_tokenizer),
        "extended_vocab_size": len(tokenizer),
        "added_tokens": added_tokens,
        "extension_splits": extension_split_names,
        "audit_before_extension": {
            name: tokenizer_audit(base_tokenizer, substrates, source_max_length, target_max_length, task_token)
            for name, substrates in splits.items()
        },
        "audit_after_extension": {
            name: tokenizer_audit(tokenizer, substrates, source_max_length, target_max_length, task_token)
            for name, substrates in splits.items()
        },
    }
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "data_manifest.json", manifest)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    _barrier()

    train_dataset = ForwardPredictionDataset(splits["train"])
    val_dataset = ForwardPredictionDataset(splits["val"])
    if not train_dataset:
        raise ValueError("Training split contains no valid forward-prediction pairs")
    if not val_dataset:
        raise ValueError("Validation split contains no valid forward-prediction pairs")
    collator = ForwardPredictionCollator(
        tokenizer, source_max_length, target_max_length, task_token
    )
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    batch_size = int(train_cfg.get("per_device_batch_size", 16))
    workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("validation_batch_size", batch_size)),
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = load_conditional_model(resume_dir if resume_dir is not None else pretrained)
    if resume_dir is None:
        _resize_and_initialize(model, tokenizer, len(base_tokenizer), task_token, initialize_from)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} != {model.config.vocab_size}")
    if bool(train_cfg.get("gradient_checkpointing", True)):
        enable_gradient_checkpointing(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    max_steps = int(train_cfg.get("max_steps", 20_000))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        num_training_steps=max_steps,
    )
    state: dict[str, Any] = {
        "step": 0,
        "epoch": 0,
        "batch_in_epoch": 0,
        "best_val_loss": float("inf"),
        "bad_validations": 0,
        "metrics": {"validations": []},
    }
    if resume_dir is not None:
        state.update(_load_training_state(resume_dir, optimizer, scheduler))
        for key in ("optimizer", "scheduler", "rng"):
            state.pop(key, None)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )

    gradient_accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    validate_every = int(train_cfg.get("validate_every_steps", 500))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    use_bf16 = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    epoch = int(state["epoch"])
    resume_batch = int(state["batch_in_epoch"])
    optimizer.zero_grad(set_to_none=True)
    stop = False
    while int(state["step"]) < max_steps and not stop:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        accumulated = 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < resume_batch:
                continue
            batch = {key: value.to(device) for key, value in batch.items()}
            accumulated += 1
            sync_now = accumulated == gradient_accumulation or batch_index + 1 == len(train_loader)
            sync_context = contextlib.nullcontext()
            if distributed and not sync_now:
                sync_context = model.no_sync()  # type: ignore[union-attr]
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else contextlib.nullcontext()
            )
            group_start = (batch_index // gradient_accumulation) * gradient_accumulation
            group_size = min(gradient_accumulation, len(train_loader) - group_start)
            with sync_context, autocast:
                loss = conditional_forward(model, batch).loss / group_size
                loss.backward()
            next_batch = batch_index + 1
            state["epoch"] = epoch + int(next_batch == len(train_loader))
            state["batch_in_epoch"] = 0 if next_batch == len(train_loader) else next_batch
            if not sync_now:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            state["step"] = int(state["step"]) + 1

            should_validate = int(state["step"]) % validate_every == 0 or int(state["step"]) >= max_steps
            if should_validate:
                val_loss = _validation_loss(model, val_loader, device, use_bf16)
                improved = val_loss < float(state["best_val_loss"])
                state["metrics"].setdefault("validations", []).append(
                    {"step": int(state["step"]), "val_loss": val_loss, "learning_rate": scheduler.get_last_lr()[0]}
                )
                if improved:
                    state["best_val_loss"] = val_loss
                    state["bad_validations"] = 0
                else:
                    state["bad_validations"] = int(state["bad_validations"]) + 1
                if rank == 0:
                    _save_checkpoint(output_dir / "latest", model, tokenizer, optimizer, scheduler, state, config)
                    if improved:
                        _save_checkpoint(output_dir / "best", model, tokenizer, optimizer, scheduler, state, config)
                    write_json(output_dir / "train_metrics.json", state["metrics"])
                    print(json.dumps(state["metrics"]["validations"][-1], sort_keys=True), flush=True)
                _barrier()
                stop = int(state["bad_validations"]) >= patience
            if int(state["step"]) >= max_steps or stop:
                break
        epoch += 1
        resume_batch = 0

    if rank == 0 and not (output_dir / "latest" / STATE_FILENAME).exists():
        _save_checkpoint(output_dir / "latest", model, tokenizer, optimizer, scheduler, state, config)
        _save_checkpoint(output_dir / "best", model, tokenizer, optimizer, scheduler, state, config)
    _barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BioChemT5 for single-product forward prediction")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
