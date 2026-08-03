from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from biochem_t5.data.smiles_tokenizer import SmilesTokenizer

from .biochem_t5 import BiochemT5ForPretraining, build_t5_config
from .llada import LladaForMaskedLM, build_llada_config


MODEL_METADATA_FILENAME = "model_metadata.json"
SUPPORTED_MODEL_FAMILIES = ("t5", "llada")


def configured_model_family(config: dict[str, Any]) -> str:
    family = str(config.get("model", {}).get("family", "t5")).lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model.family: {family}")
    return family


def detect_checkpoint_family(checkpoint: str | Path) -> str:
    checkpoint = Path(checkpoint)
    metadata_path = checkpoint / MODEL_METADATA_FILENAME
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        family = str(metadata.get("family", "")).lower()
        if family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(f"Invalid model family in {metadata_path}: {family!r}")
        return family
    if (checkpoint / "llada").is_dir():
        return "llada"
    if (checkpoint / "t5").is_dir():
        return "t5"
    raise FileNotFoundError(f"No model weights found below checkpoint: {checkpoint}")


def model_metadata(model: nn.Module) -> dict[str, Any]:
    family = str(getattr(model, "family", "llada" if isinstance(model, LladaForMaskedLM) else "t5"))
    config = model.config if isinstance(model, LladaForMaskedLM) else model.t5.config  # type: ignore[attr-defined]
    return {
        "format_version": 1,
        "family": family,
        "architecture": type(model).__name__,
        "vocab_size": int(config.vocab_size),
        "projection_dim": int(
            model.config.projection_dim
            if isinstance(model, LladaForMaskedLM)
            else model.projection[-1].out_features  # type: ignore[attr-defined]
        ),
    }


def write_model_metadata(model: nn.Module, directory: str | Path) -> None:
    path = Path(directory) / MODEL_METADATA_FILENAME
    path.write_text(json.dumps(model_metadata(model), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_pretraining_model(tokenizer: SmilesTokenizer, config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    family = configured_model_family(config)
    if family == "t5":
        return BiochemT5ForPretraining(
            build_t5_config(len(tokenizer), model_cfg),
            projection_dim=int(model_cfg.get("projection_dim", 128)),
        )
    mask_token_id = tokenizer.ensure_mask_token()
    model = LladaForMaskedLM(build_llada_config(len(tokenizer), mask_token_id, model_cfg))
    model.family = "llada"
    return model


def save_pretrained_component(model: nn.Module, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(model, BiochemT5ForPretraining):
        model.t5.save_pretrained(directory / "t5", safe_serialization=False)
    elif isinstance(model, LladaForMaskedLM):
        model.save_pretrained(directory / "llada", safe_serialization=False)
    else:
        raise TypeError(f"Unsupported pretraining model: {type(model).__name__}")
    write_model_metadata(model, directory)


def load_pretraining_model(checkpoint: str | Path, projection_dim: int | None = None) -> nn.Module:
    checkpoint = Path(checkpoint)
    family = detect_checkpoint_family(checkpoint)
    if family == "llada":
        model = LladaForMaskedLM.from_pretrained(checkpoint / "llada")
        model.family = "llada"
        return model

    from transformers import T5ForConditionalGeneration

    t5 = T5ForConditionalGeneration.from_pretrained(checkpoint / "t5")
    state_path = checkpoint / "biochem_t5_pretraining_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True) if state_path.is_file() else None
    if projection_dim is None and state is not None:
        projection_dim = int(state["model"]["projection.2.weight"].shape[0])
    model = BiochemT5ForPretraining(
        t5.config,
        projection_dim=int(projection_dim or 128),
    )
    model.t5 = t5
    if state is not None:
        model.load_state_dict(state["model"])
    return model


def load_conditional_model(checkpoint: str | Path) -> nn.Module:
    checkpoint = Path(checkpoint)
    family = detect_checkpoint_family(checkpoint)
    if family == "llada":
        model = LladaForMaskedLM.from_pretrained(checkpoint / "llada")
        model.family = "llada"
        return model
    from transformers import T5ForConditionalGeneration

    return T5ForConditionalGeneration.from_pretrained(checkpoint / "t5")


def save_conditional_model(model: nn.Module, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(model, LladaForMaskedLM):
        model.save_pretrained(directory / "llada", safe_serialization=False)
        model.family = "llada"
        write_model_metadata(model, directory)
        return
    model.save_pretrained(directory / "t5", safe_serialization=False)  # type: ignore[attr-defined]
    metadata = {
        "format_version": 1,
        "family": "t5",
        "architecture": type(model).__name__,
        "vocab_size": int(model.config.vocab_size),  # type: ignore[attr-defined]
    }
    (directory / MODEL_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resize_token_embeddings(model: nn.Module, size: int):
    return model.resize_token_embeddings(size)  # type: ignore[attr-defined]


def enable_gradient_checkpointing(model: nn.Module) -> None:
    model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
    model.config.use_cache = False  # type: ignore[attr-defined]


def _conditional_diffusion_batch(
    model: LladaForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = input_ids.device
    buckets = list(model.config.length_buckets)
    combined_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    probability_rows: list[torch.Tensor] = []
    loss_rows: list[torch.Tensor] = []
    prompt_rows: list[torch.Tensor] = []
    bucket_labels: list[int] = []
    for row in range(input_ids.size(0)):
        prompt = input_ids[row, attention_mask[row].bool()]
        if (
            prompt.numel()
            and prompt[0].item() == model.config.retro_token_id
            and prompt[-1].item() != model.config.mask_reactants_token_id
        ):
            prompt = torch.cat((prompt, prompt.new_tensor([model.config.mask_reactants_token_id])))
        target = labels[row, labels[row].ne(-100)]
        bucket_index = next((index for index, size in enumerate(buckets) if target.numel() <= size), len(buckets) - 1)
        bucket_size = buckets[bucket_index]
        target = target[:bucket_size]
        if target.numel() == bucket_size and labels[row].ne(-100).sum() > bucket_size:
            target[-1] = model.config.eos_token_id
        canvas = torch.full((bucket_size,), model.config.eos_token_id, dtype=torch.long, device=device)
        canvas[: target.numel()] = target
        max_prompt = model.config.max_position_embeddings - bucket_size
        prompt = prompt[:max_prompt]
        timestep = torch.empty((), device=device).uniform_(1e-3, 1.0)
        selected = torch.rand(bucket_size, device=device) < timestep
        noisy_canvas = canvas.masked_fill(selected, model.config.mask_token_id)
        combined_rows.append(torch.cat((prompt, noisy_canvas)))
        label_rows.append(torch.cat((torch.full_like(prompt, -100), canvas)))
        probability_rows.append(torch.cat((torch.zeros(prompt.numel(), device=device), timestep.expand(bucket_size))))
        loss_rows.append(torch.cat((torch.zeros(prompt.numel(), dtype=torch.bool, device=device), selected)))
        prompt_rows.append(prompt)
        bucket_labels.append(bucket_index)

    def pad(rows: list[torch.Tensor], value: int | float | bool) -> torch.Tensor:
        width = max(row.numel() for row in rows)
        output = rows[0].new_full((len(rows), width), value)
        for index, row in enumerate(rows):
            output[index, : row.numel()] = row
        return output

    combined = pad(combined_rows, model.config.pad_token_id)
    prompts = pad(prompt_rows, model.config.pad_token_id)
    return {
        "input_ids": combined,
        "attention_mask": combined.ne(model.config.pad_token_id).long(),
        "labels": pad(label_rows, -100),
        "loss_mask": pad(loss_rows, False),
        "noise_probabilities": pad(probability_rows, 0.0),
        "prompt_input_ids": prompts,
        "prompt_attention_mask": prompts.ne(model.config.pad_token_id).long(),
        "length_bucket_labels": torch.tensor(bucket_labels, dtype=torch.long, device=device),
    }


def conditional_forward(model: nn.Module, batch: dict[str, torch.Tensor]):
    base_model = getattr(model, "module", model)
    if isinstance(base_model, LladaForMaskedLM):
        return model(**_conditional_diffusion_batch(
            base_model, batch["input_ids"], batch["attention_mask"], batch["labels"]
        ))
    return model(**batch)


def generate_conditional(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs: Any,
):
    diffusion_steps = int(kwargs.pop("diffusion_steps", 64))
    diffusion_temperature = float(kwargs.pop("temperature", 0.5))
    candidate_validator = kwargs.pop("candidate_validator", None)
    if not isinstance(model, LladaForMaskedLM):
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)  # type: ignore[attr-defined]
    rows: list[torch.Tensor] = []
    for row in range(input_ids.size(0)):
        prompt = input_ids[row, attention_mask[row].bool()]
        if (
            prompt.numel()
            and prompt[0].item() == model.config.retro_token_id
            and prompt[-1].item() != model.config.mask_reactants_token_id
        ):
            prompt = torch.cat((prompt, prompt.new_tensor([model.config.mask_reactants_token_id])))
        rows.append(prompt)
    width = max(row.numel() for row in rows)
    input_ids = input_ids.new_full((len(rows), width), model.config.pad_token_id)
    attention_mask = attention_mask.new_zeros((len(rows), width))
    for index, row in enumerate(rows):
        input_ids[index, : row.numel()] = row
        attention_mask[index, : row.numel()] = 1
    return model.generate_diffusion(
        input_ids,
        attention_mask,
        num_steps=diffusion_steps,
        temperature=diffusion_temperature,
        num_return_sequences=int(kwargs.get("num_return_sequences", 10)),
        generator=kwargs.pop("generator", None),
        candidate_validator=candidate_validator,
    )
