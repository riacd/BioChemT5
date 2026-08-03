from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

from biochem_t5.losses.diffusion import diffusion_masked_cross_entropy


class LladaConfig(PretrainedConfig):
    model_type = "biochem_llada"

    def __init__(
        self,
        vocab_size: int = 302,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        mask_token_id: int = 301,
        projection_dim: int = 256,
        length_buckets: tuple[int, ...] | list[int] = (64, 128, 256, 512, 768),
        length_loss_weight: float = 0.1,
        forward_token_id: int = 3,
        retro_token_id: int = 4,
        mask_product_token_id: int = 7,
        mask_reactants_token_id: int = 8,
        generation_forbidden_token_ids: list[int] | tuple[int, ...] | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.mask_token_id = mask_token_id
        self.projection_dim = projection_dim
        self.length_buckets = list(length_buckets)
        self.length_loss_weight = length_loss_weight
        self.forward_token_id = forward_token_id
        self.retro_token_id = retro_token_id
        self.mask_product_token_id = mask_product_token_id
        self.mask_reactants_token_id = mask_reactants_token_id
        self.generation_forbidden_token_ids = list(
            generation_forbidden_token_ids
            if generation_forbidden_token_ids is not None
            else [0, 2, *range(3, 109), mask_token_id]
        )
        self.use_cache = False


class LladaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(dtype)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class LladaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.theta = theta

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = query.size(-2)
        if length > self.max_positions:
            raise ValueError("Rotary embedding length exceeds configured maximum")
        positions = torch.arange(length, dtype=torch.float32, device=query.device)
        dimensions = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=query.device)
        inverse = 1.0 / (self.theta ** (dimensions / self.head_dim))
        frequencies = torch.outer(positions, inverse)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().to(dtype=query.dtype)[None, None, :, :]
        sin = embedding.sin().to(dtype=query.dtype)[None, None, :, :]
        return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


class LladaAttention(nn.Module):
    def __init__(self, config: LladaConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = LladaRotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.dropout = config.attention_dropout

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        batch, length, hidden = hidden_states.shape
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        query, key = self.rotary(query, key)
        additive_mask = None
        if attention_mask is not None:
            additive_mask = torch.zeros(
                batch, 1, 1, length, dtype=query.dtype, device=query.device
            ).masked_fill(attention_mask[:, None, None, :].eq(0), torch.finfo(query.dtype).min)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).reshape(batch, length, hidden)
        return self.output(context)


class LladaMLP(nn.Module):
    def __init__(self, config: LladaConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class LladaLayer(nn.Module):
    def __init__(self, config: LladaConfig) -> None:
        super().__init__()
        self.attention_norm = LladaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = LladaAttention(config)
        self.mlp_norm = LladaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LladaMLP(config)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(
            self.attention(self.attention_norm(hidden_states), attention_mask)
        )
        return hidden_states + self.dropout(self.mlp(self.mlp_norm(hidden_states)))


class LladaPreTrainedModel(PreTrainedModel):
    config_class = LladaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LladaLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LladaModel(LladaPreTrainedModel):
    def __init__(self, config: LladaConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LladaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LladaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> BaseModelOutput:
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("Input exceeds max_position_embeddings")
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(layer.__call__, hidden_states, attention_mask)
            else:
                hidden_states = layer(hidden_states, attention_mask)
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))


@dataclass
class DiffusionGenerationOutput:
    sequences: torch.Tensor
    sequence_scores: torch.Tensor
    mask_counts: list[int]
    length_bucket_indices: torch.Tensor

    @property
    def sequences_scores(self) -> torch.Tensor:
        return self.sequence_scores


@dataclass
class LladaMaskedLMOutput(MaskedLMOutput):
    diffusion_loss: torch.Tensor | None = None
    length_loss: torch.Tensor | None = None


class LladaForMaskedLM(LladaPreTrainedModel):
    family = "llada"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LladaConfig) -> None:
        super().__init__(config)
        self.model = LladaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.projection_dim),
        )
        self.length_head = nn.Linear(config.hidden_size, len(config.length_buckets))
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_reactions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.projection(self._mean_pool(hidden, attention_mask))

    def predict_length_bucket(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.length_head(self._mean_pool(hidden, attention_mask))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        noise_probabilities: torch.Tensor | None = None,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        length_bucket_labels: torch.Tensor | None = None,
        ec_input_ids: torch.Tensor | None = None,
        ec_attention_mask: torch.Tensor | None = None,
        return_ec_representations: bool = False,
        ec_only: bool = False,
        **_: object,
    ):
        if ec_only:
            if ec_input_ids is None or ec_attention_mask is None:
                raise ValueError("EC-only mode requires ec_input_ids and ec_attention_mask")
            return self.encode_reactions(ec_input_ids, ec_attention_mask)
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required")
        logits = self.forward_logits(input_ids, attention_mask)
        loss = None
        diffusion_loss = None
        length_loss = None
        if labels is not None:
            if loss_mask is None or noise_probabilities is None:
                raise ValueError("Diffusion training requires loss_mask and noise_probabilities")
            diffusion_loss = diffusion_masked_cross_entropy(logits, labels, loss_mask, noise_probabilities)
            loss = diffusion_loss
            if length_bucket_labels is not None and torch.any(length_bucket_labels.ne(-100)):
                if prompt_input_ids is None or prompt_attention_mask is None:
                    raise ValueError("Length prediction requires prompt-only inputs")
                length_logits = self.predict_length_bucket(prompt_input_ids, prompt_attention_mask)
                length_loss = F.cross_entropy(length_logits, length_bucket_labels, ignore_index=-100)
                loss = loss + self.config.length_loss_weight * length_loss
        output = LladaMaskedLMOutput(
            loss=loss,
            logits=logits,
            diffusion_loss=diffusion_loss,
            length_loss=length_loss,
        )
        if not return_ec_representations:
            return output
        if ec_input_ids is None or ec_attention_mask is None:
            raise ValueError("EC representations require ec_input_ids and ec_attention_mask")
        return output, self.encode_reactions(ec_input_ids, ec_attention_mask)

    @torch.no_grad()
    def generate_diffusion(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        num_steps: int = 64,
        temperature: float = 0.5,
        num_return_sequences: int = 10,
        length_bucket_indices: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        candidate_validator: Callable[[list[int]], bool] | None = None,
        retry_larger_bucket: bool = True,
    ) -> DiffusionGenerationOutput:
        if num_steps < 1 or num_return_sequences < 1:
            raise ValueError("num_steps and num_return_sequences must be positive")
        original_prompt_input_ids = prompt_input_ids
        original_prompt_attention_mask = prompt_attention_mask
        batch = prompt_input_ids.size(0)
        prompt_input_ids = prompt_input_ids.repeat_interleave(num_return_sequences, dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_return_sequences, dim=0)
        if length_bucket_indices is None:
            predicted = self.predict_length_bucket(prompt_input_ids, prompt_attention_mask).argmax(dim=-1)
        else:
            predicted = length_bucket_indices.repeat_interleave(num_return_sequences).to(prompt_input_ids.device)
        lengths = torch.tensor(self.config.length_buckets, device=prompt_input_ids.device).index_select(0, predicted)
        max_canvas = int(lengths.max().item())
        prompt_width = prompt_input_ids.size(1)
        canvas = torch.full(
            (batch * num_return_sequences, max_canvas),
            self.config.mask_token_id,
            dtype=torch.long,
            device=prompt_input_ids.device,
        )
        valid_canvas = torch.arange(max_canvas, device=canvas.device)[None, :] < lengths[:, None]
        canvas = canvas.masked_fill(~valid_canvas, self.config.pad_token_id)
        committed = torch.zeros_like(valid_canvas)
        committed_log_probs = torch.zeros_like(canvas, dtype=torch.float32)
        mask_counts = [int((valid_canvas & ~committed).sum().item())]

        blocked = torch.tensor(
            sorted({
                token_id
                for token_id in self.config.generation_forbidden_token_ids
                if 0 <= token_id < self.config.vocab_size and token_id != self.config.eos_token_id
            }),
            device=canvas.device,
            dtype=torch.long,
        )
        for step in range(1, num_steps + 1):
            combined = torch.cat((prompt_input_ids, canvas), dim=1)
            combined_mask = torch.cat((prompt_attention_mask, valid_canvas.long()), dim=1)
            logits = self.forward_logits(combined, combined_mask)[:, prompt_width:, :]
            logits[..., blocked] = torch.finfo(logits.dtype).min
            if temperature <= 0.0:
                proposed = logits.argmax(dim=-1)
            else:
                scaled = logits.float() / temperature
                probabilities = scaled.softmax(dim=-1)
                proposed = torch.multinomial(
                    probabilities.reshape(-1, probabilities.size(-1)),
                    1,
                    generator=generator,
                ).reshape(canvas.shape)
            proposed = torch.where(committed, canvas, proposed)
            current_log_probs = logits.float().log_softmax(dim=-1).gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
            confidence = current_log_probs.masked_fill(~valid_canvas, float("-inf"))
            fraction = step / num_steps
            new_committed = torch.zeros_like(committed)
            for row in range(canvas.size(0)):
                keep = math.ceil(int(lengths[row].item()) * fraction)
                selected = confidence[row].topk(keep).indices
                new_committed[row, selected] = True
            canvas = torch.where(new_committed, proposed, torch.full_like(canvas, self.config.mask_token_id))
            canvas = canvas.masked_fill(~valid_canvas, self.config.pad_token_id)
            newly_committed = new_committed & ~committed
            committed_log_probs = torch.where(newly_committed, current_log_probs, committed_log_probs)
            committed = new_committed
            mask_counts.append(int((valid_canvas & ~committed).sum().item()))

        scores = []
        for row in range(canvas.size(0)):
            length = int(lengths[row].item())
            tokens = canvas[row, :length]
            eos = torch.nonzero(tokens.eq(self.config.eos_token_id), as_tuple=False)
            score_length = int(eos[0].item()) if eos.numel() else length
            scores.append(
                committed_log_probs[row, :score_length].mean()
                if score_length > 0
                else committed_log_probs.new_tensor(float("-inf"))
            )
        output = DiffusionGenerationOutput(
            sequences=canvas,
            sequence_scores=torch.stack(scores),
            mask_counts=mask_counts,
            length_bucket_indices=predicted,
        )
        if not retry_larger_bucket:
            return output

        retry_samples: list[int] = []
        for sample in range(batch):
            start = sample * num_return_sequences
            end = start + num_return_sequences
            valid = []
            for sequence in output.sequences[start:end]:
                values = sequence.tolist()
                has_eos = self.config.eos_token_id in values
                valid.append(has_eos and (candidate_validator(values) if candidate_validator else True))
            bucket_index = int(output.length_bucket_indices[start].item())
            if not any(valid) and bucket_index + 1 < len(self.config.length_buckets):
                retry_samples.append(sample)
        if not retry_samples:
            return output

        replacements: list[tuple[int, DiffusionGenerationOutput]] = []
        for sample in retry_samples:
            current_index = int(output.length_bucket_indices[sample * num_return_sequences].item())
            retried = self.generate_diffusion(
                original_prompt_input_ids[sample : sample + 1],
                original_prompt_attention_mask[sample : sample + 1],
                num_steps=num_steps,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                length_bucket_indices=torch.tensor([current_index + 1], device=canvas.device),
                generator=generator,
                candidate_validator=candidate_validator,
                retry_larger_bucket=False,
            )
            replacements.append((sample, retried))

        width = max([output.sequences.size(1), *(item.sequences.size(1) for _, item in replacements)])
        if output.sequences.size(1) < width:
            output.sequences = F.pad(
                output.sequences,
                (0, width - output.sequences.size(1)),
                value=self.config.pad_token_id,
            )
        for sample, replacement in replacements:
            if replacement.sequences.size(1) < width:
                replacement.sequences = F.pad(
                    replacement.sequences,
                    (0, width - replacement.sequences.size(1)),
                    value=self.config.pad_token_id,
                )
            start = sample * num_return_sequences
            end = start + num_return_sequences
            output.sequences[start:end] = replacement.sequences
            output.sequence_scores[start:end] = replacement.sequence_scores
            output.length_bucket_indices[start:end] = replacement.length_bucket_indices
        return output


def build_llada_config(vocab_size: int, mask_token_id: int, model_cfg: dict) -> LladaConfig:
    return LladaConfig(
        vocab_size=vocab_size,
        hidden_size=int(model_cfg.get("hidden_size", model_cfg.get("d_model", 768))),
        intermediate_size=int(model_cfg.get("intermediate_size", model_cfg.get("d_ff", 3072))),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", model_cfg.get("num_layers", 24))),
        num_attention_heads=int(model_cfg.get("num_attention_heads", model_cfg.get("num_heads", 12))),
        max_position_embeddings=int(model_cfg.get("max_position_embeddings", 2048)),
        rms_norm_eps=float(model_cfg.get("rms_norm_eps", 1e-6)),
        rope_theta=float(model_cfg.get("rope_theta", 10_000.0)),
        hidden_dropout=float(model_cfg.get("hidden_dropout", model_cfg.get("dropout_rate", 0.0))),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
        mask_token_id=mask_token_id,
        projection_dim=int(model_cfg.get("projection_dim", 256)),
        length_buckets=model_cfg.get("length_buckets", (64, 128, 256, 512, 768)),
        length_loss_weight=float(model_cfg.get("length_loss_weight", 0.1)),
    )
