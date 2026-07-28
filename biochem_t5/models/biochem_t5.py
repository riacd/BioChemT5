from __future__ import annotations

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration


class BiochemT5ForPretraining(nn.Module):
    def __init__(self, config: T5Config, projection_dim: int = 128):
        super().__init__()
        self.t5 = T5ForConditionalGeneration(config)
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, projection_dim),
        )

    def forward_seq2seq(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        return self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def encode_reactions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ec_input_ids: torch.Tensor | None = None,
        ec_attention_mask: torch.Tensor | None = None,
        return_ec_representations: bool = False,
        ec_only: bool = False,
    ):
        if ec_only:
            if ec_input_ids is None or ec_attention_mask is None:
                raise ValueError("ec_input_ids and ec_attention_mask are required for EC-only training")
            return self.encode_reactions(ec_input_ids, ec_attention_mask)
        if input_ids is None or attention_mask is None or labels is None:
            raise ValueError("input_ids, attention_mask, and labels are required for seq2seq training")
        seq_out = self.forward_seq2seq(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if not return_ec_representations:
            return seq_out
        if ec_input_ids is None or ec_attention_mask is None:
            raise ValueError("ec_input_ids and ec_attention_mask are required when return_ec_representations=True")
        ec_reps = self.encode_reactions(ec_input_ids, ec_attention_mask)
        return seq_out, ec_reps


def build_t5_config(vocab_size: int, model_cfg: dict) -> T5Config:
    return T5Config(
        vocab_size=vocab_size,
        d_model=int(model_cfg.get("d_model", 256)),
        d_ff=int(model_cfg.get("d_ff", 512)),
        d_kv=int(model_cfg.get("d_kv", 64)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_decoder_layers=int(model_cfg.get("num_decoder_layers", model_cfg.get("num_layers", 4))),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.1)),
        relative_attention_num_buckets=int(model_cfg.get("relative_attention_num_buckets", 32)),
        relative_attention_max_distance=int(model_cfg.get("relative_attention_max_distance", 128)),
        layer_norm_epsilon=float(model_cfg.get("layer_norm_epsilon", 1e-6)),
        feed_forward_proj=str(model_cfg.get("feed_forward_proj", "relu")),
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
