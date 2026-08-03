from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import T5EncoderModel

from .factory import detect_checkpoint_family
from .llada import LladaForMaskedLM


class ECRetrievalModel(nn.Module):
    def __init__(self, encoder: nn.Module, projection_dim: int = 128, seed: int = 13):
        super().__init__()
        self.encoder = encoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            hidden_size = int(
                encoder.config.d_model if hasattr(encoder.config, "d_model") else encoder.config.hidden_size
            )
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, projection_dim),
            )
        self.family = "llada" if encoder.__class__.__name__.startswith("Llada") else "t5"

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, projection_dim: int = 128, seed: int = 13):
        checkpoint = Path(checkpoint)
        family = detect_checkpoint_family(checkpoint)
        if family == "t5":
            encoder = T5EncoderModel.from_pretrained(checkpoint / "t5")
            return cls(encoder, projection_dim=projection_dim, seed=seed)
        pretrained = LladaForMaskedLM.from_pretrained(checkpoint / "llada")
        model = cls(pretrained.model, projection_dim=projection_dim, seed=seed)
        if pretrained.config.projection_dim == projection_dim:
            model.projection.load_state_dict(pretrained.projection.state_dict())
        model.family = "llada"
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model": self.state_dict(),
            "encoder_config": self.encoder.config.to_dict(),
            "family": self.family,
        }
