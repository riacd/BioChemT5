from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import T5EncoderModel


class ECRetrievalModel(nn.Module):
    def __init__(self, encoder: T5EncoderModel, projection_dim: int = 128, seed: int = 13):
        super().__init__()
        self.encoder = encoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projection = nn.Sequential(
                nn.Linear(encoder.config.d_model, encoder.config.d_model),
                nn.Tanh(),
                nn.Linear(encoder.config.d_model, projection_dim),
            )

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, projection_dim: int = 128, seed: int = 13):
        checkpoint = Path(checkpoint)
        encoder = T5EncoderModel.from_pretrained(checkpoint / "t5")
        return cls(encoder, projection_dim=projection_dim, seed=seed)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {"model": self.state_dict(), "encoder_config": self.encoder.config.to_dict()}
