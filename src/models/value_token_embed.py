from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


class ValueAwareEmbedding(nn.Module):
    """
    Wraps a base embedding to add value-aware offsets for <VAL_xxx> tokens.
    Keeps the base weight for tying with LM head.
    """

    def __init__(
        self,
        base_embedding: nn.Embedding,
        token_to_value: Mapping[int, float],
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        vocab_size, d_model = base_embedding.weight.shape
        lookup = torch.zeros(vocab_size)
        is_value = torch.zeros(vocab_size, dtype=torch.bool)
        for tid, v in token_to_value.items():
            idx = int(tid)
            if 0 <= idx < vocab_size:
                lookup[idx] = float(v)
                is_value[idx] = True
        self.register_buffer("value_lookup", lookup, persistent=False)
        self.register_buffer("is_value_token", is_value, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_embedding(input_ids)
        mask = self.is_value_token[input_ids]
        if mask.any():
            vals = self.value_lookup[input_ids][mask].float()
            val_vec = self.mlp(torch.log10(vals.unsqueeze(-1).clamp_min(1e-16)))
            val_vec = val_vec.to(base.dtype)
            offset = torch.zeros_like(base)
            offset[mask] = val_vec
            return base + offset
        return base
