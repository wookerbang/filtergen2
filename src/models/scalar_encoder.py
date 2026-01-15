from __future__ import annotations

import torch
import torch.nn as nn


class SpecEncoder(nn.Module):
    """
    Encodes filter spec (type + spec scalars) into a single token embedding.
    """

    def __init__(
        self,
        d_model: int = 512,
        type_vocab_size: int = 4,
        spec_dim: int = 1,
        hidden_dim: int = 128,
        use_learnable_token: bool = True,
    ):
        super().__init__()
        self.spec_dim = int(spec_dim)
        self.type_emb = nn.Embedding(type_vocab_size, d_model)
        self.spec_mlp = nn.Sequential(
            nn.Linear(self.spec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.use_learnable_token = use_learnable_token
        self.base_token = nn.Parameter(torch.zeros(d_model)) if use_learnable_token else None

    def forward(self, filter_type_ids: torch.Tensor, spec_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            filter_type_ids: (B,) long tensor (0=lowpass,1=highpass,2=bandpass,3=bandstop)
            spec_values: (B,) or (B, spec_dim) float tensor of spec scalars (e.g., log10 fc)
        Returns:
            (B, d_model) spec token
        """
        if spec_values.ndim == 1:
            spec_values = spec_values.unsqueeze(-1)
        if spec_values.shape[-1] != self.spec_dim:
            raise ValueError(f"spec_values must have dim {self.spec_dim}, got shape={tuple(spec_values.shape)}")
        t = self.type_emb(filter_type_ids)
        spec = self.spec_mlp(spec_values)
        base = self.base_token if self.use_learnable_token else 0.0
        return t + spec + base
