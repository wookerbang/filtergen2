from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from .scalar_encoder import SpecEncoder
from .waveform_encoder import MultiScaleWaveformEncoder


class Wave2StructureModel(nn.Module):
    """
    Waveform/spec encoder with a structure head for bilevel training.

    Outputs:
      - g_logits: (B, K, M+1) macro gate logits (last index = SKIP)
      - slot_values_raw: (B, K, S_max) unconstrained slot values
    """

    def __init__(
        self,
        *,
        k_max: int,
        macro_vocab_size: int,
        slot_count: int,
        waveform_in_channels: int = 1,
        d_model: int = 512,
        hidden_mult: int = 2,
        dropout: float = 0.1,
        spec_mode: Literal["type_fc", "none"] = "type_fc",
    ) -> None:
        super().__init__()
        self.k_max = int(k_max)
        self.macro_vocab_size = int(macro_vocab_size)
        self.slot_count = int(slot_count)

        self.wave_encoder = MultiScaleWaveformEncoder(d_model=d_model, in_channels=waveform_in_channels, dropout=dropout)
        if spec_mode == "none":
            self.spec_encoder = None
        elif spec_mode == "type_fc":
            self.spec_encoder = SpecEncoder(d_model=d_model, type_vocab_size=4)
        else:
            raise ValueError(f"Unknown spec_mode: {spec_mode}")

        hidden = int(d_model * max(1, hidden_mult))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_head = nn.Linear(hidden, self.k_max * (self.macro_vocab_size + 1))
        self.value_head = nn.Linear(hidden, self.k_max * self.slot_count)
        nn.init.constant_(self.value_head.bias, -22.0)

    def forward(
        self,
        wave: torch.Tensor,
        filter_type: Optional[torch.Tensor] = None,
        fc_hz: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wave_feat = self.wave_encoder(wave)  # (B, L, d)
        pooled = wave_feat.mean(dim=1)
        if self.spec_encoder is not None:
            if filter_type is None or fc_hz is None:
                raise ValueError("Spec encoder enabled but filter_type/fc_hz not provided.")
            log_fc = torch.log10(fc_hz.clamp_min(1e-6))
            spec_vec = self.spec_encoder(filter_type, log_fc)
            pooled = pooled + spec_vec
        h = self.mlp(pooled)
        g_logits = self.gate_head(h).view(-1, self.k_max, self.macro_vocab_size + 1)
        slot_values_raw = self.value_head(h).view(-1, self.k_max, self.slot_count)
        return g_logits, slot_values_raw
