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
        attn_heads: Optional[int] = None,
        gate_skip_bias: float = 0.0,
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

        if attn_heads is None:
            attn_heads = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 1)
        if d_model % attn_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by attn_heads ({attn_heads}).")

        hidden = int(d_model * max(1, hidden_mult))
        self.slot_queries = nn.Parameter(torch.zeros(self.k_max, d_model))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, attn_heads, dropout=dropout, batch_first=True)
        self.cross_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, attn_heads, dropout=dropout, batch_first=True)
        self.self_ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_head = nn.Linear(hidden, self.macro_vocab_size + 1)
        self.value_head = nn.Linear(hidden, self.slot_count)
        if float(gate_skip_bias) != 0.0:
            with torch.no_grad():
                self.gate_head.bias[self.macro_vocab_size] = float(gate_skip_bias)
        nn.init.constant_(self.value_head.bias, -22.0)

    def forward(
        self,
        wave: torch.Tensor,
        filter_type: Optional[torch.Tensor] = None,
        fc_hz: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wave_feat = self.wave_encoder(wave)  # (B, L, d)
        if self.spec_encoder is not None:
            if filter_type is None or fc_hz is None:
                raise ValueError("Spec encoder enabled but filter_type/fc_hz not provided.")
            log_fc = torch.log10(fc_hz.clamp_min(1e-6))
            spec_vec = self.spec_encoder(filter_type, log_fc).to(wave_feat.dtype)
            wave_feat = torch.cat([spec_vec.unsqueeze(1), wave_feat], dim=1)

        batch = wave_feat.size(0)
        slot_q = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        cross_out, _ = self.cross_attn(slot_q, wave_feat, wave_feat, need_weights=False)
        cross_out = self.cross_ln(slot_q + cross_out)
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out, need_weights=False)
        self_out = self.self_ln(cross_out + self_out)
        h = self.mlp(self_out)
        g_logits = self.gate_head(h)
        slot_values_raw = self.value_head(h)
        return g_logits, slot_values_raw
