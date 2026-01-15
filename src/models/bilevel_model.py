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
        spec_mode: Literal[
            "type_fc",
            "type_fc_bw",
            "type_fc_bw_ripple",
            "type_fc_bw_ripple_stop",
            "type_fc_bw_ripple_stop_order",
            "type_fmin_fmax",
            "type_fmin_fmax_ripple",
            "type_fmin_fmax_ripple_stop",
            "type_fmin_fmax_ripple_stop_order",
            "none",
        ] = "type_fc",
        attn_heads: Optional[int] = None,
        gate_skip_bias: float = 0.0,
        use_role_queries: bool = False,
        role_input_frac: float = 0.2,
        role_output_frac: float = 0.2,
    ) -> None:
        super().__init__()
        self.k_max = int(k_max)
        self.macro_vocab_size = int(macro_vocab_size)
        self.slot_count = int(slot_count)
        self.use_role_queries = bool(use_role_queries)
        self.spec_mode = str(spec_mode)

        self.wave_encoder = MultiScaleWaveformEncoder(d_model=d_model, in_channels=waveform_in_channels, dropout=dropout)
        spec_mode = str(spec_mode)
        spec_dims = {
            "type_fc": 1,
            "type_fc_bw": 2,
            "type_fc_bw_ripple": 3,
            "type_fc_bw_ripple_stop": 4,
            "type_fc_bw_ripple_stop_order": 5,
            "type_fmin_fmax": 2,
            "type_fmin_fmax_ripple": 3,
            "type_fmin_fmax_ripple_stop": 4,
            "type_fmin_fmax_ripple_stop_order": 5,
        }
        if spec_mode == "none":
            self.spec_encoder = None
        elif spec_mode in spec_dims:
            self.spec_encoder = SpecEncoder(d_model=d_model, type_vocab_size=4, spec_dim=spec_dims[spec_mode])
        else:
            raise ValueError(f"Unknown spec_mode: {spec_mode}")

        if attn_heads is None:
            attn_heads = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 1)
        if d_model % attn_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by attn_heads ({attn_heads}).")

        hidden = int(d_model * max(1, hidden_mult))
        self.slot_queries = nn.Parameter(torch.zeros(self.k_max, d_model))
        nn.init.normal_(self.slot_queries, std=0.02)
        n_in = max(0, int(round(self.k_max * float(role_input_frac))))
        n_out = max(0, int(round(self.k_max * float(role_output_frac))))
        if n_in + n_out > self.k_max:
            n_out = max(0, self.k_max - n_in)
        role_ids = torch.full((self.k_max,), 1, dtype=torch.long)
        if n_in > 0:
            role_ids[:n_in] = 0
        if n_out > 0:
            role_ids[-n_out:] = 2
        self.register_buffer("role_ids", role_ids, persistent=False)
        if self.use_role_queries:
            self.role_embed = nn.Embedding(3, d_model)
            nn.init.zeros_(self.role_embed.weight)
        else:
            self.role_embed = None
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
        f_min_hz: Optional[torch.Tensor] = None,
        f_max_hz: Optional[torch.Tensor] = None,
        bw_frac: Optional[torch.Tensor] = None,
        ripple_db: Optional[torch.Tensor] = None,
        stopband_max_db: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wave_feat = self.wave_encoder(wave)  # (B, L, d)
        if self.spec_encoder is not None:
            if filter_type is None:
                raise ValueError("Spec encoder enabled but filter_type not provided.")
            eps = 1e-6
            if fc_hz is None:
                fc_hz = f_min_hz if f_min_hz is not None else f_max_hz
            if f_min_hz is None:
                f_min_hz = fc_hz
            if f_max_hz is None:
                f_max_hz = fc_hz
            log_fc = torch.log10(fc_hz.clamp_min(eps)) if fc_hz is not None else None
            log_fmin = torch.log10(f_min_hz.clamp_min(eps))
            log_fmax = torch.log10(f_max_hz.clamp_min(eps))

            if bw_frac is None:
                bw_frac = (f_max_hz - f_min_hz) / (fc_hz.clamp_min(eps) if fc_hz is not None else f_max_hz)
            log_bw = torch.log10(bw_frac.clamp_min(1e-4))

            if ripple_db is None:
                ripple_db = torch.zeros_like(log_fmin)
            log_ripple = torch.log10(ripple_db.clamp_min(1e-3))

            if stopband_max_db is None:
                stopband_max_db = torch.zeros_like(log_fmin)
            stop_depth = (-stopband_max_db).clamp_min(1e-3)
            log_stop = torch.log10(stop_depth)

            if order is None:
                order = torch.ones_like(log_fmin)
            log_order = torch.log10(order.clamp_min(1.0))

            if self.spec_mode == "type_fc":
                if log_fc is None:
                    raise ValueError("Spec encoder type_fc requires fc_hz.")
                spec_vals = log_fc
            elif self.spec_mode == "type_fc_bw":
                if log_fc is None:
                    raise ValueError("Spec encoder type_fc_bw requires fc_hz.")
                spec_vals = torch.stack([log_fc, log_bw], dim=-1)
            elif self.spec_mode == "type_fc_bw_ripple":
                if log_fc is None:
                    raise ValueError("Spec encoder type_fc_bw_ripple requires fc_hz.")
                spec_vals = torch.stack([log_fc, log_bw, log_ripple], dim=-1)
            elif self.spec_mode == "type_fc_bw_ripple_stop":
                if log_fc is None:
                    raise ValueError("Spec encoder type_fc_bw_ripple_stop requires fc_hz.")
                spec_vals = torch.stack([log_fc, log_bw, log_ripple, log_stop], dim=-1)
            elif self.spec_mode == "type_fc_bw_ripple_stop_order":
                if log_fc is None:
                    raise ValueError("Spec encoder type_fc_bw_ripple_stop_order requires fc_hz.")
                spec_vals = torch.stack([log_fc, log_bw, log_ripple, log_stop, log_order], dim=-1)
            elif self.spec_mode == "type_fmin_fmax":
                spec_vals = torch.stack([log_fmin, log_fmax], dim=-1)
            elif self.spec_mode == "type_fmin_fmax_ripple":
                spec_vals = torch.stack([log_fmin, log_fmax, log_ripple], dim=-1)
            elif self.spec_mode == "type_fmin_fmax_ripple_stop":
                spec_vals = torch.stack([log_fmin, log_fmax, log_ripple, log_stop], dim=-1)
            elif self.spec_mode == "type_fmin_fmax_ripple_stop_order":
                spec_vals = torch.stack([log_fmin, log_fmax, log_ripple, log_stop, log_order], dim=-1)
            else:
                raise ValueError(f"Unknown spec_mode: {self.spec_mode}")

            spec_vec = self.spec_encoder(filter_type, spec_vals).to(wave_feat.dtype)
            wave_feat = torch.cat([spec_vec.unsqueeze(1), wave_feat], dim=1)

        batch = wave_feat.size(0)
        slot_q = self.slot_queries
        if self.role_embed is not None:
            slot_q = slot_q + self.role_embed(self.role_ids)
        slot_q = slot_q.unsqueeze(0).expand(batch, -1, -1)
        cross_out, _ = self.cross_attn(slot_q, wave_feat, wave_feat, need_weights=False)
        cross_out = self.cross_ln(slot_q + cross_out)
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out, need_weights=False)
        self_out = self.self_ln(cross_out + self_out)
        h = self.mlp(self_out)
        g_logits = self.gate_head(h)
        slot_values_raw = self.value_head(h)
        return g_logits, slot_values_raw

    def query_symmetry_loss(self, *, core_only: bool = False) -> torch.Tensor:
        q = self.slot_queries
        if self.role_embed is not None:
            q = q + self.role_embed(self.role_ids)
        if q.numel() == 0:
            return torch.tensor(0.0, device=q.device)
        q_rev = torch.flip(q, dims=[0])
        diff = q - q_rev
        if core_only:
            core_mask = self.role_ids == 1
            core_mask = core_mask & torch.flip(core_mask, dims=[0])
            diff = diff[core_mask]
            if diff.numel() == 0:
                return torch.tensor(0.0, device=q.device)
        return (diff.pow(2).sum(dim=-1)).mean()
