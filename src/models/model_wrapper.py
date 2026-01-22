from __future__ import annotations

from typing import Literal, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from src.data.value_decomp import MDRConfig, torch_compose_value, torch_decompose_mdr

from .hybrid_value_head import HybridValueHead
from .scalar_encoder import SpecEncoder
from .waveform_encoder import MultiScaleWaveformEncoder


class VACTT5(nn.Module):
    """
    Waveform/spec encoder + T5 decoder for VACT generation.
    Defaults to t5-small for快速迭代，可切换 t5-base.
    """

    def __init__(
        self,
        t5_name: str = "t5-small",
        waveform_in_channels: int = 1,
        d_model_override: Optional[int] = None,
        vocab_size: Optional[int] = None,
        spec_mode: Literal["type_fc", "none"] = "type_fc",
        value_loss_weight: float = 1.0,
        mdr_cfg: MDRConfig = MDRConfig(),
        value_token_ids: Optional[Sequence[int]] = None,
        slot_type_token_to_idx: Optional[Mapping[int, int]] = None,
        macro_slot_count: Optional[int] = None,
        macro_value_loss_weight: float = 1.0,
        macro_slot_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        if vocab_size is not None:
            self.t5.resize_token_embeddings(vocab_size)
        d_model = d_model_override or self.t5.config.d_model

        self.wave_encoder = MultiScaleWaveformEncoder(d_model=d_model, in_channels=waveform_in_channels)
        if spec_mode == "none":
            self.spec_encoder = None
        elif spec_mode == "type_fc":
            self.spec_encoder = SpecEncoder(d_model=d_model, type_vocab_size=4)
        else:
            raise ValueError(f"Unknown spec_mode: {spec_mode}")

        self.mdr_cfg = mdr_cfg
        self.value_loss_weight = value_loss_weight
        self.register_buffer("value_token_ids", torch.tensor(value_token_ids, dtype=torch.long) if value_token_ids else None, persistent=False)
        vocab_size = self.t5.get_input_embeddings().weight.shape[0]
        slot_lookup = torch.full((vocab_size,), -1, dtype=torch.long)
        if slot_type_token_to_idx:
            for tok_id, slot_idx in slot_type_token_to_idx.items():
                if 0 <= int(tok_id) < vocab_size:
                    slot_lookup[int(tok_id)] = int(slot_idx)
        self.register_buffer("slot_type_lookup", slot_lookup, persistent=False)

        self.value_head = HybridValueHead(
            d_model,
            num_mantissa=24,
            num_decade=int(mdr_cfg.num_decades),
            residual_log10_scale=float(mdr_cfg.residual_log10_scale),
            num_slot_types=int(torch.max(slot_lookup).item() + 1) if slot_type_token_to_idx else 0,
        )

        self.macro_value_loss_weight = float(macro_value_loss_weight)
        self.macro_value_head = None
        if macro_slot_count is not None and int(macro_slot_count) > 0:
            self.macro_value_head = nn.Linear(d_model, int(macro_slot_count))
        if macro_slot_mask is not None:
            self.register_buffer("macro_slot_mask", macro_slot_mask, persistent=False)
        else:
            self.register_buffer("macro_slot_mask", None, persistent=False)

    def encode(
        self,
        wave: torch.Tensor,
        filter_type: Optional[torch.Tensor] = None,
        fc_hz: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        """
        Args:
            wave: (B, C, L) waveform (freq channels + S21(/S11))
            filter_type: (B,) long tensor 0..3 (LP/HP/BP/BS)
            fc_hz: (B,) float tensor of cutoff freq in Hz
        """
        wave_feat = self.wave_encoder(wave)  # (B,Lw,d)
        if self.spec_encoder is None:
            return BaseModelOutput(last_hidden_state=wave_feat)
        if filter_type is None or fc_hz is None:
            raise ValueError("Spec encoder enabled but filter_type/fc_hz not provided.")
        log_fc = torch.log10(fc_hz.clamp_min(1e-6))
        spec_tok = self.spec_encoder(filter_type, log_fc).unsqueeze(1)  # (B,1,d)
        enc = torch.cat([spec_tok, wave_feat], dim=1)
        return BaseModelOutput(last_hidden_state=enc)

    def forward(
        self,
        wave: torch.Tensor,
        filter_type: Optional[torch.Tensor] = None,
        fc_hz: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        value_targets: Optional[torch.Tensor] = None,
        macro_slot_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        dec_hidden = None
        if value_targets is not None or macro_slot_targets is not None:
            dec_hidden = outputs.decoder_hidden_states[-1]  # (B,L,d)

        # mixed discrete-continuous loss: token CE + (mantissa CE + decade CE + residual regression)
        if value_targets is not None:
            mant_tgt, dec_tgt, res_tgt, valid_vals = torch_decompose_mdr(value_targets, cfg=self.mdr_cfg)
            valid_mask = valid_vals
            if labels is not None and self.value_token_ids is not None:
                val_ids = self.value_token_ids.to(labels.device)
                if hasattr(torch, "isin"):
                    is_val_tok = torch.isin(labels, val_ids)
                else:
                    is_val_tok = (labels.unsqueeze(-1) == val_ids.view(1, 1, -1)).any(-1)
                valid_mask = valid_mask & is_val_tok
            if valid_mask.any():
                slot_type_ids = None
                if labels is not None and self.slot_type_lookup is not None and self.value_head.slot_type_emb is not None:
                    vocab_size = self.slot_type_lookup.shape[0]
                    safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
                    safe_labels = torch.clamp(safe_labels, 0, vocab_size - 1)
                    slot_type_ids = self.slot_type_lookup[safe_labels]
                    slot_type_ids = torch.where(slot_type_ids >= 0, slot_type_ids, torch.zeros_like(slot_type_ids))
                pred = self.value_head(dec_hidden, slot_type_ids=slot_type_ids)
                mant_loss = F.cross_entropy(pred.mantissa_logits[valid_mask], mant_tgt[valid_mask])
                dec_loss = F.cross_entropy(pred.decade_logits[valid_mask], dec_tgt[valid_mask])
                res_loss = F.mse_loss(pred.residual_log10[valid_mask], res_tgt[valid_mask])
                value_loss = mant_loss + dec_loss + res_loss
                loss = outputs.loss + self.value_loss_weight * value_loss if outputs.loss is not None else value_loss
                outputs.loss = loss
                outputs.value_loss = value_loss
                outputs.value_loss_mantissa = mant_loss
                outputs.value_loss_decade = dec_loss
                outputs.value_loss_residual = res_loss

        if macro_slot_targets is not None:
            if self.macro_value_head is None or self.macro_slot_mask is None or labels is None:
                raise ValueError("macro_slot_targets provided but macro_value_head/macro_slot_mask/labels missing.")
            pred_raw = self.macro_value_head(dec_hidden)
            vocab_size = self.macro_slot_mask.shape[0]
            safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            safe_labels = torch.clamp(safe_labels, 0, vocab_size - 1)
            slot_mask = self.macro_slot_mask[safe_labels]
            targets = macro_slot_targets.to(pred_raw.dtype)
            valid = (slot_mask > 0) & torch.isfinite(targets)
            if valid.any():
                macro_loss = F.mse_loss(pred_raw[valid], targets[valid])
                outputs.macro_value_loss = macro_loss
                loss = outputs.loss + self.macro_value_loss_weight * macro_loss if outputs.loss is not None else macro_loss
                outputs.loss = loss
        return outputs

    @torch.no_grad()
    def predict_values(
        self,
        wave: torch.Tensor,
        filter_type: Optional[torch.Tensor] = None,
        fc_hz: Optional[torch.Tensor] = None,
        *,
        token_ids: torch.Tensor,
        mode: str = "precision",
    ) -> torch.Tensor:
        """
        Predict per-position physical values from decoder hidden states.

        The caller should mask out non-<VAL_*> positions (e.g., set them to NaN) if desired.
        """
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        out = self.t5(
            encoder_outputs=encoder_outputs,
            labels=token_ids,
            output_hidden_states=True,
        )
        dec_hidden = out.decoder_hidden_states[-1]
        slot_type_ids = None
        if self.slot_type_lookup is not None and self.value_head.slot_type_emb is not None:
            vocab_size = self.slot_type_lookup.shape[0]
            safe_ids = torch.clamp(token_ids, 0, vocab_size - 1)
            slot_type_ids = self.slot_type_lookup[safe_ids]
            slot_type_ids = torch.where(slot_type_ids >= 0, slot_type_ids, torch.zeros_like(slot_type_ids))
        pred = self.value_head(dec_hidden, slot_type_ids=slot_type_ids)
        mant_idx = torch.argmax(pred.mantissa_logits, dim=-1)
        dec_idx = torch.argmax(pred.decade_logits, dim=-1)
        values = torch_compose_value(mant_idx, dec_idx, pred.residual_log10, cfg=self.mdr_cfg, mode=mode)  # (B,L)
        return values

    @torch.no_grad()
    def predict_macro_values(
        self,
        wave: torch.Tensor,
        filter_type: Optional[torch.Tensor] = None,
        fc_hz: Optional[torch.Tensor] = None,
        *,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict per-position raw slot values for macro tokens.
        Returns raw values in log-space (same scale as bilevel slot_raw).
        """
        if self.macro_value_head is None:
            raise ValueError("macro_value_head not initialized.")
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        out = self.t5(
            encoder_outputs=encoder_outputs,
            labels=token_ids,
            output_hidden_states=True,
        )
        dec_hidden = out.decoder_hidden_states[-1]
        return self.macro_value_head(dec_hidden)

    @torch.no_grad()
    def generate(
        self,
        wave: torch.Tensor,
        filter_type: Optional[torch.Tensor] = None,
        fc_hz: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        return self.t5.generate(encoder_outputs=encoder_outputs, **kwargs)
