from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .scalar_encoder import SpecEncoder
from .value_token_embed import ValueAwareEmbedding
from .waveform_encoder import MultiScaleWaveformEncoder


class VACTT5(nn.Module):
    """
    Waveform/spec encoder + T5 decoder for VACT generation.
    Defaults to t5-small for快速迭代，可切换 t5-base.
    """

    def __init__(
        self,
        t5_name: str = "t5-small",
        value_token_to_value: Optional[Mapping[int, float]] = None,
        waveform_in_channels: int = 1,
        d_model_override: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        if vocab_size is not None:
            self.t5.resize_token_embeddings(vocab_size)
        d_model = d_model_override or self.t5.config.d_model

        self.wave_encoder = MultiScaleWaveformEncoder(d_model=d_model, in_channels=waveform_in_channels)
        self.spec_encoder = SpecEncoder(d_model=d_model, type_vocab_size=2)

        if value_token_to_value:
            base_emb = self.t5.get_input_embeddings()
            vae = ValueAwareEmbedding(base_emb, value_token_to_value)
            self.t5.set_input_embeddings(vae)

    def encode(self, wave: torch.Tensor, filter_type: torch.Tensor, fc_hz: torch.Tensor) -> BaseModelOutput:
        """
        Args:
            wave: (B, C, L) waveform (S21 or S21+S11)
            filter_type: (B,) long tensor 0/1
            fc_hz: (B,) float tensor of cutoff freq in Hz
        """
        log_fc = torch.log10(fc_hz.clamp_min(1e-6))
        spec_tok = self.spec_encoder(filter_type, log_fc).unsqueeze(1)  # (B,1,d)
        wave_feat = self.wave_encoder(wave)  # (B,Lw,d)
        enc = torch.cat([spec_tok, wave_feat], dim=1)
        return BaseModelOutput(last_hidden_state=enc)

    def forward(
        self,
        wave: torch.Tensor,
        filter_type: torch.Tensor,
        fc_hz: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        return self.t5(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        wave: torch.Tensor,
        filter_type: torch.Tensor,
        fc_hz: torch.Tensor,
        **kwargs,
    ):
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        return self.t5.generate(encoder_outputs=encoder_outputs, **kwargs)
