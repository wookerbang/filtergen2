"""
PyTorch Dataset 包装 processed jsonl。
"""

from __future__ import annotations

import json
import math
from typing import Literal, Sequence

import torch
from torch.utils.data import Dataset


class FilterDesignDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        use_wave: Literal["ideal", "real", "both", "ideal_s21", "real_s21", "mix"] = "real",
        mix_real_prob: float = 0.3,
        use_repr: Literal["vact", "vact_struct", "dsl", "sfci", "action"] = "vact",
        normalize_wave: bool = False,
        freq_mode: Literal["none", "log_fc", "linear_fc", "log_f", "log_f_centered"] = "log_fc",
        freq_scale: Literal["none", "log_fc", "log_f_mean"] = "none",
        include_s11: bool = True,
    ):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.use_wave = use_wave
        self.mix_real_prob = mix_real_prob
        self.use_repr = use_repr
        self.normalize_wave = normalize_wave
        self.freq_mode = freq_mode
        self.freq_scale = freq_scale
        self.include_s11 = include_s11

    def _tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        """
        Convert pre-split tokens (already without special tokens) to ids.

        Note: the tokenizer here is WordLevel with a whitespace pre-tokenizer,
        so feeding the tokens back into ``tokenizer(...)`` will split the
        angle-bracket tokens and turn everything into <unk>. We therefore
        map the tokens directly via ``convert_tokens_to_ids`` when available.
        """
        if not tokens:
            return []

        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
        elif callable(getattr(self.tokenizer, "__call__", None)):
            out = self.tokenizer(tokens, is_split_into_words=True, add_special_tokens=False)
            ids = out.get("input_ids") or out.get("ids") or []
            if ids and isinstance(ids[0], list):
                # flatten batch-style output
                ids = sum(ids, [])
        elif hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(tokens, add_special_tokens=False)
        else:
            ids = [self.tokenizer.get(t, 0) for t in tokens]

        return [int(i) for i in ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        freq = torch.tensor(s["freq_hz"], dtype=torch.float32)
        ideal_s21 = torch.tensor(s["ideal_s21_db"], dtype=torch.float32)
        ideal_s11 = torch.tensor(s["ideal_s11_db"], dtype=torch.float32)
        real_s21 = torch.tensor(s["real_s21_db"], dtype=torch.float32)
        real_s11 = torch.tensor(s["real_s11_db"], dtype=torch.float32)

        fc_hz = float(s.get("fc_hz", 0.0) or 0.0)
        if not math.isfinite(fc_hz) or fc_hz <= 0.0:
            valid = torch.isfinite(freq)
            if valid.any():
                fmin = float(freq[valid].min().item())
                fmax = float(freq[valid].max().item())
                fc_hz = math.sqrt(max(fmin * fmax, 1e-12))
            else:
                fc_hz = 1.0

        mode = self.use_wave
        if mode == "mix":
            mode = "real" if torch.rand(1).item() < self.mix_real_prob else "ideal"

        if mode == "ideal":
            wave = torch.stack([ideal_s21, ideal_s11], dim=0)
        elif mode == "real":
            wave = torch.stack([real_s21, real_s11], dim=0)
        elif mode == "ideal_s21":
            wave = ideal_s21.unsqueeze(0)
        elif mode == "real_s21":
            wave = real_s21.unsqueeze(0)
        else:
            wave = torch.stack([ideal_s21, ideal_s11, real_s21, real_s11], dim=0)

        if not self.include_s11:
            if wave.shape[0] == 4:
                wave = wave[[0, 2], :]
            elif wave.shape[0] > 1:
                wave = wave[:1]

        freq_channels = 0
        if self.freq_mode != "none" or self.freq_scale != "none":
            eps = 1e-12
            freq_clamped = freq.clamp_min(eps)
            freq_feats = []
            logf = None
            mean_logf = None
            if self.freq_mode == "log_fc":
                freq_feats.append(torch.log10(freq_clamped / fc_hz))
            elif self.freq_mode == "linear_fc":
                freq_feats.append(freq / fc_hz)
            elif self.freq_mode == "log_f":
                logf = torch.log10(freq_clamped)
                freq_feats.append(logf)
            elif self.freq_mode == "log_f_centered":
                logf = torch.log10(freq_clamped)
                mean_logf = float(logf.mean().item())
                freq_feats.append(logf - mean_logf)
            elif self.freq_mode != "none":
                raise ValueError(f"Unknown freq_mode: {self.freq_mode}")

            if self.freq_scale == "log_fc":
                freq_feats.append(torch.full_like(freq, math.log10(fc_hz)))
            elif self.freq_scale == "log_f_mean":
                if logf is None:
                    logf = torch.log10(freq_clamped)
                if mean_logf is None:
                    mean_logf = float(logf.mean().item())
                freq_feats.append(torch.full_like(freq, mean_logf))
            elif self.freq_scale != "none":
                raise ValueError(f"Unknown freq_scale: {self.freq_scale}")

            if freq_feats:
                freq_wave = torch.stack(freq_feats, dim=0)
                wave = torch.cat([freq_wave, wave], dim=0)
                freq_channels = freq_wave.shape[0]

        ftype = s.get("filter_type", "lowpass")
        type_map = {"lowpass": 0, "highpass": 1, "bandpass": 2, "bandstop": 3}
        type_id = type_map.get(ftype, 0)
        scalar = torch.tensor([type_id, fc_hz], dtype=torch.float32)

        value_targets = None
        if self.use_repr == "vact":
            tokens_raw = s.get("vact_tokens")
        elif self.use_repr == "vact_struct":
            tokens_raw = s.get("vact_struct_tokens")
        elif self.use_repr == "dsl":
            tokens_raw = s.get("dsl_tokens")
            value_targets = s.get("dsl_slot_values")
        elif self.use_repr == "sfci":
            tokens_raw = s.get("sfci_tokens")
        else:
            tokens_raw = s.get("action_tokens")
        tokens_raw = tokens_raw or []
        token_ids = self._tokens_to_ids(tokens_raw)

        if self.normalize_wave:
            # channel-wise standardization to stabilize optimization
            if freq_channels < wave.shape[0]:
                wave_sig = wave[freq_channels:]
                wave_sig = wave_sig - wave_sig.mean(dim=-1, keepdim=True)
                wave_std = wave_sig.std(dim=-1, keepdim=True).clamp_min(1e-4)
                wave[freq_channels:] = wave_sig / wave_std

        return {
            "freq": freq,
            "wave": wave,
            "scalar": scalar,
            "ideal_s21_db": ideal_s21,
            "real_s21_db": real_s21,
            # keep both keys so collate_fn can pick by repr
            "input_ids": token_ids,
            "vact_tokens": token_ids if self.use_repr == "vact" else None,
            "vact_struct_tokens": token_ids if self.use_repr == "vact_struct" else None,
            "dsl_tokens": token_ids if self.use_repr == "dsl" else None,
            "dsl_tokens_raw": list(tokens_raw or []) if self.use_repr == "dsl" else None,
            "sfci_tokens": token_ids if self.use_repr == "sfci" else None,
            "action_tokens": token_ids if self.use_repr == "action" else None,
            "value_targets": value_targets,
        }
