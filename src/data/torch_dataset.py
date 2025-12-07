"""
PyTorch Dataset 包装 processed jsonl。
"""

from __future__ import annotations

import json
from typing import Literal

import torch
from torch.utils.data import Dataset


class FilterDesignDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        use_wave: Literal["ideal", "real", "both"] = "real",
    ):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.use_wave = use_wave

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        freq = torch.tensor(s["freq_hz"], dtype=torch.float32)
        ideal_s21 = torch.tensor(s["ideal_s21_db"], dtype=torch.float32)
        ideal_s11 = torch.tensor(s["ideal_s11_db"], dtype=torch.float32)
        real_s21 = torch.tensor(s["real_s21_db"], dtype=torch.float32)
        real_s11 = torch.tensor(s["real_s11_db"], dtype=torch.float32)

        if self.use_wave == "ideal":
            wave = torch.stack([ideal_s21, ideal_s11], dim=0)
        elif self.use_wave == "real":
            wave = torch.stack([real_s21, real_s11], dim=0)
        else:
            wave = torch.stack([ideal_s21, ideal_s11, real_s21, real_s11], dim=0)

        type_id = 0 if s["filter_type"] == "lowpass" else 1
        scalar = torch.tensor([type_id, s["fc_hz"]], dtype=torch.float32)

        sfci_tokens = s.get("sfci_tokens") or s.get("sfci_sequence", "").split()
        if hasattr(self.tokenizer, "encode"):
            token_ids = self.tokenizer.encode(sfci_tokens)
        else:
            token_ids = [self.tokenizer[t] for t in sfci_tokens]
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        return {
            "freq": freq,
            "wave": wave,
            "scalar": scalar,
            "input_ids": input_ids,
        }
