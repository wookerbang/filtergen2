from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _default_hard_ban_self_macros(id_to_macro: Sequence[str]) -> list[str]:
    return [m for m in id_to_macro if "WIRE" in m or "NULL" in m]


def build_transition_matrices(
    *,
    id_to_macro: Sequence[str],
    skip_id: int,
    soft_skip_penalty: float = 100.0,
    soft_redundant_penalty: float = 1.0,
    redundant_macros: Iterable[str] | None = None,
    hard_ban_skip_to_non_skip: bool = True,
    hard_ban_self_macros: Iterable[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build hard transition mask C and soft penalty matrix P for macro sequences.
    """
    macro_count = len(id_to_macro)
    if skip_id != macro_count:
        raise ValueError("skip_id must equal len(id_to_macro) for C/P construction.")
    size = macro_count + 1
    c_hard = torch.zeros((size, size), dtype=torch.float32)
    p_soft = torch.zeros((size, size), dtype=torch.float32)

    if hard_ban_skip_to_non_skip:
        c_hard[skip_id, :skip_id] = float("-inf")

    if float(soft_skip_penalty) > 0.0:
        p_soft[skip_id, :skip_id] = float(soft_skip_penalty)

    macro_to_id = {m: i for i, m in enumerate(id_to_macro)}
    if redundant_macros is not None:
        for macro in redundant_macros:
            idx = macro_to_id.get(macro)
            if idx is None:
                continue
            if float(soft_redundant_penalty) > 0.0:
                p_soft[idx, idx] = float(soft_redundant_penalty)

    if hard_ban_self_macros is None:
        hard_ban_self_macros = _default_hard_ban_self_macros(id_to_macro)
    for macro in hard_ban_self_macros:
        idx = macro_to_id.get(macro)
        if idx is None:
            continue
        c_hard[idx, idx] = float("-inf")

    return c_hard, p_soft


def expected_transition_penalty(probs: torch.Tensor, p_soft: torch.Tensor) -> torch.Tensor:
    """
    Expected transition penalty over a batch of sequences.
    """
    if probs.dim() == 2:
        probs = probs.unsqueeze(0)
    if probs.dim() != 3:
        raise ValueError(f"probs must have shape (B,K,M); got {tuple(probs.shape)}")
    if probs.shape[-1] != p_soft.shape[0]:
        raise ValueError("probs last dim must match p_soft shape.")
    if probs.shape[1] < 2:
        return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

    p_soft = p_soft.to(device=probs.device, dtype=probs.dtype)
    p_t = probs[:, :-1, :]
    p_next = probs[:, 1:, :]
    penalty = torch.matmul(p_t, p_soft)
    penalty = (penalty * p_next).sum(dim=-1)
    return penalty.sum(dim=1).mean()


def viterbi_decode(unary_logits: torch.Tensor, c_hard: torch.Tensor) -> torch.Tensor:
    """
    Viterbi decode for linear-chain transitions.
    """
    if unary_logits.dim() == 2:
        unary_logits = unary_logits.unsqueeze(0)
        squeeze = True
    elif unary_logits.dim() == 3:
        squeeze = False
    else:
        raise ValueError(f"unary_logits must have shape (K,M) or (B,K,M); got {tuple(unary_logits.shape)}")

    batch, steps, states = unary_logits.shape
    if c_hard.shape != (states, states):
        raise ValueError("c_hard shape must match unary state dim.")
    c_hard = c_hard.to(device=unary_logits.device, dtype=unary_logits.dtype)

    scores = unary_logits[:, 0, :]
    backpointers: list[torch.Tensor] = []
    for t in range(1, steps):
        prev = scores.unsqueeze(2) + c_hard.unsqueeze(0)
        best_score, best_idx = torch.max(prev, dim=1)
        scores = best_score + unary_logits[:, t, :]
        backpointers.append(best_idx)

    paths = torch.zeros((batch, steps), device=unary_logits.device, dtype=torch.long)
    paths[:, -1] = torch.argmax(scores, dim=1)
    for t in range(steps - 1, 0, -1):
        prev_idx = backpointers[t - 1]
        paths[:, t - 1] = prev_idx.gather(1, paths[:, t].unsqueeze(1)).squeeze(1)

    if torch.isinf(c_hard).any():
        prev = paths[:, :-1]
        nxt = paths[:, 1:]
        if not torch.isfinite(c_hard[prev, nxt]).all():
            raise ValueError("Viterbi decode produced a hard-banned transition.")

    if squeeze:
        return paths[0]
    return paths
