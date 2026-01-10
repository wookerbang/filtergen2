# Experiment Record Template (Bilevel + STE/Viterbi)

Use this template to record training runs. Fill in the Value column and keep Notes for context.

## Run Info

| Field | Value | Notes |
|---|---|---|
| Run ID | ____ | Unique name for tracking (e.g., `ste_k3_tau1e0_seed0`) |
| Date | ____ | YYYY-MM-DD |
| Git commit / tag | ____ | `git rev-parse --short HEAD` |
| Script | `scripts/train_bilevel.py` | Training entry point |
| Output dir | ____ | Matches `--output` |
| Notes | ____ | Short description of goal/ablation |

## Environment

| Field | Value | Notes |
|---|---|---|
| Machine | ____ | Hostname or GPU server |
| GPU | ____ | Model + VRAM |
| CUDA | ____ | `nvidia-smi` |
| PyTorch | ____ | `python -c "import torch; print(torch.__version__)"` |
| Python | ____ | `python --version` |

## Data & Input

| Parameter | Value | Notes |
|---|---|---|
| `--data` | ____ | Train jsonl path |
| `--eval-data` | ____ | Optional eval jsonl |
| `--use-wave` | ____ | `ideal/real/both/ideal_s21/real_s21/mix` |
| `--wave-norm` | ____ | Normalize waveform channels |
| `--freq-mode` | ____ | `log_f_centered` recommended |
| `--freq-scale` | ____ | `log_f_mean` recommended |
| `--no-s11` | ____ | Use only S21 channels |
| `--seed` | ____ | Seed for data/model |

## Model & Topology

| Parameter | Value | Notes |
|---|---|---|
| `k_max` | ____ | From dataset scan |
| `--k-percentile` | ____ | Length percentile |
| `--k-cap` | ____ | Hard cap on K |
| `--k-min` | ____ | Min K |
| `macro_vocab_size` | ____ | From dataset scan |
| `slot_count` | ____ | Max slots per macro |
| `--d-model` | ____ | Model width |
| `--hidden-mult` | ____ | MLP width multiplier |
| `--dropout` | ____ | Dropout prob |
| `--spec-mode` | ____ | `type_fc` or `none` |
| `--gate-skip-bias` | ____ | SKIP logit bias |

## Role Queries & Symmetry

| Parameter | Value | Notes |
|---|---|---|
| `--use-role-queries` | ____ | Role-aware slot queries |
| `--role-input-frac` | ____ | Fraction near input |
| `--role-output-frac` | ____ | Fraction near output |
| `--sym-weight` | ____ | Symmetry regularizer |
| `--sym-core-only` | ____ | Apply to core only |

## Optimization & Training

| Parameter | Value | Notes |
|---|---|---|
| `--epochs` | ____ | Total epochs |
| `--batch-size` | ____ | Batch size |
| `--grad-accum` | ____ | Gradient accumulation |
| `--lr` | ____ | Adam lr |
| `--clip-grad` | ____ | 0 disables |
| `--amp-bf16` | ____ | BF16 autocast |
| `--num-workers` | ____ | DataLoader workers |
| `--prefetch-factor` | ____ | When workers > 0 |
| `--pin-memory` | ____ | CUDA only |
| `--persistent-workers` | ____ | Worker reuse |
| `--skip-nonfinite` | ____ | Skip non-finite batches |
| `--device` | ____ | `cuda` or `cpu` |
| `--dtype` | ____ | `float32/float64` |
| `--circuit-cache-size` | ____ | 0 disables |

## Bilevel / Physics

| Parameter | Value | Notes |
|---|---|---|
| `--phys-weight` | ____ | Physics loss weight |
| `--use-unroll` | ____ | Inner-loop on/off |
| `--unroll-steps` | ____ | Inner GD steps |
| `--unroll-create-graph` | ____ | Needed for hypergrad |
| `--inner-lr` | ____ | Inner GD lr |
| `--inner-max-step` | ____ | Step size clamp |
| `--inner-raw-min` | ____ | Raw log range min |
| `--inner-raw-max` | ____ | Raw log range max |
| `--inner-nan-backoff` | ____ | Backoff factor |
| `--inner-nan-tries` | ____ | Backoff attempts |

## Gumbel & Schedules

| Parameter | Value | Notes |
|---|---|---|
| `--gumbel-tau` | ____ | Start temperature |
| `--gumbel-tau-min` | ____ | End temperature |
| `--gumbel-tau-decay-frac` | ____ | Decay fraction |
| `--alpha-start` | ____ | Macro CE weight start |
| `--alpha-min` | ____ | Macro CE weight end |
| `--alpha-decay-frac` | ____ | CE decay fraction |
| `--len-weight` | ____ | Length loss weight |

## STE (Hard Forward + Soft Backward)

| Parameter | Value | Notes |
|---|---|---|
| `--ste-phys` | ____ | Enables STE; overrides `--matrix-mix` |
| `--ste-topk` | ____ | Top-k sparsity (default 3) |
| `--ste-topk-anneal-frac` | ____ | Last fraction to anneal k→1 |
| `--ste-phys-weight` | ____ | Weight for soft physics loss |

## Matrix Mix (only if STE is off)

| Parameter | Value | Notes |
|---|---|---|
| `--matrix-mix` | ____ | Soft mix mode |
| `--mix-topk` | ____ | Top-k for mix |

## Transition C / Viterbi

| Parameter | Value | Notes |
|---|---|---|
| `--c-reg-weight` | ____ | Transition regularizer |
| `--c-skip-penalty` | ____ | SKIP→nonSKIP penalty |
| `--c-redundant-penalty` | ____ | Redundant self-transition |
| Eval `--use-viterbi` | ____ | Use during eval |

## Checkpoint & Logging

| Parameter | Value | Notes |
|---|---|---|
| `--output` | ____ | Checkpoint dir |
| `--init-from` | ____ | Resume/finetune |
| `--log-steps` | ____ | Log frequency |
| `--save-steps` | ____ | Checkpoint frequency |
| `--save-total-limit` | ____ | Max checkpoints |

## Results (fill after training/eval)

| Metric | Value | Notes |
|---|---|---|
| `macro_acc` | ____ | Overall macro accuracy |
| `macro_non_skip_acc` | ____ | Non-SKIP accuracy |
| `len_mae` | ____ | Length MAE |
| `len_exact` | ____ | Exact length rate |
| `phys_loss` | ____ | Mean physics loss |
| `phys_soft` | ____ | STE soft loss (if on) |
| `yield_pre` | ____ | Pre-refine yield |
| `yield_post` | ____ | Post-refine yield |
| `runtime/epoch` | ____ | Wall time |

## Command Snapshot

Paste the exact training command here for reproducibility.

```
python scripts/train_bilevel.py ...
```
