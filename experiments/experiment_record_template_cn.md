# 实验记录模板（Bilevel / STE / Viterbi）

> 使用方法：直接在“值”栏填写，默认值已列在“默认值”栏。  
> 建议每次训练都复制一份此模板，保存为 `experiments/records/exp_*.md`。

## 0) 运行概览（必填）

| 项目 | 默认值 | 值 | 说明 |
|---|---|---|---|
| Run ID |  |  | 唯一标识（如 `ste_k3_tau1_seed0`） |
| 日期 |  |  | YYYY-MM-DD |
| Git 版本 |  |  | `git rev-parse --short HEAD` |
| 训练脚本 | `scripts/train_bilevel.py` |  | 固定 |
| 输出目录 | `checkpoints/bilevel` |  | 对应 `--output` |
| 实验目标 |  |  | 一句话说明这次实验目的 |

## 1) 核心开关摘要（勾选/填写）

- [ ] STE（`--ste-phys`）
- [ ] Matrix Mix（`--matrix-mix`，仅在 STE 关闭时生效）
- [ ] Viterbi 评估（`scripts/eval_bilevel.py --use-viterbi`）
- [ ] Role Queries（`--use-role-queries`）
- [ ] Symmetry 正则（`--sym-weight` > 0）
- [ ] Unroll（`--use-unroll`）

## 2) 数据与输入

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `--data` |  |  | 训练集 jsonl |
| `--eval-data` |  |  | 验证集（可空） |
| `--use-wave` | `ideal` |  | ideal/real/both/ideal_s21/real_s21/mix |
| `--wave-norm` | `false` |  | 是否标准化波形 |
| `--freq-mode` | `log_f_centered` |  | log_fc/linear_fc/log_f/log_f_centered/none |
| `--freq-scale` | `log_f_mean` |  | log_fc/log_f_mean/none |
| `--no-s11` | `false` |  | 若启用则仅 S21 |
| `--seed` | `0` |  | 随机种子 |

## 3) 模型与拓扑

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `k_max` |  |  | 数据扫描结果 |
| `--k-percentile` | `95.0` |  | K 分位数 |
| `--k-cap` | `12` |  | K 上限 |
| `--k-min` | `12` |  | K 下限 |
| `macro_vocab_size` |  |  | 数据扫描结果 |
| `slot_count` |  |  | 宏库决定 |
| `--d-model` | `768` |  | 模型维度 |
| `--hidden-mult` | `2` |  | MLP 宽度倍数 |
| `--dropout` | `0.1` |  | Dropout |
| `--spec-mode` | `type_fc` |  | type_fc / none |
| `--gate-skip-bias` | `1.0` |  | SKIP 初始偏置 |

## 4) 训练与优化

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `--epochs` | `5` |  | 训练轮数 |
| `--batch-size` | `32` |  | 批大小 |
| `--grad-accum` | `1` |  | 梯度累积 |
| `--lr` | `1e-4` |  | 学习率 |
| `--clip-grad` | `5.0` |  | 梯度裁剪（<=0 关闭） |
| `--amp-bf16` | `true` |  | BF16 自动混精度 |
| `--num-workers` | `8` |  | DataLoader 进程 |
| `--prefetch-factor` | `2` |  | 预取因子 |
| `--pin-memory` | `true` |  | CUDA 加速 |
| `--persistent-workers` | `true` |  | 复用 worker |
| `--skip-nonfinite` | `false` |  | 跳过非有限 batch |
| `--device` | 自动（cuda 如可用，否则 cpu） |  | 训练设备 |
| `--dtype` | `float32` |  | float32 / float64 |
| `--circuit-cache-size` | `2048` |  | 电路缓存大小 |

## 5) Bilevel / 物理内环

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `--phys-weight` | `1e-4` |  | 物理损失权重 |
| `--use-unroll` | `true` |  | 是否启用内环 |
| `--unroll-steps` | `5` |  | 内环步数 |
| `--unroll-create-graph` | `true` |  | 是否保留高阶图 |
| `--inner-lr` | `1e-2` |  | 内环学习率 |
| `--inner-max-step` | `0.5` |  | 内环步长上限 |
| `--inner-raw-min` | `-32.0` |  | raw 最小 |
| `--inner-raw-max` | `-12.0` |  | raw 最大 |
| `--inner-nan-backoff` | `0.5` |  | nan 回退系数 |
| `--inner-nan-tries` | `3` |  | nan 回退次数 |

## 6) Gumbel 与调度

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `--gumbel-tau` | `1.0` |  | 初始温度 |
| `--gumbel-tau-min` | 同 `--gumbel-tau` |  | 最小温度 |
| `--gumbel-tau-decay-frac` | `0.5` |  | 温度退火比例 |
| `--alpha-start` | `1.0` |  | Macro CE 初始权重 |
| `--alpha-min` | `0.1` |  | Macro CE 最小权重 |
| `--alpha-decay-frac` | `0.3` |  | Macro CE 退火比例 |
| `--len-weight` | `1e-3` |  | 长度损失权重 |

## 7) STE（硬前向 + 软反向）

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `--ste-phys` | `false` |  | 开启 STE（会覆盖 matrix-mix） |
| `--ste-topk` | `3` |  | soft top-k |
| `--ste-topk-anneal-frac` | `0.2` |  | 最后阶段退火到 k=1 |
| `--ste-phys-weight` | `1.0` |  | soft 物理损失权重 |

## 8) Matrix Mix（仅 STE 关闭时）

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `--matrix-mix` | `false` |  | 混合物理模式 |
| `--mix-topk` | `0` |  | mix 的 top-k |

## 9) 结构先验 C / Viterbi

| 参数 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `--c-reg-weight` | `0.0` |  | 转移正则权重 |
| `--c-skip-penalty` | `100.0` |  | SKIP->nonSKIP 惩罚 |
| `--c-redundant-penalty` | `1.0` |  | 冗余自转移惩罚 |
| Eval `--use-viterbi` | `false` |  | 评估时是否启用 |

## 10) 评估指标（训练/验证完成后填写）

| 指标 | 默认值 | 值 | 说明 |
|---|---|---|---|
| `macro_acc` |  |  | 宏整体准确率 |
| `macro_non_skip_acc` |  |  | 非 SKIP 宏准确率 |
| `len_mae` |  |  | 长度 MAE |
| `len_exact` |  |  | 长度完全匹配率 |
| `phys_loss` |  |  | 物理损失均值 |
| `phys_soft` |  |  | STE soft 物理损失 |
| `yield_pre` |  |  | refine 前良率 |
| `yield_post` |  |  | refine 后良率 |
| `runtime/epoch` |  |  | 每轮耗时 |

## 11) 命令快照（必填）

```
python scripts/train_bilevel.py ...
```

## 12) 备注与结论

填“观察到的现象 + 下一步计划”，避免只写数值。
