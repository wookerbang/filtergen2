#!/usr/bin/env bash
set -euo pipefail

DATA_DEBUG="data/processed/debug_1k.jsonl"
DATA_BASE="data/processed/balanced_100k_motif/train.jsonl"
OUT_ROOT="checkpoints/sandbox_sweep"
EVAL_DIR="outputs/sandbox_sweep"
SUMMARY_CSV="${EVAL_DIR}/summary.csv"

mkdir -p "${OUT_ROOT}" "${EVAL_DIR}"

if [[ ! -f "${DATA_DEBUG}" ]]; then
  if [[ -f "${DATA_BASE}" ]]; then
    echo "[prep] creating ${DATA_DEBUG} from ${DATA_BASE}"
    shuf -n 1000 "${DATA_BASE}" > "${DATA_DEBUG}"
  else
    echo "[error] missing ${DATA_DEBUG} and ${DATA_BASE}" >&2
    exit 1
  fi
fi

DEVICE="${DEVICE:-cuda}"

EPOCHS=10
BATCH=64
LR=1e-4
ALPHA_START_BASE=0.5
ALPHA_MIN_BASE=0.1
ALPHA_DECAY=0.5
PHYS_WEIGHT_BASE=1e-4
LEN_WEIGHT_BASE=1e-3
UNROLL_BASE=5

COMMON_ARGS=(
  --data "${DATA_DEBUG}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --lr "${LR}"
  --k-min 12 --k-cap 12
  --freq-mode log_f_centered
  --spec-mode type_fc
  --no-s11
  --amp-bf16
  --num-workers 0
  --log-steps 5
  --save-steps 0
  --clip-grad 1.0
  --skip-nonfinite
  --device "${DEVICE}"
)

append_summary() {
  local eval_json="$1"
  local run_name="$2"
  local alpha_start="$3"
  local alpha_min="$4"
  local phys_weight="$5"
  local len_weight="$6"
  local unroll_steps="$7"
  RUN_NAME="${run_name}" ALPHA_START="${alpha_start}" ALPHA_MIN="${alpha_min}" \
    PHYS_WEIGHT="${phys_weight}" LEN_WEIGHT="${len_weight}" UNROLL_STEPS="${unroll_steps}" \
    EVAL_JSON="${eval_json}" SUMMARY_CSV="${SUMMARY_CSV}" python - <<'PY'
import csv
import json
import os

eval_json = os.environ["EVAL_JSON"]
summary_csv = os.environ["SUMMARY_CSV"]
with open(eval_json, "r") as f:
    data = json.load(f)

row = {
    "run": os.environ["RUN_NAME"],
    "alpha_start": os.environ["ALPHA_START"],
    "alpha_min": os.environ["ALPHA_MIN"],
    "phys_weight": os.environ["PHYS_WEIGHT"],
    "len_weight": os.environ["LEN_WEIGHT"],
    "unroll_steps": os.environ["UNROLL_STEPS"],
    "macro_acc": data.get("macro_acc"),
    "macro_non_skip_acc": data.get("macro_non_skip_acc"),
    "len_exact": data.get("len_exact"),
    "mse_post": data.get("mse_post"),
    "yield_post": data.get("yield_post"),
    "yield_oracle": data.get("yield_oracle"),
}

write_header = not os.path.exists(summary_csv)
with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_one() {
  local run_name="$1"
  local alpha_start="$2"
  local alpha_min="$3"
  local phys_weight="$4"
  local len_weight="$5"
  local unroll_steps="$6"

  local run_dir="${OUT_ROOT}/${run_name}"
  local eval_json="${EVAL_DIR}/${run_name}.json"

  PYTHONUNBUFFERED=1 python -u scripts/train_bilevel.py \
    --output "${run_dir}" \
    --alpha-start "${alpha_start}" \
    --alpha-min "${alpha_min}" \
    --alpha-decay-frac "${ALPHA_DECAY}" \
    --phys-weight "${phys_weight}" \
    --len-weight "${len_weight}" \
    --unroll-steps "${unroll_steps}" \
    "${COMMON_ARGS[@]}"

  python scripts/eval_bilevel.py \
    --data "${DATA_DEBUG}" \
    --ckpt "${run_dir}/epoch_${EPOCHS}/pytorch_model.bin" \
    --device "${DEVICE}" \
    --max-samples 200 \
    --unroll-steps 10 \
    --output "${eval_json}"

  append_summary "${eval_json}" "${run_name}" "${alpha_start}" "${alpha_min}" "${phys_weight}" "${len_weight}" "${unroll_steps}"
}

run_one "baseline" "${ALPHA_START_BASE}" "${ALPHA_MIN_BASE}" "${PHYS_WEIGHT_BASE}" "${LEN_WEIGHT_BASE}" "${UNROLL_BASE}"

# alpha_start sweep (alpha_min fixed)
run_one "alpha_start_0.3" "0.3" "${ALPHA_MIN_BASE}" "${PHYS_WEIGHT_BASE}" "${LEN_WEIGHT_BASE}" "${UNROLL_BASE}"
run_one "alpha_start_0.8" "0.8" "${ALPHA_MIN_BASE}" "${PHYS_WEIGHT_BASE}" "${LEN_WEIGHT_BASE}" "${UNROLL_BASE}"

# alpha_min sweep (alpha_start fixed)
run_one "alpha_min_0.05" "${ALPHA_START_BASE}" "0.05" "${PHYS_WEIGHT_BASE}" "${LEN_WEIGHT_BASE}" "${UNROLL_BASE}"
run_one "alpha_min_0.2" "${ALPHA_START_BASE}" "0.2" "${PHYS_WEIGHT_BASE}" "${LEN_WEIGHT_BASE}" "${UNROLL_BASE}"

# phys_weight sweep
run_one "phys_5e-4" "${ALPHA_START_BASE}" "${ALPHA_MIN_BASE}" "5e-4" "${LEN_WEIGHT_BASE}" "${UNROLL_BASE}"
run_one "phys_1e-3" "${ALPHA_START_BASE}" "${ALPHA_MIN_BASE}" "1e-3" "${LEN_WEIGHT_BASE}" "${UNROLL_BASE}"

# len_weight sweep
run_one "len_0" "${ALPHA_START_BASE}" "${ALPHA_MIN_BASE}" "${PHYS_WEIGHT_BASE}" "0" "${UNROLL_BASE}"
run_one "len_1e-2" "${ALPHA_START_BASE}" "${ALPHA_MIN_BASE}" "${PHYS_WEIGHT_BASE}" "1e-2" "${UNROLL_BASE}"

# unroll_steps sweep
run_one "unroll_3" "${ALPHA_START_BASE}" "${ALPHA_MIN_BASE}" "${PHYS_WEIGHT_BASE}" "${LEN_WEIGHT_BASE}" "3"
run_one "unroll_8" "${ALPHA_START_BASE}" "${ALPHA_MIN_BASE}" "${PHYS_WEIGHT_BASE}" "${LEN_WEIGHT_BASE}" "8"

echo "[done] summary -> ${SUMMARY_CSV}"
