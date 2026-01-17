from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]


def _get_arg_value(args_list: Iterable[str], key: str) -> Optional[str]:
    found = None
    key_eq = f"{key}="
    for idx, tok in enumerate(args_list):
        if tok == key and idx + 1 < len(args_list):
            found = args_list[idx + 1]
        elif tok.startswith(key_eq):
            found = tok.split("=", 1)[1]
    return found


def _has_flag(args_list: Iterable[str], flag: str) -> bool:
    return any(tok == flag or tok.startswith(f"{flag}=") for tok in args_list)


def _ensure_flag(args_list: list[str], flag: str) -> None:
    if not _has_flag(args_list, flag):
        args_list.append(flag)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = Path(f"{path}.{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free path for {path}")


def _load_target_wave(output_dir: Path) -> Optional[str]:
    cfg_path = output_dir / "input_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None
    target = cfg.get("target_wave")
    if target in ("ideal", "real"):
        return str(target)
    return None


def _run_and_log(cmd: list[str], log_path: Path, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = _unique_path(log_path)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {shlex.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def _parse_epoch_metrics(line: str) -> Optional[Dict[str, float]]:
    if "avg " not in line:
        return None
    matches = re.findall(r"(\w+)=([0-9.eE+-]+)", line)
    if not matches:
        return None
    metrics = {}
    for key, value in matches:
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics if metrics else None


def _select_best_epoch(log_path: Path, *, metric: str, mode: str) -> Optional[int]:
    best_epoch = None
    best_val = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("[epoch "):
                continue
            if "avg " not in line:
                continue
            m = re.search(r"\[epoch (\d+)\]", line)
            if not m:
                continue
            epoch = int(m.group(1))
            metrics = _parse_epoch_metrics(line)
            if not metrics or metric not in metrics:
                continue
            val = metrics[metric]
            if best_val is None:
                best_val = val
                best_epoch = epoch
                continue
            if mode == "min" and val < best_val:
                best_val = val
                best_epoch = epoch
            elif mode == "max" and val > best_val:
                best_val = val
                best_epoch = epoch
    return best_epoch


def _latest_epoch_dir(output_dir: Path) -> Optional[Path]:
    candidates = []
    if not output_dir.exists():
        return None
    for sub in output_dir.iterdir():
        if not sub.is_dir():
            continue
        if not sub.name.startswith("epoch_"):
            continue
        try:
            num = int(sub.name.split("_", 1)[1])
        except ValueError:
            continue
        ckpt = sub / "pytorch_model.bin"
        if ckpt.exists():
            candidates.append((num, sub))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bilevel model and auto-evaluate after training.")
    p.add_argument("--train-args", required=True, help="Quoted args string for scripts/train_bilevel.py.")
    p.add_argument("--eval-args", required=True, help="Quoted args string for scripts/eval_bilevel.py.")
    p.add_argument("--train-script", type=Path, default=Path("scripts/train_bilevel.py"))
    p.add_argument("--eval-script", type=Path, default=Path("scripts/eval_bilevel.py"))
    p.add_argument("--log-dir", type=Path, help="Directory to store train/eval logs (default: <output>/logs).")
    p.add_argument("--best-metric", default="phys", help="Metric name from epoch avg line (default: phys).")
    p.add_argument("--best-mode", choices=["min", "max"], default="min", help="Optimize for min or max.")
    p.add_argument("--no-best", dest="select_best", action="store_false", help="Skip best-epoch selection.")
    p.set_defaults(select_best=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_args = shlex.split(args.train_args)
    eval_args = shlex.split(args.eval_args)

    _ensure_flag(train_args, "--log-epoch-metrics")

    output_dir = _get_arg_value(train_args, "--output")
    if output_dir is None:
        output_dir = "checkpoints/bilevel"
    output_dir = Path(output_dir)

    log_dir = args.log_dir or (output_dir / "logs")
    train_log = log_dir / "train.log"
    eval_log = log_dir / "eval.log"

    train_cmd = [sys.executable, str(args.train_script)] + train_args
    rc = _run_and_log(train_cmd, train_log, cwd=ROOT)
    if rc != 0:
        raise SystemExit(f"Training failed with exit code {rc}. See {train_log}")

    ckpt_path = None
    if args.select_best:
        best_epoch = _select_best_epoch(train_log, metric=str(args.best_metric), mode=str(args.best_mode))
        if best_epoch is not None:
            ckpt_path = output_dir / f"epoch_{best_epoch}"
    if ckpt_path is None:
        ckpt_path = _latest_epoch_dir(output_dir)
    if ckpt_path is None:
        raise SystemExit(f"No epoch checkpoint found under {output_dir}")

    if not _has_flag(eval_args, "--ckpt"):
        eval_args = eval_args + ["--ckpt", str(ckpt_path)]

    if not _has_flag(eval_args, "--output"):
        eval_out = log_dir / f"eval_results_epoch_{ckpt_path.name.split('_')[-1]}.json"
        eval_args = eval_args + ["--output", str(eval_out)]

    if not _has_flag(eval_args, "--target-wave"):
        target_wave = _load_target_wave(output_dir)
        if target_wave:
            eval_args = eval_args + ["--target-wave", str(target_wave)]

    eval_cmd = [sys.executable, str(args.eval_script)] + eval_args
    rc = _run_and_log(eval_cmd, eval_log, cwd=ROOT)
    if rc != 0:
        raise SystemExit(f"Eval failed with exit code {rc}. See {eval_log}")


if __name__ == "__main__":
    main()
