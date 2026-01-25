#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/hhdisk/filtergen2_storage}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/migrate_storage.sh [--root PATH] [--target PATH] [--dry-run]

Moves large repo subdirs to TARGET_ROOT and replaces them with symlinks.
Edit MOVE_DIRS in this script to customize what gets moved.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --target)
      TARGET_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "Repo root not found at: $ROOT (missing pyproject.toml)"
  exit 1
fi

MOVE_DIRS=(
  "data/processed"
  "checkpoints"
  "outputs"
  "exports"
)

mkdir -p "$TARGET_ROOT"
export TMPDIR="${TMPDIR:-$TARGET_ROOT/tmp}"
mkdir -p "$TMPDIR"

copy_dir() {
  local src="$1"
  local dest="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$src"/ "$dest"/
  else
    cp -a "$src"/. "$dest"/
  fi
}

for rel in "${MOVE_DIRS[@]}"; do
  src="$ROOT/$rel"
  dest="$TARGET_ROOT/$rel"

  if [[ ! -e "$src" ]]; then
    echo "[skip] missing: $rel"
    continue
  fi
  if [[ -L "$src" ]]; then
    echo "[skip] already symlink: $rel"
    continue
  fi

  echo "[move] $src -> $dest"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  mkdir -p "$(dirname "$dest")"
  copy_dir "$src" "$dest"
  rm -rf "$src"
  mkdir -p "$(dirname "$src")"
  ln -s "$dest" "$src"
done

echo "Done."
