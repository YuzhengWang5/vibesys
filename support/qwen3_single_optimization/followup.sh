#!/usr/bin/env bash
set -euo pipefail
source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
paths=("$source/support" "$source/src" "$source/packaging" "$source")
for path in "$source"/libs/*/src "$source"/sdk/*/src; do
  [[ ! -d "$path" ]] || paths+=("$path")
done
export PYTHONPATH
PYTHONPATH="$(IFS=:; echo "${paths[*]}")"
export QWEN3_SOURCE="$source" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec /mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python -m qwen3_single_optimization.followup "$@"
