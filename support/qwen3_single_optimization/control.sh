#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$here/source.txt" ]]; then
    IFS= read -r source < "$here/source.txt"
else
    source="$(cd -- "$here/../.." && pwd)"
fi
python=/mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python
paths=("$here" "$source/support" "$source/src" "$source/packaging" "$source")
for path in "$source"/libs/*/src "$source"/sdk/*/src; do
    [[ ! -d "$path" ]] || paths+=("$path")
done
export PYTHONPATH
PYTHONPATH="$(IFS=:; echo "${paths[*]}")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export QWEN3_SOURCE="$source"
exec "$python" -m qwen3_single_optimization.controller "$@"
