#!/usr/bin/env bash
set -euo pipefail

CANDIDATE="$(realpath "$1")"
VENDOR="$CANDIDATE/.vibesys/tasks/default/vendor"
test ! -e "$VENDOR/wheels"

# Setuptools writes into its source tree; evaluation mounts the task read-only.
SOURCE_DATE_EPOCH=315532800 uv build --wheel \
    --out-dir "$VENDOR/wheels" "$VENDOR/vs-bench"
WHEELS=("$VENDOR/wheels/"*.whl)
test "${#WHEELS[@]}" -eq 1
cd "$CANDIDATE"
uv add --offline --no-sync \
    ".vibesys/tasks/default/vendor/wheels/$(basename "${WHEELS[0]}")"
