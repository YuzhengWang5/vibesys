#!/usr/bin/env bash
set -euo pipefail
log="$1"
shift
exec >> "$log" 2>&1
exec "$@"
