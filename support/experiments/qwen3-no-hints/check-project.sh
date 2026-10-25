#!/usr/bin/env bash
set -euo pipefail

project=$(realpath "$1")
# Run records and machine-local state belong to the framework, not candidate source.
changes=$(git -C "$project" status --porcelain --untracked-files=all -- \
    . ':(exclude).vibesys/state/runs' ':(exclude).vibesys/state/local')
if test -n "$changes"; then
    printf 'ERROR: 候选源码存在未提交改动：%s；请先建立候选 checkpoint。\n%s\n' \
        "$project" "$changes" >&2
    exit 1
fi
