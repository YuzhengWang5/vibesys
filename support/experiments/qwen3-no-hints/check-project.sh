#!/usr/bin/env bash
set -euo pipefail

project=$1
mode=$2
case "$mode" in
    restart)
        changes=$(git -C "$project" status --porcelain)
        ;;
    resume)
        # Resume state is framework-owned, but tracked changes anywhere still fail.
        changes=$(git -C "$project" status --porcelain --untracked-files=no)
        untracked=$(git -C "$project" ls-files --others --exclude-standard -- \
            . ':(exclude).vibesys/state/runs/')
        if test -n "$untracked"; then
            printf '候选仓库有未跟踪的非运行状态文件，请先检查：\n%s\n' "$untracked" >&2
            exit 1
        fi
        ;;
    *)
        printf '无效的候选检查模式：%s\n' "$mode" >&2
        exit 2
        ;;
esac
if test -n "$changes"; then
    printf '候选仓库有未提交变更，请先检查：\n%s\n' "$changes" >&2
    exit 1
fi
