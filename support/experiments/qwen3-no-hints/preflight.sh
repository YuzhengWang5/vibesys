#!/usr/bin/env bash
set -euo pipefail

run_dir=$(realpath "$1")
project=$(realpath "$run_dir/project")
uv=$(realpath "$(command -v uv)")
docker run --rm --runtime runc \
    --name "qwen-no-hints-preflight-$(basename "$(dirname "$(dirname "$run_dir")")")" \
    --user "$(id -u):$(id -g)" --cap-drop ALL \
    -e HOME=/tmp -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
    -e UV_PROJECT_ENVIRONMENT=/tmp/preflight-venv -e UV_CACHE_DIR=/tmp/uv-cache \
    -e UV_PYTHON_DOWNLOADS=never -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$uv:/usr/local/bin/uv:ro" \
    -v "$project:/workspace" -v /mnt_ssd/data1/models/Qwen3-32B:/model:ro \
    -w /workspace --entrypoint bash nvcr.io/nvidia/pytorch:25.04-py3 -lc '
set -euo pipefail
uv run --no-dev --no-editable python - <<'"'"'PY'"'"'
import datasets, httpx, jinja2, transformers, vs_bench
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": "print(1)"}],
    enable_thinking=False, tokenize=True, add_generation_prompt=True,
)
assert tokens
print("dependency resolution and real Qwen3 chat template: OK")
PY
uv run --no-dev --no-editable python .vibesys/tasks/default/accuracy_checker/checker.py --help
uv run --no-dev --no-editable python .vibesys/tasks/default/benchmark/benchmark.py --help
' >"$run_dir/logs/preflight.log" 2>&1
