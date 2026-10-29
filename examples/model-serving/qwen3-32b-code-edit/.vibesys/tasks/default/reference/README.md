# Reference bundle — Qwen/Qwen3-32B

The pinned `OBJECTIVE.md` defines the only allowed optimization. This directory
provides model facts and an upstream reference, not a prescribed engine design.

## Files in this directory

| File | Description |
|:--|:--|
| `config.json` | `Qwen/Qwen3-32B` verifier config. |
| `reference.py` | `transformers.models.qwen3.modeling_qwen3` reference module. |
| `meta.json` | model id + pinned commit revision. |

The model weights themselves are not in this directory — populate `model/` with a `Qwen/Qwen3-32B` snapshot (e.g. via `huggingface-cli download` or a symlink to a shared cache) before running.

(`Qwen/Qwen3-32B` BF16, ~64 GiB across 17 safetensors shards.)

Disable thinking mode in the chat template (`enable_thinking=False`) — code edits at `max_tokens=512` would otherwise get eaten by Qwen3's `<think>...</think>` preamble.
