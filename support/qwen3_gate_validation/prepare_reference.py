"""Operator-only HF/vLLM calibration; outputs become immutable evaluator inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
import vllm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from vs_bench.logprobs import LogprobTolerance, TokenLogprobs, compare_logprobs, summarize_logits

SOURCES = (
    ("python-short", 32, "def add(a, b):\n    return a - b\n"),
    ("cpp-medium", 128, "int sum(int a,int b){return a-b;}\n"),
    (
        "python-long",
        256,
        "def search(xs, x):\n    for i,v in enumerate(xs):\n        if v==x: return i\n",
    ),
)
CONTINUATION_LENGTH = 8
DEFAULT_TOLERANCE = LogprobTolerance()


def prepare_hf(model_dir: Path, output: Path) -> None:
    """Cover short/medium/long prefill, single-token decode, chunk verification and reset."""
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    cases = []
    with torch.inference_mode():
        for name, length, text in SOURCES:
            ids = tokenizer.encode(text * (length + 8), add_special_tokens=False)[: length + 8]
            for chunks in ([1] * 8, [1, 4, 1, 2]):
                result = model(torch.tensor([ids[:length]], device="cuda"), use_cache=True)
                cache, cursor = result.past_key_values, length
                rows = [result.logits[0, -1:].float().cpu()]
                for size in chunks:
                    result = model(
                        torch.tensor([ids[cursor : cursor + size]], device="cuda"),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    rows.append(result.logits[0].float().cpu())
                    cache, cursor = result.past_key_values, cursor + size
                values = torch.cat(rows).numpy()
                summary = summarize_logits(values, values.argmax(axis=1).tolist())
                cases.append(
                    {
                        "name": f"{name}-{'decode' if len(chunks) == CONTINUATION_LENGTH else 'chunk'}",
                        "input_ids": ids,
                        "prefill_length": length,
                        "decode_chunk_sizes": chunks,
                        "positions": list(range(length - 1, len(ids))),
                        "rows": [row.model_dump() for row in summary],
                    }
                )
    bundle = {
        "schema_version": 1,
        "model": "Qwen/Qwen3-32B",
        "vocabulary": AutoConfig.from_pretrained(model_dir).vocab_size,
        "config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
        "weights_index_sha256": hashlib.sha256(
            (model_dir / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "reference": "huggingface-eager-bfloat16-teacher-forced",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tolerance": LogprobTolerance().model_dump(),
        "calibration": {"status": "pending"},
        "cases": cases,
    }
    output.write_text(json.dumps(bundle, indent=2) + "\n")


def calibrate_vllm(model_dir: Path, output: Path) -> None:
    """Query HF top-one as the forced next token, even if outside vLLM's top five."""
    bundle = json.loads(output.read_text())
    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=False,
    )
    prompts, reference = [], []
    for case in bundle["cases"]:
        for position, row in zip(case["positions"], case["rows"], strict=True):
            expected = TokenLogprobs.model_validate(row)
            reference.append(expected)
            prompts.append(
                {
                    "prompt_token_ids": [
                        *case["input_ids"][: position + 1],
                        expected.top_token_ids[0],
                    ]
                }
            )
    results = llm.generate(
        prompts, SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=5), use_tqdm=False
    )
    candidate = []
    for result, expected in zip(results, reference, strict=True):
        observed = result.prompt_logprobs[-1]
        ordered = sorted(observed.items(), key=lambda item: (-item[1].logprob, item[0]))[:5]
        candidate.append(
            TokenLogprobs(
                top_token_ids=[token for token, _ in ordered],
                top_logprobs=[value.logprob for _, value in ordered],
                reference_token_logprob=observed[expected.top_token_ids[0]].logprob,
            )
        )
    report = compare_logprobs(reference, candidate, LogprobTolerance(max_reference_top1_delta=100))
    if report.mutual_top5_rate < DEFAULT_TOLERANCE.min_mutual_top5_rate:
        message = f"vLLM control fails mutual top-five gate: {report}"
        raise ValueError(message)
    error = report.max_reference_top1_delta
    margin = DEFAULT_TOLERANCE.max_reference_top1_delta
    limit = margin if error <= margin else error + margin
    bundle["tolerance"] = LogprobTolerance(max_reference_top1_delta=limit).model_dump()
    bundle["calibration"] = {
        "status": "measured",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "positions": report.positions,
        "mutual_top5_rate": report.mutual_top5_rate,
        "max_reference_top1_delta": error,
        "selected_limit": limit,
        "rule": "0.2 if vLLM max <= 0.2, otherwise vLLM max + 0.2",
    }
    output.write_text(json.dumps(bundle, indent=2) + "\n")
    sys.stdout.write(json.dumps(bundle["calibration"]) + "\n")


def main() -> None:
    """Run one independent calibration process; HF and vLLM never share GPU residency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hf", "vllm"))
    parser.add_argument("--model-dir", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "hf":
        prepare_hf(args.model_dir, args.output)
    else:
        calibrate_vllm(args.model_dir, args.output)


if __name__ == "__main__":
    main()
