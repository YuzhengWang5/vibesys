"""Evaluate collected production results with unmodified Hugging Face Qwen3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

from vs_bench.logits import LogitTolerance, compare_logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    cases = json.loads((args.artifacts / "cases.json").read_text())
    reports = []
    passed = True
    with torch.inference_mode():
        for case in cases:
            ids = torch.tensor([case["input_ids"]], device="cuda")
            if "file" in case:
                # Match GEMM shapes: BF16 cached vs monolithic forwards can
                # differ materially even in an unmodified HF implementation.
                length = case["prefill_length"]
                result = model(ids[:, :length], use_cache=True)
                cache = result.past_key_values
                rows = [result.logits[0, -1:].float().cpu()]
                for size in case["decode_chunk_sizes"]:
                    result = model(
                        ids[:, length : length + size],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = result.past_key_values
                    rows.append(result.logits[0].float().cpu())
                    length += size
                output = torch.cat(rows).numpy()
                candidate = np.load(args.artifacts / case["file"], allow_pickle=False)
                report = compare_logits(output, candidate).model_dump()
            else:
                output = (
                    model(ids, use_cache=False).logits[0, case["positions"]].float().cpu().numpy()
                )
                chosen = np.asarray(case["generated"])
                regret = output.max(axis=1) - output[np.arange(len(chosen)), chosen]
                report = {
                    "passed": bool((regret <= LogitTolerance().max_winner_regret).all()),
                    "generated_tokens": len(chosen),
                    "max_winner_regret": float(regret.max()),
                }
                # Early termination is legal only when the model would emit EOS.
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else eos
                if len(chosen) < 16 and int(chosen[-1]) not in eos:
                    last = model(ids, use_cache=False).logits[0, -1].float()
                    eos_regret = float(last.max() - last[eos].max())
                    report["eos_regret"] = eos_regret
                    report["passed"] &= eos_regret <= LogitTolerance().max_winner_regret
            reports.append(report)
            passed &= report["passed"]
            print(json.dumps(report), flush=True)
    print(
        json.dumps(
            {
                "passed": passed,
                "reference": "huggingface-eager-bfloat16",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "tolerance": LogitTolerance().model_dump(),
                "cases": reports,
            }
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
