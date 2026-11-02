"""The trusted reference cannot choose a smaller workload or a looser policy."""

import copy
import runpy
from pathlib import Path

import pytest
from pydantic import ValidationError

CHECKER = Path(
    "examples/model-serving/qwen3-32b-code-edit/.vibesys/tasks/default/accuracy_checker/checker.py"
)


def fixed_bundle() -> dict:
    row = {
        "top_token_ids": [0, 1, 2, 3, 4],
        "top_logprobs": [-2] * 5,
        "reference_token_logprob": -2,
    }
    cases = [
        {
            "name": f"{name}-{plan}",
            "input_ids": [0] * (length + 50),
            "prefill_length": length,
            "decode_chunk_sizes": chunks,
            "positions": list(range(length - 1, length + 50)),
            "rows": [copy.deepcopy(row) for _ in range(51)],
        }
        for name, length in (("python-short", 32), ("cpp-medium", 128), ("python-long", 256))
        for plan, chunks in (("decode", [1] * 50), ("chunk", [1, 4, 1, 2] * 6 + [1, 1]))
    ]
    return {
        "schema_version": 1,
        "model": "Qwen/Qwen3-32B",
        "vocabulary": 10,
        "config_sha256": "config",
        "weights_index_sha256": "weights",
        "reference": "huggingface-eager-bfloat16-teacher-forced",
        "torch": "version",
        "transformers": "version",
        "tolerance": {"top_k": 2, "min_mutual_top5_rate": 1, "max_reference_top1_delta": 0.02},
        "calibration": {"status": "fixed", "rule": "top2/100%/0.02, never calibrated upward"},
        "cases": cases,
    }


def test_reference_has_six_fixed_50_token_plans_and_306_positions() -> None:
    bundle_type = runpy.run_path(str(CHECKER))["ReferenceBundle"]
    reference = bundle_type.model_validate(fixed_bundle())
    assert sum(len(case.rows) for case in reference.cases) == 306
    assert all(sum(case.decode_chunk_sizes) == 50 for case in reference.cases)


@pytest.mark.parametrize("attack", ["cases", "length", "topk", "rate", "limit", "prefix"])
def test_reference_rejects_denominator_and_policy_tampering(attack: str) -> None:
    bundle = fixed_bundle()
    if attack == "cases":
        bundle["cases"].pop()
    elif attack == "length":
        case = bundle["cases"][0]
        case["decode_chunk_sizes"] = [1] * 8
        case["input_ids"] = case["input_ids"][:40]
        case["positions"] = list(range(31, 40))
        case["rows"] = case["rows"][:9]
    elif attack == "prefix":
        bundle["cases"][1]["input_ids"][0] = 1
    else:
        key, value = {
            "topk": ("top_k", 5),
            "rate": ("min_mutual_top5_rate", 0.95),
            "limit": ("max_reference_top1_delta", 0.02001),
        }[attack]
        bundle["tolerance"][key] = value
    bundle_type = runpy.run_path(str(CHECKER))["ReferenceBundle"]
    with pytest.raises(ValidationError):
        bundle_type.model_validate(bundle)
