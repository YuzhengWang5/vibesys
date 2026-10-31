"""Lightweight teacher-forced production logprob qualification against pinned HF."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from transformers import AutoTokenizer

from vs_bench.logprobs import LogprobTolerance, TokenLogprobs, compare_logprobs


class Contract(BaseModel):
    """Reject unknown external fields instead of interpreting a partial result."""

    model_config = ConfigDict(extra="forbid")


class ProbeReply(Contract):
    """One top-five observation per requested causal position."""

    positions: list[StrictInt]
    rows: list[TokenLogprobs]

    @model_validator(mode="after")
    def aligned(self) -> ProbeReply:
        if len(self.positions) != len(self.rows):
            raise ValueError("positions and rows counts differ")
        return self


class ReferenceCase(ProbeReply):
    """An independently computed HF cached-forward schedule."""

    name: str
    input_ids: list[StrictInt]
    prefill_length: int = Field(gt=0)
    decode_chunk_sizes: list[StrictInt]

    @model_validator(mode="after")
    def schedule(self) -> ReferenceCase:
        if any(size <= 0 for size in self.decode_chunk_sizes):
            raise ValueError("decode_chunk_sizes must be positive")
        if self.prefill_length + sum(self.decode_chunk_sizes) != len(self.input_ids):
            raise ValueError("teacher-forced schedule must consume all input_ids")
        if self.positions != list(range(self.prefill_length - 1, len(self.input_ids))):
            raise ValueError("reference positions do not match the teacher-forced schedule")
        return self


class ReferenceBundle(Contract):
    """Immutable reference and independently calibrated policy, shared across evaluations."""

    schema_version: Literal[1]
    model: Literal["Qwen/Qwen3-32B"]
    vocabulary: int = Field(ge=5)
    config_sha256: str
    weights_index_sha256: str
    reference: str
    torch: str
    transformers: str
    tolerance: LogprobTolerance
    calibration: dict
    cases: list[ReferenceCase] = Field(min_length=1)


def check(args: argparse.Namespace) -> int:
    """Use the same fixed token prefixes in every engine; never compare diverging generations."""
    here = Path(__file__).resolve().parent
    bundle = ReferenceBundle.model_validate_json((here / "reference_logprobs.json").read_bytes())
    for filename, expected in (
        ("config.json", bundle.config_sha256),
        ("model.safetensors.index.json", bundle.weights_index_sha256),
    ):
        actual = hashlib.sha256((Path(args.tokenizer_path) / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"reference model identity mismatch: {filename}")
    reference = []
    candidate = []
    with httpx.Client(timeout=180) as client:
        for case in bundle.cases:
            response = client.post(
                args.url.rstrip("/") + "/v1/logprobs",
                json={
                    "input_ids": case.input_ids,
                    "prefill_length": case.prefill_length,
                    "decode_chunk_sizes": case.decode_chunk_sizes,
                    "reference_token_ids": [row.top_token_ids[0] for row in case.rows],
                },
            )
            response.raise_for_status()
            reply = ProbeReply.model_validate(response.json())
            if reply.positions != case.positions:
                raise ValueError(f"{case.name}: missing, reordered or incorrect positions")
            if any(token >= bundle.vocabulary for row in reply.rows for token in row.top_token_ids):
                raise ValueError(f"{case.name}: out-of-vocabulary top-five ID")
            reference.extend(case.rows)
            candidate.extend(reply.rows)
        # A short production stream check catches empty/echo/protocol-only servers.
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Fix: def add(a, b): return a - b\nReturn only code."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prediction in ("", "def add(a, b):\n    return a + b\n"):
            generated, fragments, done = [], [], False
            with client.stream(
                "POST",
                args.url.rstrip("/") + "/v1/completions",
                json={
                    "prompt": prompt,
                    "prompt_is_preformatted": True,
                    "prediction": {"type": "content", "content": prediction},
                    "max_tokens": 8,
                    "temperature": 0,
                    "stream": True,
                    "return_token_ids": True,
                },
            ) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    event = json.loads(payload)
                    if "error" in event:
                        raise ValueError(f"completion error: {event['error']}")
                    choice = event["choices"][0]
                    ids = choice["token_ids"]
                    if not isinstance(ids, list) or any(
                        type(token) is not int or not 0 <= token < bundle.vocabulary
                        for token in ids
                    ):
                        raise ValueError(
                            "completion token_ids must be valid integer vocabulary IDs"
                        )
                    generated.extend(ids)
                    fragments.append(choice.get("text", ""))
            if not done or not 1 <= len(generated) <= 8:
                raise ValueError("completion must finish with 1..8 generated token IDs and [DONE]")
            text = tokenizer.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if text != "".join(fragments):
                raise ValueError("completion text disagrees with emitted token IDs")
    report = compare_logprobs(reference, candidate, bundle.tolerance)
    print(
        json.dumps(
            {
                "passed": report.passed,
                "reference": bundle.reference,
                "tolerance": bundle.tolerance.model_dump(),
                "calibration": bundle.calibration,
                "comparison": report.model_dump(),
            }
        ),
        flush=True,
    )
    return 0 if report.passed else 1


def main() -> int:
    """Managed mode starts only the candidate; HF observations are prepared once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed", action="store_true")
    parser.add_argument("--tokenizer-path", default="/model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if not args.managed:
        return check(args)
    here = Path(__file__).resolve()
    return subprocess.run(
        [
            sys.executable,
            str(here.parent.parent / "managed_evaluation.py"),
            "--",
            sys.executable,
            str(here),
            "--tokenizer-path",
            args.tokenizer_path,
            "--seed",
            str(args.seed),
            "--url",
            args.url,
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
