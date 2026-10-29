"""Qualify production logits and streamed greedy decisions against native HF.

Managed mode first collects candidate results, stops the candidate, then loads
HF. Thus a single 80GB GPU never needs two copies of Qwen3-32B.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, StrictInt, TypeAdapter, model_validator
from transformers import AutoConfig, AutoTokenizer

CHUNKS = [1, 1, 4, 1, 8, 1]
SOURCES = [
    "Fix this Python program. Return only code.\ndef add(a, b):\n    return a - b\n",
    "Fix this C++ program:\nint sum(int a,int b){return a-b;}\n",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
    "function unique(xs) { return [...new Set(xs)]; }\n",
    "def search(xs, x):\n    for i, value in enumerate(xs):\n        if value == x:\n            return i\n    return -1\n",
    "SELECT name FROM users WHERE active = 1 ORDER BY name;\n",
]


class ProbeReply(BaseModel):
    """The candidate must echo positions, not reorder or omit vocabulary rows."""

    model_config = ConfigDict(extra="forbid")

    positions: list[StrictInt]
    logits: list[list[float]]

    @model_validator(mode="after")
    def aligned(self) -> ProbeReply:
        if len(self.positions) != len(self.logits):
            raise ValueError("positions and logits row counts differ")
        return self


def collect(args: argparse.Namespace) -> None:
    """Collect both diagnostic forwards and actual production SSE completions."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    vocabulary = AutoConfig.from_pretrained(args.tokenizer_path).vocab_size
    rng = np.random.default_rng(args.seed)
    cases = []
    with httpx.Client(timeout=180) as client:
        for index, (source, length) in enumerate(
            zip(SOURCES, [32, 127, 256, 511, 1024, 65], strict=True)
        ):
            text = source * (length + 1)
            ids = tokenizer.encode(text, add_special_tokens=False)[: length + sum(CHUNKS)]
            if index == len(SOURCES) - 1:
                ids = rng.integers(0, tokenizer.vocab_size, size=len(ids)).tolist()
            positions = list(range(length - 1, len(ids)))
            for schedule_index, chunks in enumerate(([1] * sum(CHUNKS), CHUNKS)):
                response = client.post(
                    args.url.rstrip("/") + "/v1/logits",
                    json={"input_ids": ids, "prefill_length": length, "decode_chunk_sizes": chunks},
                )
                response.raise_for_status()
                reply = ProbeReply.model_validate(response.json())
                if reply.positions != positions:
                    raise ValueError(f"case {index}: positions {reply.positions} != {positions}")
                array = np.asarray(reply.logits, dtype=np.float32)
                if array.shape != (len(positions), vocabulary):
                    raise ValueError(f"case {index}: invalid full-vocabulary shape {array.shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"case {index}: nonfinite logits")
                filename = f"candidate-{index}-{schedule_index}.npy"
                np.save(args.artifacts / filename, array, allow_pickle=False)
                cases.append({"input_ids": ids, "positions": positions, "file": filename})
            if index < 3:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": source}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prediction in ("", source):
                    fragments = []
                    generated = []
                    done = False
                    with client.stream(
                        "POST",
                        args.url.rstrip("/") + "/v1/completions",
                        json={
                            "prompt": prompt,
                            "prompt_is_preformatted": True,
                            "prediction": {"type": "content", "content": prediction},
                            "max_tokens": 16,
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
                            fragments.append(event["choices"][0].get("text", ""))
                            generated.extend(
                                TypeAdapter(list[StrictInt]).validate_python(
                                    event["choices"][0]["token_ids"]
                                )
                            )
                    if not done:
                        raise ValueError("completion stream is missing [DONE]")
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    if not generated or len(generated) > 16:
                        raise ValueError(f"invalid generated token count: {len(generated)}")
                    if any(token < 0 or token >= vocabulary for token in generated):
                        raise ValueError("completion contains out-of-vocabulary token IDs")
                    decoded = tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if decoded != "".join(fragments):
                        raise ValueError("streamed text does not match emitted token_ids")
                    cases.append(
                        {
                            "input_ids": prompt_ids + generated,
                            "positions": list(
                                range(len(prompt_ids) - 1, len(prompt_ids) + len(generated) - 1)
                            ),
                            "generated": generated,
                        }
                    )
    (args.artifacts / "cases.json").write_text(json.dumps(cases), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--managed", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--tokenizer-path", default="/model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.collect:
        if args.artifacts is None:
            parser.error("--collect requires --artifacts")
        collect(args)
        return 0
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="vibesys-logits-") as directory:
        artifacts = Path(directory)
        result = subprocess.run(
            [
                sys.executable,
                str(here.parent / "managed_evaluation.py"),
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                "--collect",
                "--artifacts",
                str(artifacts),
                "--tokenizer-path",
                args.tokenizer_path,
                "--seed",
                str(args.seed),
            ],
            check=False,
        )
        if result.returncode:
            return result.returncode
        return subprocess.run(
            [
                sys.executable,
                str(here / "reference_logits.py"),
                "--artifacts",
                str(artifacts),
                "--model-dir",
                args.tokenizer_path,
            ],
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
