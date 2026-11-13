"""Code-edit sends prompt-only requests and scores real aggregate concurrency."""

from __future__ import annotations

import argparse
import asyncio
import json
import runpy
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vs_sim.api import yield_now

if TYPE_CHECKING:
    from vs_sim.api.testing import Sim

BENCHMARK = Path(
    "examples/model-serving/qwen3-32b-code-edit/.vibesys/tasks/default/benchmark/benchmark.py"
)


class CharacterTokenizer:
    """Deterministic tokenizer for testing the client's counting contract."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode())

    def apply_chat_template(self, messages: list[dict], **_kwargs: object) -> str:
        return "\n".join(message["content"] for message in messages)


class CompletionTransport(httpx.AsyncBaseTransport):
    """In-memory HTTP service, with yielding requests but no real sockets."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.active = 0
        self.peak = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await yield_now()
        self.active -= 1
        return httpx.Response(
            200,
            text='data: {"choices":[{"text":"fixed","finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        )


async def test_requests_overlap_without_out_of_band_drafts(sim: Sim) -> None:
    benchmark = runpy.run_path(str(BENCHMARK))
    samples = [
        {
            "unique_id": str(index),
            "language": "python3",
            "bug_type": "",
            "incorrect_code": f"original_{index}",
            "gold_code": "secret_gold_not_for_server",
        }
        for index in range(7)
    ]
    args = argparse.Namespace(
        url="http://in-memory",
        endpoint="/v1/completions",
        languages="python3",
        warmup=3,
        num_samples=7,
        max_tokens=8,
        temperature=0,
        model="",
        print_stream=False,
        seed=42,
        tokenizer_path="unused",
        request_timeout=3600,
        max_dispatch_lag=1,
    )
    transport = CompletionTransport()
    report = await benchmark["run_level"](
        args,
        samples,
        CharacterTokenizer(),
        100000,
        transport=transport,
        clock=sim.clock.now,
        sleep=sim.clock.sleep,
    )
    assert len(transport.bodies) == 7
    for sample, body in zip(samples, transport.bodies, strict=True):
        assert set(body) == {
            "prompt",
            "max_tokens",
            "temperature",
            "stream",
            "prompt_is_preformatted",
        }
        assert sample["incorrect_code"] in body["prompt"]
        assert sample["gold_code"] not in json.dumps(body)
    assert report["valid"]
    assert report["num_completed"] == 4
    assert report["total_output_tokens"] == 20
    assert report["throughput_tok_per_sec"] == 20 / report["actual_duration_sec"]
    assert report["median_chunks"] == 1
    assert report["request_rate"] == 100000
    assert report["arrival_process"] == "poisson"


async def test_slow_responses_do_not_throttle_arrivals(sim: Sim) -> None:
    benchmark = runpy.run_path(str(BENCHMARK))
    count = 40
    offsets = benchmark["arrival_offsets"](count, 16, 42)
    started: list[float] = []
    completed: list[float] = []
    release = asyncio.Event()

    async def send(_index: int) -> dict:
        started.append(sim.clock.now())
        if len(started) == count:
            release.set()
        await release.wait()
        await yield_now()
        completed.append(sim.clock.now())
        return {"error": None, "output_tokens": 5}

    results, duration, load = await benchmark["dispatch_open_loop"](
        count,
        16,
        42,
        send,
        clock=sim.clock.now,
        sleep=sim.clock.sleep,
    )
    assert len(results) == count
    assert [value - started[0] for value in started] == pytest.approx(offsets)
    assert min(completed) >= started[-1]
    assert load["observed_peak_in_flight"] == count
    assert load["max_dispatch_lag_sec"] == pytest.approx(0)
    assert duration >= offsets[-1]


@given(
    rate=st.floats(min_value=0.01, max_value=1000, allow_nan=False),
    count=st.integers(min_value=2, max_value=100),
)
def test_seeded_arrivals_scale_with_rate(rate: float, count: int) -> None:
    benchmark = runpy.run_path(str(BENCHMARK))
    offsets = benchmark["arrival_offsets"](count, rate, 42)
    faster = benchmark["arrival_offsets"](count, rate * 2, 42)
    assert offsets[0] == 0
    assert all(a < b for a, b in pairwise(offsets))
    assert faster == pytest.approx([offset / 2 for offset in offsets])


@given(
    tokens=st.lists(st.integers(min_value=1, max_value=4096), min_size=4, max_size=30),
    duration=st.floats(min_value=0.01, max_value=10000, allow_nan=False),
)
def test_capacity_is_total_tokens_over_wall_time(tokens: list[int], duration: float) -> None:
    benchmark = runpy.run_path(str(BENCHMARK))
    results = [{"error": None, "output_tokens": count} for count in tokens]
    report = benchmark["summarize_capacity"](results, duration, 4)
    assert report["throughput_tok_per_sec"] == sum(tokens) / duration
    for bad in ({"error": "HTTP 500", "output_tokens": 10}, {"error": None, "output_tokens": 0}):
        rejected = benchmark["summarize_capacity"]([*results, bad], duration, 4)
        assert not rejected["valid"]
        assert rejected["throughput_tok_per_sec"] is None


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_measurement_duration_fails(duration: float) -> None:
    benchmark = runpy.run_path(str(BENCHMARK))
    with pytest.raises(ValueError, match="duration"):
        benchmark["summarize_capacity"]([], duration, 2)
