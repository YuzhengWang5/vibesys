"""原始 gate 的字节、参数与计时输出契约; 不加载模型或数据集。"""

import ast
import hashlib
import io
import json
import runpy
import tomllib
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

TASK = (
    Path(__file__).resolve().parents[2]
    / "examples/model-serving/qwen3-32b-code-edit/.vibesys/tasks/default"
)


def test_original_checker_is_the_exact_historical_git_blob() -> None:
    content = (TASK / "accuracy_checker/checker.py").read_bytes()
    header = f"blob {len(content)}\0".encode()
    assert hashlib.sha1(header + content, usedforsecurity=False).hexdigest() == (
        "ee67b5ff252584cfa22bc28c0b5e74181c63ce30"
    )
    assert hashlib.sha256(content).hexdigest() == (
        "716f21093c6da5c187fa9e9a56ab80f565c07a6d08d1184dd4a7055bb11c3c45"
    )


def test_original_generation_and_gate_defaults_are_unchanged() -> None:
    tree = ast.parse((TASK / "accuracy_checker/checker.py").read_text())
    defaults = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(keyword.value)
    assert {
        key: defaults[key]
        for key in (
            "--num-samples",
            "--warmup",
            "--max-tokens",
            "--temperature",
            "--seed",
            "--min-gold-rate",
            "--min-gold-similarity",
        )
    } == {
        "--num-samples": 10,
        "--warmup": 2,
        "--max-tokens": 512,
        "--temperature": 0,
        "--seed": None,
        "--min-gold-rate": 0.5,
        "--min-gold-similarity": 0.5,
    }


def test_accuracy_preserves_random_defaults_and_benchmark_is_50_3_512() -> None:
    manifest = tomllib.loads((TASK / "vibesys.input.toml").read_text())
    accuracy = manifest["accuracy"]["command"]
    assert accuracy == [
        "python",
        "${PROJECT_ROOT}/.vibesys/tasks/default/managed_evaluation.py",
        "--",
        "uv",
        "run",
        "python",
        "${PROJECT_ROOT}/.vibesys/tasks/default/accuracy_checker/checker.py",
        "--tokenizer-path",
        "/model",
    ]
    benchmark = manifest["benchmark"]["command"]
    for option, value in (
        ("--num-samples", "50"),
        ("--warmup", "3"),
        ("--max-tokens", "512"),
        ("--seed", "42"),
    ):
        assert benchmark[benchmark.index(option) + 1] == value
    assert manifest["benchmark"]["result"]["metric"] == "median_tok_per_sec"
    assert not (TASK / "accuracy_checker/launch.py").exists()
    assert not (TASK / "accuracy_checker/reference_logprobs.json").exists()


@given(duration=st.floats(min_value=0, max_value=100000, allow_nan=False))
def test_managed_timing_records_total_without_inventing_cold_start_costs(duration: float) -> None:
    wrapper = runpy.run_path(str(TASK / "managed_evaluation.py"))
    timing = wrapper["EvaluationTiming"](started=0, returncode=0)
    timing.launcher_to_health_seconds = duration / 4
    timing.client_seconds = duration / 2
    timing.client_returncode = 0
    timing.begin_cleanup()
    stream = io.StringIO()
    timing.emit(stream, finished=duration)
    marker, payload = stream.getvalue().split(" ", 1)
    record = json.loads(payload)
    assert marker == "__VIBESYS_MANAGED_TIMING__"
    assert record["phase"] == "completed"
    assert record["managed_seconds"] == duration
    assert record["client_subprocess_seconds"] == duration / 2
    assert record["startup_seconds"] == duration / 4
    assert record["dependency_cold_start_seconds"] is None
    assert record["model_load_seconds"] is None
    assert record["checker_seconds"] is None
    assert record["outer_environment_seconds"] is None


def synthetic_failure() -> None:
    raise ValueError("合成失败")


@pytest.mark.parametrize("phase", ["preflight", "startup", "client", "cleanup"])
@pytest.mark.parametrize("returncode", [None, 1, 143])
def test_timing_preserves_failed_phase_and_unknown_measurements(
    phase: str,
    returncode: int | None,
) -> None:
    wrapper = runpy.run_path(str(TASK / "managed_evaluation.py"))
    timing = wrapper["EvaluationTiming"](started=10, phase=phase, returncode=returncode)
    try:
        synthetic_failure()
    except ValueError:
        timing.begin_cleanup()
    stream = io.StringIO()
    timing.emit(stream, finished=15)
    record = json.loads(stream.getvalue().split(" ", 1)[1])
    assert record["phase"] == phase
    assert record["returncode"] == returncode
    assert record["client_subprocess_seconds"] is None
    assert record["startup_seconds"] is None
    assert record["managed_seconds"] == 5
