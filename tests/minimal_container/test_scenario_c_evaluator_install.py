"""Cold evaluation installs must not build inside the read-only task."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest
from tests.support import run_test_command

pytestmark = pytest.mark.minimal_container
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("directory", ["candidate", "candidate with spaces"])
def test_frozen_evaluator_installs_with_a_readonly_task(tmp_path: Path, directory: str) -> None:
    candidate = tmp_path / directory
    task = candidate / ".vibesys/tasks/default"
    vendor = task / "vendor/vs-bench"
    shutil.copytree(ROOT / "sdk/vs-bench", vendor, ignore=shutil.ignore_patterns("*.egg-info"))
    (candidate / "pyproject.toml").write_text(
        '[project]\nname = "readonly-evaluator"\nversion = "0.1.0"\n'
        'requires-python = ">=3.12"\ndependencies = ["vs-bench"]\n'
        "[tool.uv]\npackage = false\n[tool.uv.sources]\n"
        'vs-bench = {path = ".vibesys/tasks/default/vendor/vs-bench"}\n'
    )
    run_test_command(["uv", "lock", "--project", str(candidate)], check=True, timeout=180)
    before = tomllib.loads((candidate / "uv.lock").read_text())
    run_test_command(
        ["bash", str(ROOT / "support/scenario-c-repro/freeze-evaluator.sh"), str(candidate)],
        check=True,
        timeout=180,
    )
    after = tomllib.loads((candidate / "uv.lock").read_text())
    assert [(p["name"], p["version"]) for p in before["package"]] == [
        (p["name"], p["version"]) for p in after["package"]
    ]
    wheels = list((task / "vendor/wheels").glob("*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as wheel:
        for source in (vendor / "src/vs_bench").rglob("*.py"):
            assert wheel.read(str(source.relative_to(vendor / "src"))) == source.read_bytes()
    uv = shutil.which("uv")
    assert uv is not None
    run_test_command(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "UV_CACHE_DIR=/tmp/cold-uv-cache",
            "-e",
            "UV_PROJECT_ENVIRONMENT=/tmp/evaluator",
            "-v",
            f"{candidate}:/workspace",
            "-v",
            f"{task}:/workspace/.vibesys/tasks/default:ro",
            "-v",
            f"{Path(uv).resolve()}:/usr/local/bin/uv:ro",
            "-w",
            "/workspace",
            "--entrypoint",
            "/usr/local/bin/uv",
            "nvcr.io/nvidia/pytorch:25.04-py3",
            "run",
            "--frozen",
            "--no-dev",
            "--python",
            "/usr/bin/python3",
            "python",
            "-c",
            "import vs_bench",
        ],
        check=True,
        timeout=180,
    )
