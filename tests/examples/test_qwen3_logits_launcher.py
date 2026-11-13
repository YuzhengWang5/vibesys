"""Cold empty candidates must fail before invoking dependency resolution."""

import runpy
from pathlib import Path

import pytest


def test_missing_candidate_fails_without_importing_gpu_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entrypoint = Path(
        "examples/model-serving/qwen3-32b-code-edit/"
        ".vibesys/tasks/default/accuracy_checker/launch.py"
    ).resolve()
    module = runpy.run_path(str(entrypoint))
    monkeypatch.chdir(tmp_path)
    assert module["main"]() == 1
    assert "candidate startup entrypoint is missing" in capsys.readouterr().err
