"""Exercise the external experiment inspector through its shell entry point."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from tests.support import run_test_command

from vs_sim.api.testing import HANG_GUARD_S

_INSPECTOR = Path(__file__).resolve().parents[2] / "support/experiments/qwen3-no-hints/inspect.sh"
_SCRIPTS = _INSPECTOR.parent


@pytest.mark.parametrize("version", ["3.11", "3.12", "3.13", "3.14"])
def test_example_python_support_is_compatible_with_its_sdk(version: str) -> None:
    root = Path(__file__).resolve().parents[2]
    example = root / "examples/model-serving/qwen3-32b-code-edit/pyproject.toml"
    sdk = root / "sdk/vs-bench/pyproject.toml"
    example_python = SpecifierSet(tomllib.loads(example.read_text())["project"]["requires-python"])
    sdk_python = SpecifierSet(tomllib.loads(sdk.read_text())["project"]["requires-python"])

    assert version not in example_python or version in sdk_python


def _inspect(root: Path, running: str) -> dict[str, object]:
    result = run_test_command(
        ["bash", str(_INSPECTOR), str(root), "--now", "12345", "--running", running],
        capture_output=True,
        text=True,
        check=True,
        timeout=HANG_GUARD_S,
    )
    status = json.loads(result.stdout)
    assert json.loads((root / "logs/supervisor-health.json").read_text()) == status
    return status


def _events(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "logs/vibesys-state/projects/project/runs/run/logs/core-events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_inspector_ignores_empty_baseline_but_counts_a_real_evaluation_failure(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "run_id": "run",
            "sequence": index,
            "type": "gate_finished",
            "status": "failed",
            "data": {"gate": "accuracy", "output_tail": output},
        }
        for index, output in enumerate(
            [
                "candidate startup entrypoint is missing: /workspace/run_server.sh",
                "cannot normalize a relative path beyond the base directory",
                None,
            ],
            start=1,
        )
    ]
    path = _events(tmp_path, rows)
    with path.open("a") as stream:
        stream.write('{"type": "partially written')

    status = _inspect(tmp_path, "yes")

    assert status["failure_count"] == 2
    assert status["running"] is True
    assert status["terminal"] is None


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_inspector_reads_trusted_outcome_not_process_liveness(tmp_path: Path, outcome: str) -> None:
    _events(
        tmp_path,
        [{"run_id": "run", "sequence": 1, "type": "run_failed", "status": "failed"}],
    )
    publications = tmp_path / "project/.vibesys/state/runs/run/publications/publications.json"
    publications.parent.mkdir(parents=True)
    terminal = {"outcome": outcome, "reason": "fixture", "selection": None}
    publications.write_text(
        json.dumps({"publications": [{"event": {"kind": "run_ended", "result": terminal}}]})
    )

    status = _inspect(tmp_path, "no")

    assert status["running"] is False
    assert status["terminal"] == terminal


def _git(root: Path, *args: str) -> str:
    return run_test_command(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=wangyuzheng",
            "-c",
            "user.email=yuuzhwang@gmail.com",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=HANG_GUARD_S,
    ).stdout.strip()


@pytest.mark.parametrize(
    ("path", "tracked", "accepted"),
    [
        (".vibesys/state/runs/run/core-store/store.json", False, True),
        (".vibesys/state/runs/run/publications/publications.json", False, True),
        (".vibesys/state/local/runtime.json", False, True),
        ("candidate.py", False, False),
        ("candidate.py", True, False),
        (".vibesys/tasks/default/vibesys.input.toml", True, False),
        (".vibesys/state/project.json", True, False),
    ],
)
def test_candidate_source_check_preserves_framework_state(
    tmp_path: Path, path: str, *, tracked: bool, accepted: bool
) -> None:
    # A real Git repository verifies Git's pathspec and tracked/untracked semantics.
    _git(tmp_path, "init", "-q")
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("initial")
    if tracked:
        _git(tmp_path, "add", "--all")
        _git(tmp_path, "commit", "-qm", "initial")
    target.write_text("changed")

    result = run_test_command(
        ["bash", str(_SCRIPTS / "check-project.sh"), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=HANG_GUARD_S,
    )

    assert (result.returncode == 0) is accepted
    if not accepted:
        assert path in result.stderr
        assert str(tmp_path) in result.stderr


def test_inspector_keeps_resume_identity_before_first_event(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/resume_run_id").write_text("resume-target\n")
    publications = (
        tmp_path / "project/.vibesys/state/runs/resume-target/publications/publications.json"
    )
    publications.parent.mkdir(parents=True)
    terminal = {"outcome": "failure", "reason": "budget", "selection": None}
    publications.write_text(
        json.dumps({"publications": [{"event": {"kind": "run_ended", "result": terminal}}]})
    )

    status = _inspect(tmp_path, "no")

    assert status["run_id"] == "resume-target"
    assert status["last_event"] is None
    assert status["terminal"] == terminal


def test_prepare_recovers_resume_identity_before_first_event(tmp_path: Path) -> None:
    # The shell entrypoint must carry the checkpoint across a failed pre-launch resume.
    worktree = tmp_path / "worktree"
    scripts = worktree / "support/experiments/qwen3-no-hints"
    shutil.copytree(_SCRIPTS, scripts)
    _git(worktree, "init", "-q")
    _git(worktree, "add", "--all")
    _git(worktree, "commit", "-qm", "initial")
    previous = tmp_path / "experiment/runs/previous"
    (previous / "scripts").mkdir(parents=True)
    (previous / "scripts/resume_run_id").write_text("resume-target\n")
    project = previous / "project"
    project.mkdir()
    _git(project, "init", "-q")
    _git(project, "commit", "--allow-empty", "-qm", "input")

    result = run_test_command(
        [
            "bash",
            str(scripts / "prepare.sh"),
            str(worktree),
            str(tmp_path / "experiment"),
            "4",
            "resume",
            str(previous),
            "",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=HANG_GUARD_S,
    )
    resumed = Path(result.stdout.strip())

    assert (resumed / "scripts/resume_run_id").read_text() == "resume-target\n"
    assert (resumed / "project").resolve() == project
    assert _git(project, "status", "--porcelain") == ""
