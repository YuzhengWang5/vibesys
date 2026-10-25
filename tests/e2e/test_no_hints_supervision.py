"""Exercise the external experiment inspector through its shell entry point."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from tests.support import run_test_command

from vs_sim.api.testing import HANG_GUARD_S

_INSPECTOR = Path(__file__).resolve().parents[2] / "support/experiments/qwen3-no-hints/inspect.sh"


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


@pytest.mark.parametrize("mode", ["restart", "resume"])
@pytest.mark.parametrize(
    ("path", "tracked", "accepted_on_resume"),
    [
        ("candidate.py", False, False),
        ("candidate.py", True, False),
        (".vibesys/state/project.json", False, False),
        (".vibesys/state/runs/run/run.json", True, False),
        (".vibesys/state/runs/run/core-store/store.json", False, True),
        (".vibesys/state/runs/run/artifacts/objects/hash", False, True),
        (".vibesys/state/runs/run/publications/publications.json", False, True),
        (".vibesys/state/runs/run/roadmap/roadmap.json", False, True),
        (".vibesys/state/runs-other/file", False, False),
        ("file with\nnewline", False, False),
    ],
)
def test_candidate_startup_checks_preserve_resume_state(
    tmp_path: Path, mode: str, path: str, *, tracked: bool, accepted_on_resume: bool
) -> None:
    # Real tier: exercise Git's pathspec and tracked/untracked semantics, not a model of them.
    def git(*args: str) -> None:
        run_test_command(
            ["git", "-C", str(tmp_path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=HANG_GUARD_S,
        )

    git("init", "--quiet")
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("initial\n")
    if tracked:
        git("add", "--all")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "Initial fixture",
        )
        target.write_text("changed\n")
    checker = _INSPECTOR.with_name("check-project.sh")
    result = run_test_command(
        ["bash", str(checker), str(tmp_path), mode],
        capture_output=True,
        text=True,
        check=False,
        timeout=HANG_GUARD_S,
    )

    assert (result.returncode == 0) == (mode == "resume" and accepted_on_resume)
    if result.returncode:
        assert result.stderr


@pytest.mark.parametrize(("mode", "returncode"), [("restart", 0), ("resume", 0), ("other", 2)])
def test_candidate_startup_checks_clean_repository_and_mode(
    tmp_path: Path, mode: str, returncode: int
) -> None:
    run_test_command(
        ["git", "-C", str(tmp_path), "init", "--quiet"],
        capture_output=True,
        check=True,
        timeout=HANG_GUARD_S,
    )
    result = run_test_command(
        ["bash", str(_INSPECTOR.with_name("check-project.sh")), str(tmp_path), mode],
        capture_output=True,
        text=True,
        check=False,
        timeout=HANG_GUARD_S,
    )

    assert result.returncode == returncode
    if returncode:
        assert mode in result.stderr


@pytest.mark.parametrize("local_state", [False, True])
def test_resume_preparation_reuses_saved_run_id_and_local_receipts(
    tmp_path: Path, *, local_state: bool
) -> None:
    worktree = tmp_path / "worktree"
    scripts = worktree / "support/experiments/qwen3-no-hints"
    scripts.mkdir(parents=True)
    (scripts / "fixture.sh").write_text("# fixture\n")
    project = tmp_path / "experiment/runs/previous/project"
    project.mkdir(parents=True)
    previous = project.parent
    (previous / "scripts").mkdir()
    (previous / "scripts/resume_run_id").write_text("saved-run\n")
    state_home = previous / "logs/vibesys-state"
    receipt = state_home / "projects/project/runs/saved-run/receipts/workspace-revisions/fixture"
    if local_state:
        receipt.parent.mkdir(parents=True)
        receipt.write_text("retained-revision\n")
    for repo in (worktree, project):
        for args in (
            ("init", "--quiet"),
            ("add", "--all"),
            (
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "Initial fixture",
            ),
        ):
            run_test_command(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                check=True,
                timeout=HANG_GUARD_S,
            )
    result = run_test_command(
        [
            "bash",
            str(_INSPECTOR.with_name("prepare.sh")),
            str(worktree),
            str(tmp_path / "experiment"),
            "5",
            "resume",
            str(previous),
            "",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=HANG_GUARD_S,
    )
    if not local_state:
        assert result.returncode != 0
        assert "vibesys-state" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    prepared = Path(result.stdout.strip())

    assert (prepared / "scripts/resume_run_id").read_text() == "saved-run\n"
    assert (prepared / "project").resolve() == project
    copied = prepared / "logs/vibesys-state" / receipt.relative_to(state_home)
    assert copied.read_text() == "retained-revision\n"
    assert not copied.samefile(receipt)
    copied.write_text("new-host\n")
    assert receipt.read_text() == "retained-revision\n"
