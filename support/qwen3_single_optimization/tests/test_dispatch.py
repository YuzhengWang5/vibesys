"""Public operator boundaries with simulated Git and foreground processes."""

import asyncio
import signal
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from qwen3_single_optimization.controller import (
    TASK,
    all_specs,
    digest,
    dispatch_run,
    latest_specs,
    launch_run,
    parser,
    planned_queue,
    prepare_experiment,
    queue,
    read_state,
    records,
    run_foreground,
    save,
    worker,
)
from qwen3_single_optimization.core import (
    Optimization,
    RunSpec,
    RunState,
    Slot,
    Status,
    slots,
)

from vs_sim.api import ProcessSpec
from vs_sim.api.testing import (
    FakeForegroundLauncher,
    FakeProcesses,
    ForegroundScript,
    ManualClock,
    ProcessScript,
    Sim,
)

HEAD = "a" * 40


def existing_experiment(root: Path) -> Path:
    experiment = root / "experiment"
    (experiment / "runs").mkdir(parents=True, exist_ok=True)
    (experiment / "README.md").write_text("操作者现有方向说明\n")
    (experiment / "CODE_PIN.md").write_text("操作者现有 checkpoint 历史\n")
    return experiment


def prepared_run(experiment: Path, slot: Slot, *, attempt: int = 1) -> RunSpec:
    run = experiment / "runs" / f"{slot.phase}-{slot.optimization}-r{slot.repeat}-a{attempt}"
    for relative in ("scripts", "logs", f"project/{TASK}"):
        (run / relative).mkdir(parents=True)
    (run / "scripts/launch.sh").write_text("test launcher input\n")
    (run / "project" / TASK / "OBJECTIVE.md").write_text(slot.optimization.value)
    spec = RunSpec(
        slot=slot,
        attempt=attempt,
        source=experiment.parent / "source",
        experiment=experiment,
        run=run,
        run_id=run.name,
        tested_commit=HEAD,
        branch="test",
        candidate_commit="b" * 40,
        objective_sha256=digest(run / "project" / TASK / "OBJECTIVE.md"),
        trusted_sha256={},
        script_sha256={"launch.sh": digest(run / "scripts/launch.sh")},
        python=Path("/python"),
        model=Path("/model"),
    )
    save(run / "scripts/spec.json", spec)
    records(spec, RunState(updated_at="fixed-input"))
    return spec


def git_processes(*, dirty: bool = False, head: str = HEAD, branch: str = "test") -> FakeProcesses:
    clock = ManualClock()
    outputs = {
        ("git", "status", "--porcelain", "--untracked-files=normal"): b"M changed"
        if dirty
        else b"",
        ("git", "rev-parse", "HEAD"): head.encode(),
        ("git", "branch", "--show-current"): branch.encode(),
    }

    def script(spec: ProcessSpec) -> ProcessScript:
        return ProcessScript(stdout=outputs[spec.argv])

    return FakeProcesses(script, pass_time=clock.advance)


def test_reuse_preserves_direction_and_ignores_setup_runs(tmp_path: Path) -> None:
    experiment = existing_experiment(tmp_path)
    before = {path.name: path.read_bytes() for path in experiment.iterdir() if path.is_file()}
    for name in ("hf-setup", "calibration"):
        run = experiment / "runs" / name
        (run / "scripts").mkdir(parents=True)
        (run / "logs").mkdir()
        (run / "README.md").write_text("existing setup")
        (run / "scripts/source.txt").write_text("setup source provenance")
        (run / "logs/status.json").write_text('{"setup": "done"}')
    assert prepare_experiment(tmp_path / "source", "pilot", experiment) == experiment
    assert all_specs(experiment) == []
    assert {
        path.name: path.read_bytes() for path in experiment.iterdir() if path.is_file()
    } == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["experiment"]


@pytest.mark.parametrize("missing", ["README.md", "CODE_PIN.md", "runs"])
def test_reuse_requires_existing_direction_contract(tmp_path: Path, missing: str) -> None:
    experiment = existing_experiment(tmp_path)
    path = experiment / missing
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    with pytest.raises(ValueError, match=missing):
        prepare_experiment(tmp_path / "source", "pilot", experiment)


def test_batch_never_creates_a_parent_direction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="需要 --experiment"):
        prepare_experiment(tmp_path / "source", "batch", None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("artifact", ["project", ".preparing"])
def test_partial_preparations_are_not_setup_runs(tmp_path: Path, artifact: str) -> None:
    experiment = existing_experiment(tmp_path)
    path = experiment / "runs/interrupted" / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact == "project":
        path.mkdir()
    else:
        path.touch()
    with pytest.raises(ValueError, match="准备不完整"):
        all_specs(experiment)


@pytest.mark.parametrize("phase", ["pilot", "batch"])
def test_legacy_partial_run_names_are_not_setup_runs(tmp_path: Path, phase: str) -> None:
    experiment = existing_experiment(tmp_path)
    (experiment / "runs" / f"20261102_000000_{phase}-cuda-graph-r1-a1").mkdir()
    with pytest.raises(ValueError, match="准备不完整"):
        all_specs(experiment)


@given(repeat=st.integers().filter(lambda value: value < 1 or value > 3))
def test_slot_repeat_boundary_is_closed(repeat: int) -> None:
    with pytest.raises(ValidationError):
        Slot(phase="batch", optimization=Optimization.CUDA_GRAPH, repeat=repeat)


@pytest.mark.parametrize("count", [0, 1, 4, 8, 11, 12])
def test_dispatch_checks_whole_plan_before_approval(tmp_path: Path, count: int) -> None:
    experiment = existing_experiment(tmp_path)
    for slot in slots("batch")[:count]:
        prepared_run(experiment, slot)
    message = "pilot approval" if count == 12 else "slot 不完整"
    with pytest.raises(ValueError, match=message):
        planned_queue(experiment, "batch", None)


def test_duplicate_attempt_and_incomplete_spec_remain_visible(tmp_path: Path) -> None:
    experiment = existing_experiment(tmp_path)
    spec = prepared_run(experiment, slots("pilot")[0])
    duplicate = experiment / "runs/duplicate"
    (duplicate / "scripts").mkdir(parents=True)
    (duplicate / "logs").mkdir()
    save(duplicate / "scripts/spec.json", spec.model_copy(update={"run": duplicate}))
    records(spec.model_copy(update={"run": duplicate}), RunState(updated_at="fixed"))
    with pytest.raises(ValueError, match="attempt 重复"):
        latest_specs(experiment, "pilot")
    (duplicate / "logs/status.json").unlink()
    with pytest.raises(ValueError, match="准备不完整"):
        all_specs(experiment)


async def test_dispatch_requires_approval_even_for_a_complete_batch(tmp_path: Path) -> None:
    experiment = existing_experiment(tmp_path)
    specs = [prepared_run(experiment, slot) for slot in slots("batch")]
    launcher = FakeForegroundLauncher(lambda _: ForegroundScript(exits_immediately=True))
    with pytest.raises(ValueError, match="pilot approval"):
        await dispatch_run(specs[-1].run, launcher=launcher)
    assert not launcher.children
    assert all(read_state(spec.run).status is Status.PREPARED for spec in specs)


async def test_dispatch_consumes_one_intent_and_refuses_duplicate(tmp_path: Path, sim: Sim) -> None:
    spec = prepared_run(existing_experiment(tmp_path), slots("pilot")[0])
    observed = []

    def worker(_argv: tuple[str, ...]) -> ForegroundScript:
        observed.append(read_state(spec.run).status)
        records(
            spec,
            RunState(status=Status.FINISHED, updated_at="worker", returncode=0, cleanup_ok=True),
        )
        return ForegroundScript(exits_immediately=True)

    launcher = FakeForegroundLauncher(worker)
    for expected in (None, ValueError):
        if expected is None:
            await dispatch_run(
                spec.run, launcher=launcher, processes=git_processes(), clock=sim.clock
            )
        else:
            with pytest.raises(expected, match="不可重跑"):
                await dispatch_run(
                    spec.run, launcher=launcher, processes=git_processes(), clock=sim.clock
                )
    assert observed == [Status.LAUNCHING]
    assert len(launcher.children) == 1
    assert str(spec.run / "scripts/launch.sh") in launcher.children[0].argv


async def test_dispatch_refuses_superseded_attempt(tmp_path: Path) -> None:
    experiment = existing_experiment(tmp_path)
    old = prepared_run(experiment, slots("pilot")[0])
    prepared_run(experiment, slots("pilot")[0], attempt=2)
    launcher = FakeForegroundLauncher(lambda _: ForegroundScript(exits_immediately=True))
    with pytest.raises(ValueError, match="最新 attempt"):
        await dispatch_run(old.run, launcher=launcher)
    assert not launcher.children


@pytest.mark.parametrize("returncode", [0, 1, 124])
async def test_uncertain_launch_preserves_intent_and_cannot_replay(
    tmp_path: Path, sim: Sim, returncode: int
) -> None:
    spec = prepared_run(existing_experiment(tmp_path), slots("pilot")[0])
    launcher = FakeForegroundLauncher(
        lambda _: ForegroundScript(returncode=returncode, exits_immediately=True)
    )
    with pytest.raises(ValueError, match="保留 intent"):
        await dispatch_run(spec.run, launcher=launcher, processes=git_processes(), clock=sim.clock)
    assert read_state(spec.run).status is Status.LAUNCHING
    with pytest.raises(ValueError, match="不可重跑"):
        await dispatch_run(spec.run, launcher=launcher, processes=git_processes(), clock=sim.clock)
    assert len(launcher.children) == 1


async def test_dispatch_and_queue_competition_never_duplicates_launch(
    tmp_path: Path, sim: Sim
) -> None:
    spec = prepared_run(existing_experiment(tmp_path), slots("pilot")[0])
    launcher = FakeForegroundLauncher(lambda _: ForegroundScript())
    first = asyncio.create_task(
        dispatch_run(spec.run, launcher=launcher, processes=git_processes(), clock=sim.clock)
    )
    child = await asyncio.wait_for(launcher.child(), timeout=10)
    with pytest.raises(BlockingIOError):
        await dispatch_run(spec.run, launcher=launcher, processes=git_processes(), clock=sim.clock)
    args = parser().parse_args(["queue", "--experiment", str(spec.experiment), "--phase", "pilot"])
    with pytest.raises(BlockingIOError):
        queue(args)
    records(spec, RunState(status=Status.FINISHED, updated_at="worker"))
    child.exit(0)
    await asyncio.wait_for(first, timeout=10)
    assert len(launcher.children) == 1


async def test_admitted_same_treatment_trials_have_no_direction_serialization(
    tmp_path: Path, sim: Sim
) -> None:
    experiment = existing_experiment(tmp_path)
    specs = [
        prepared_run(
            experiment, Slot(phase="batch", optimization=Optimization.CUDA_GRAPH, repeat=repeat)
        )
        for repeat in (1, 2, 3)
    ]
    launcher = FakeForegroundLauncher(lambda _: ForegroundScript())
    tasks = [
        asyncio.create_task(
            launch_run(spec, launcher=launcher, processes=git_processes(), clock=sim.clock)
        )
        for spec in specs
    ]
    await asyncio.wait_for(launcher.child(2), timeout=10)
    assert all(read_state(spec.run).status is Status.LAUNCHING for spec in specs)
    for spec, child in zip(specs, launcher.children, strict=True):
        records(spec, RunState(status=Status.FINISHED, updated_at="worker"))
        child.exit(0)
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)


async def test_interruption_forwards_then_reaps_owned_child(tmp_path: Path) -> None:
    launcher = FakeForegroundLauncher(
        lambda _: ForegroundScript(ignored_signals=frozenset({signal.SIGINT}))
    )
    task = asyncio.create_task(
        run_foreground(
            ["gpuctl", "test"],
            tmp_path / "host.log",
            cwd=tmp_path,
            wait_seconds=3600,
            launcher=launcher,
        )
    )
    child = await asyncio.wait_for(launcher.child(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1000)
    assert child.received == [signal.SIGINT, signal.SIGKILL]
    assert not child.running


@pytest.mark.parametrize(
    "status", [Status.PREPARED, Status.RUNNING, Status.FINISHED, Status.FAILED]
)
def test_worker_never_reconsumes_nonlaunching_intent(tmp_path: Path, status: Status) -> None:
    spec = prepared_run(existing_experiment(tmp_path), slots("pilot")[0])
    records(spec, RunState(status=status, updated_at="fixed"))
    with pytest.raises(ValueError, match="launching intent"):
        worker(parser().parse_args(["worker", "--run", str(spec.run)]))
    assert read_state(spec.run).status is status


def test_worker_claim_is_durable_before_first_external_command(tmp_path: Path) -> None:
    spec = prepared_run(existing_experiment(tmp_path), slots("pilot")[0])
    records(spec, RunState(status=Status.LAUNCHING, updated_at="intent"))
    observed = []
    clock = ManualClock()

    def script(command: ProcessSpec) -> ProcessScript:
        observed.append(read_state(spec.run).status)
        if command.argv[0] == "git":
            return ProcessScript(stdout=b"M dirty")
        assert command.argv[:3] == ("docker", "ps", "-aq")
        assert f"label=vibesys.run-id={spec.run_id}" in command.argv
        return ProcessScript()

    processes = FakeProcesses(script, pass_time=clock.advance)
    with pytest.raises(ValueError, match="run 失败"):
        worker(
            parser().parse_args(["worker", "--run", str(spec.run)]),
            processes=processes,
            clock=clock,
        )
    assert observed
    assert all(status is Status.RUNNING for status in observed)
    assert read_state(spec.run).status is Status.FAILED


@pytest.mark.parametrize("changed", ["source", "head", "branch", "script", "objective", "trusted"])
async def test_pin_failures_never_record_launch_intent(
    tmp_path: Path, sim: Sim, changed: str
) -> None:
    spec = prepared_run(existing_experiment(tmp_path), slots("pilot")[0])
    if changed in {"script", "objective", "trusted"}:
        paths = {
            "script": spec.run / "scripts/launch.sh",
            "objective": spec.run / "project" / TASK / "OBJECTIVE.md",
            "trusted": spec.run / "project" / TASK / "checker.py",
        }
        paths[changed].write_text("changed")
    processes = git_processes(
        dirty=changed == "source",
        head="other" if changed == "head" else HEAD,
        branch="other" if changed == "branch" else "test",
    )
    launcher = FakeForegroundLauncher(lambda _: ForegroundScript(exits_immediately=True))
    with pytest.raises(ValueError, match=r"checkpoint|变化|不一致|已改变"):
        await dispatch_run(spec.run, launcher=launcher, processes=processes, clock=sim.clock)
    assert read_state(spec.run).status is Status.PREPARED
    assert not launcher.children
