"""Serial operator CLI: prepare, queue, inspect, approve and retry.

No command starts a GPU experiment except ``queue`` and its leased ``worker``.
State is durable before dispatch; an interrupted attempt is not replayed.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import io
import json
import logging
import os
import shlex
import shutil
import signal
import sys
import tarfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from qwen3_single_optimization.core import (
    Approval,
    Inspection,
    PilotReview,
    RunSpec,
    RunState,
    Slot,
    Status,
    acknowledged_failures,
    can_launch,
    fail,
    slots,
    verify_ab,
)
from qwen3_single_optimization.evidence import inspect_run
from vs_project.api import Project
from vs_prompts.api import TemplateRenderer
from vs_runtime.api.infrastructure import SDKRoots, materialize_input_project
from vs_sim.api import (
    DISCARD,
    InheritedStdioLauncher,
    OsProcesses,
    ProcessOutcome,
    ProcessSpec,
    SystemClock,
    utc_datetime,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
BUNDLE = Path("examples/model-serving/qwen3-32b-code-edit")
TASK = Path(".vibesys/tasks/default")
PYTHON = Path("/mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python")
MODEL = Path("/mnt_ssd/data1/models/Qwen3-32B")
GRACE_SECONDS = 180
LEASE_WAIT_SECONDS = 43200
COPILOT_TRAILER = "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"


def command(args: list[str], *, cwd: Path | None = None) -> str:
    """Run a bounded CPU/control command, preserving errors."""
    return command_bytes(args, cwd=cwd).decode().strip()


def command_bytes(args: list[str], *, cwd: Path | None = None) -> bytes:
    """Call the process-owning public interface without a shell."""
    outcome = OsProcesses().run(ProcessSpec(argv=tuple(args), cwd=cwd), timeout=120, output=DISCARD)
    if not isinstance(outcome, ProcessOutcome):
        fail(f"控制命令未正常结束: {args}: {outcome}")
    if outcome.returncode != 0:
        fail(f"控制命令失败 ({outcome.returncode}): {args}: {outcome.stderr.decode()}")
    return outcome.stdout


def timestamp() -> str:
    """Read the real clock only in this operator shell."""
    return utc_datetime(SystemClock()).strftime("%Y%m%d_%H%M%S")


def digest(path: Path) -> str:
    """Hash a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, model: BaseModel) -> None:
    """Atomically publish JSON beside its destination, never in a temporary directory."""
    pending = path.with_name(path.name + ".pending")
    pending.write_text(model.model_dump_json(indent=2) + "\n")
    pending.replace(path)


def read_spec(run: Path) -> RunSpec:
    """Load immutable identity and reject a moved/aliased attempt."""
    spec = RunSpec.model_validate_json((run / "scripts/spec.json").read_bytes())
    if spec.run != run.resolve():
        fail(f"spec.run 不匹配: {run}")
    return spec


def read_state(run: Path) -> RunState:
    """Read durable attempt state."""
    return RunState.model_validate_json((run / "logs/status.json").read_bytes())


def records(spec: RunSpec, state: RunState) -> None:
    """Publish run records from templates, without rewriting earlier attempts."""
    renderer = TemplateRenderer(HERE / "templates")
    save(spec.run / "logs/status.json", state)
    for document, name in (("run", "README.md"), ("summary", "summary.md")):
        (spec.run / name).write_text(
            renderer.render_template("record.md.j2", document=document, spec=spec, state=state)
        )


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Own one queue or execution; refuse rather than wait with an idle GPU lease."""
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def checkpoint(source: Path, expected: str | None = None) -> tuple[str, str]:
    """Require the current clean source checkpoint, not a hard-coded historic pin."""
    dirty = command(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=source)
    if dirty:
        fail(f"源码未 checkpoint; 拒绝准备/运行:\n{dirty}")
    head = command(["git", "rev-parse", "HEAD"], cwd=source)
    if expected is not None and head != expected:
        fail(f"HEAD 已变化: expected={expected}; actual={head}")
    return head, command(["git", "branch", "--show-current"], cwd=source)


def all_specs(experiment: Path) -> list[RunSpec]:
    """Enumerate immutable run specs; partial preparations remain visibly broken."""
    return [read_spec(run) for run in sorted((experiment / "runs").iterdir()) if run.is_dir()]


def latest_specs(experiment: Path, phase: str) -> list[RunSpec]:
    """Choose newest attempt per planned slot, without deleting failed history."""
    latest = {}
    for spec in all_specs(experiment):
        if spec.slot.phase == phase:
            key = (spec.slot.optimization, spec.slot.repeat)
            if key not in latest or spec.attempt > latest[key].attempt:
                latest[key] = spec
    return list(latest.values())


def clean_docs(project: Path, optimization: str) -> str:
    """Render task-owned goals and remove inherited combined-algorithm directives."""
    renderer = TemplateRenderer(HERE / "templates")
    objective = renderer.render_template("OBJECTIVE.md.j2", optimization=optimization)
    (project / TASK / "OBJECTIVE.md").write_text(objective)
    for document, relative in (
        ("root", Path("README.md")),
        ("benchmark", TASK / "benchmark/README.md"),
        ("reference", TASK / "reference/README.md"),
    ):
        (project / relative).write_text(
            renderer.render_template("candidate-doc.md.j2", document=document)
        )
    return objective


def copy_directory(source: Path, destination: Path) -> None:
    """Materialize SDK source, excluding machine-local/generated state."""
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "*.pyc"),
    )


def trusted_files(project: Path) -> dict[str, str]:
    """Pin all evaluator inputs and the public-API-materialized SDK."""
    paths = [
        path
        for root in (project / TASK, project / "_input_libs")
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"README.md", "OBJECTIVE.md"}
    ]
    return {path.relative_to(project).as_posix(): digest(path) for path in sorted(paths)}


def export_candidate(source: Path, project: Path, head: str, slot: Slot) -> str:
    """Export a clean bundle and initialize new independent candidate history."""
    archive = command_bytes(["git", "archive", f"{head}:{BUNDLE}"], cwd=source)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(project, filter="data")
    materialize_input_project(
        source / BUNDLE,
        project,
        sdk_roots=SDKRoots(checkout=source / "sdk", packaged=None),
        copy_directory=copy_directory,
    )
    objective = clean_docs(project, slot.optimization.value)
    if "/v1/logits" not in (project / TASK / "accuracy_checker/README.md").read_text():
        fail("导出 bundle 尚无新 logits accuracy gate; 拒绝准备。")
    command(["git", "init", "-q", "-b", "main"], cwd=project)
    command(["git", "config", "user.name", "wangyuzheng"], cwd=project)
    command(["git", "config", "user.email", "yuuzhwang@gmail.com"], cwd=project)
    command(["git", "add", "."], cwd=project)
    command(
        [
            "git",
            "commit",
            "-q",
            "-m",
            "experiment: fresh single-optimization input",
            "-m",
            COPILOT_TRAILER,
        ],
        cwd=project,
    )
    (project / TASK / "reference/model").symlink_to(MODEL, target_is_directory=True)
    with (project / ".git/info/exclude").open("a") as stream:
        stream.write(f"\n/{TASK}/reference/model\n")
    return objective


def prepare_run(
    source: Path, experiment: Path, slot: Slot, previous: RunSpec | None = None
) -> RunSpec:
    """Persist a fresh attempt and provenance before any GPU work."""
    head, branch = checkpoint(source)
    attempt = 1 if previous is None else previous.attempt + 1
    tag = f"{slot.phase}-{slot.optimization}-r{slot.repeat}-a{attempt}"
    run = experiment / "runs" / f"{timestamp()}_{tag}-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    for name in ("scripts", "logs", "plots", "project"):
        (run / name).mkdir()
    copy_directory(HERE, run / "scripts/qwen3_single_optimization")
    for name in ("control.sh", "launch.sh"):
        shutil.copy2(HERE / name, run / "scripts" / name)
    (run / "scripts/source.txt").write_text(str(source) + "\n")
    objective = export_candidate(source, run / "project", head, slot)
    (run / "scripts/OBJECTIVE.md").write_text(objective)
    script_hashes = {
        path.relative_to(run / "scripts").as_posix(): digest(path)
        for path in sorted((run / "scripts").rglob("*"))
        if path.is_file()
    }
    spec = RunSpec(
        slot=slot,
        attempt=attempt,
        retry_of=None if previous is None else previous.run,
        source=source,
        experiment=experiment,
        run=run,
        run_id=str(uuid.uuid4()),
        tested_commit=head,
        branch=branch,
        candidate_commit=command(["git", "rev-parse", "HEAD"], cwd=run / "project"),
        objective_sha256=digest(run / "scripts/OBJECTIVE.md"),
        trusted_sha256=trusted_files(run / "project"),
        script_sha256=script_hashes,
        python=PYTHON,
        model=MODEL,
    )
    save(run / "scripts/spec.json", spec)
    (run / "scripts/SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in script_hashes.items())
    )
    records(spec, RunState(updated_at=timestamp()))
    with (experiment / "CODE_PIN.md").open("a") as stream:
        stream.write(f"\n- `{run.name}` → `{head}`; source=`{source}`; branch=`{branch}`\n")
    return spec


def validate_inputs(spec: RunSpec) -> None:
    """Reject changed source/scripts/input contracts before launching."""
    _, branch = checkpoint(spec.source, spec.tested_commit)
    if branch != spec.branch:
        fail(f"branch 已变化: expected={spec.branch}; actual={branch}")
    for name, expected in spec.script_sha256.items():
        if digest(spec.run / "scripts" / name) != expected:
            fail(f"脚本快照发生变化: {name}")
    if digest(spec.run / "project" / TASK / "OBJECTIVE.md") != spec.objective_sha256:
        fail("candidate OBJECTIVE 与快照不一致。")
    if trusted_files(spec.run / "project") != spec.trusted_sha256:
        fail("可信 evaluator/SDK 输入已改变。")


def inspect_spec(spec: RunSpec) -> Inspection:
    """Resolve this attempt's local receipt namespace exactly as its worker did."""
    os.environ["VIBESYS_STATE_HOME"] = str(spec.run / "machine-local")
    return inspect_run(spec.run / "project", spec.run_id)


def verify_revision_inputs(spec: RunSpec, revision: str) -> None:
    """Check immutable A/B commits against the exported trusted evaluator bytes."""
    for relative, expected in spec.trusted_sha256.items():
        content = command_bytes(["git", "show", f"{revision}:{relative}"], cwd=spec.run / "project")
        if hashlib.sha256(content).hexdigest() != expected:
            fail(f"revision {revision} 更改可信输入: {relative}")
    objective = command_bytes(
        ["git", "show", f"{revision}:{TASK}/OBJECTIVE.md"], cwd=spec.run / "project"
    )
    if hashlib.sha256(objective).hexdigest() != spec.objective_sha256:
        fail(f"revision {revision} 更改 OBJECTIVE。")


def load_approval(experiment: Path, source: Path) -> Approval:
    """Revalidate the pilot against live canonical evidence, not an exit code."""
    approval = Approval.model_validate_json((experiment / "pilot-approval.json").read_bytes())
    spec = read_spec(approval.pilot)
    state = read_state(spec.run)
    checkpoint(source, approval.tested_commit)
    if (
        spec.source != source
        or spec.experiment != experiment
        or spec.slot.phase != "pilot"
        or spec.tested_commit != approval.tested_commit
        or spec.objective_sha256 != approval.objective_sha256
        or state.status is not Status.FINISHED
        or state.returncode != 0
        or not state.cleanup_ok
    ):
        fail("pilot approval 与源码/状态不符。")
    baseline, candidate = verify_ab(inspect_spec(spec), approval.review)
    verify_revision_inputs(spec, baseline.revision)
    verify_revision_inputs(spec, candidate.revision)
    if baseline != approval.baseline or candidate != approval.candidate:
        fail("pilot approval 的可信测量发生变化。")
    return approval


def prepare(args: argparse.Namespace) -> None:
    """Prepare a pilot or gated batch without acquiring a GPU."""
    source = Path(os.environ["QWEN3_SOURCE"]).resolve()
    checkpoint(source)
    if args.phase == "pilot":
        experiment = source.parents[2] / "exps" / f"{timestamp()}_{source.name}"
        experiment.mkdir(parents=True, exist_ok=False)
        (experiment / "runs").mkdir()
        (experiment / "CODE_PIN.md").write_text("# 每 run 的实际源码 checkpoint\n")
        (experiment / "README.md").write_text(
            TemplateRenderer(HERE / "templates").render_template(
                "record.md.j2", document="direction", source=source
            )
        )
    else:
        if args.experiment is None:
            fail("prepare batch 需要 --experiment。")
        experiment = args.experiment.resolve()
        load_approval(experiment, source)
        if latest_specs(experiment, "batch"):
            fail("batch 已准备; 使用 status/queue/retry, 不覆盖原计划。")
    with locked(experiment / ".queue.lock"):
        for slot in slots(args.phase):
            spec = prepare_run(source, experiment, slot)
            sys.stdout.write(str(spec.run) + "\n")


def framework_args(spec: RunSpec) -> list[str]:
    """Build the fixed operator budget, with no inherited historic run flags."""
    return [
        str(spec.python),
        "-m",
        "entrypoints.headless",
        "--outer-loop",
        spec.outer_loop,
        "--inner-loop",
        "multi-agent",
        "--project",
        str(spec.run / "project"),
        "--task",
        "default",
        "--headless",
        "--local",
        "--run-id",
        spec.run_id,
        "--exp-name",
        spec.source.name,
        "--agent-backend",
        "cli",
        "--cli-provider",
        spec.provider,
        "--model",
        spec.agent_model,
        "--allow-unvetted-model",
        "--backend",
        "cuda",
        "--run-environment",
        "docker",
        "--interface",
        "service",
        "--max-rounds",
        str(spec.max_rounds),
        "--max-in-flight",
        str(spec.max_in_flight),
        "--evaluation-slots",
        "1",
        "--max-backlog",
        "0",
        "--max-run-seconds",
        str(spec.max_run_seconds),
    ]


def run_process(args: list[str], log: Path, *, cwd: Path, timeout: int) -> int:
    """Forward interruption before killing only this owned child's PID."""
    return asyncio.run(run_foreground(args, log, cwd=cwd, wait_seconds=timeout))


async def run_foreground(args: list[str], log: Path, *, cwd: Path, wait_seconds: int) -> int:
    """Keep process ownership in vs-sim; bound graceful shutdown before force kill."""
    child = await InheritedStdioLauncher().start(
        ["bash", str(HERE / "logged-command.sh"), str(log), *args], cwd=cwd
    )
    try:
        return await asyncio.wait_for(asyncio.shield(child.wait()), timeout=wait_seconds)
    except BaseException:
        child.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(asyncio.shield(child.wait()), timeout=GRACE_SECONDS)
        except TimeoutError:
            child.send_signal(signal.SIGKILL)
            await asyncio.wait_for(child.wait(), timeout=GRACE_SECONDS)
        raise


def cleanup(spec: RunSpec) -> bool:
    """Remove only containers labeled with this run's unique framework ID."""
    owned = command(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=vibesys.run-id={spec.run_id}",
        ]
    ).split()
    if owned:
        command(["docker", "rm", "-f", *owned])
    remaining = command(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=vibesys.run-id={spec.run_id}",
        ]
    )
    return not remaining


def archive_framework(spec: RunSpec) -> None:
    """Archive public Project-owned locations, never reconstruct private layout."""
    project = Project.open(spec.run / "project")
    for source, name in (
        (project.state.log_directory(spec.run_id), "framework-final"),
        (project.state.local_namespace(spec.run_id, "receipts").external_directory(), "receipts"),
    ):
        if source.exists():
            shutil.copytree(source, spec.run / "logs" / name, dirs_exist_ok=True, symlinks=True)
    snapshot = project.state.portable_run_export(spec.run_id)
    archive = spec.run / "logs/portable-state"
    archive.mkdir(exist_ok=True)
    for item in snapshot.files:
        destination = archive / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.contents)


def execute(spec: RunSpec) -> None:
    """Run within the gpuctl lease, preserving failure and cleanup evidence."""
    state = read_state(spec.run)
    if state.status is not Status.LAUNCHING:
        fail("worker 只能消费队列预先记录的 launching intent。")
    result = None
    detail = ""
    inspection = None
    cleanup_ok = False
    try:
        validate_inputs(spec)
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        if not visible or len(visible.split(",")) != 1:
            fail(f"每 run 必须恰好一张 GPU: {visible}")
        gpu_state = command(["nvidia-smi"])
        (spec.run / "logs/nvidia-smi-before.txt").write_text(gpu_state + "\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        (spec.run / "scripts/device.env").write_text(
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible)}\n"
        )
        busy = command(
            [
                "nvidia-smi",
                "-i",
                visible,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        if busy:
            fail(f"gpuctl 分配卡仍有计算进程, 拒绝启动: {busy}")
        state = state.model_copy(
            update={
                "status": Status.RUNNING,
                "updated_at": timestamp(),
                "cuda_visible_devices": visible,
                "detail": "运行中, 尚无成功结论。",
            }
        )
        records(spec, state)
        os.environ["VIBESYS_STATE_HOME"] = str(spec.run / "machine-local")
        args = framework_args(spec)
        (spec.run / "scripts/argv.json").write_text(json.dumps(args, indent=2) + "\n")
        result = run_process(
            [
                "timeout",
                "--signal=INT",
                f"--kill-after={GRACE_SECONDS}s",
                f"{spec.max_run_seconds + GRACE_SECONDS}s",
                *args,
            ],
            spec.run / "logs/vibesys.log",
            cwd=spec.run / "project",
            timeout=spec.max_run_seconds + 3 * GRACE_SECONDS,
        )
        inspection = inspect_spec(spec)
        save(spec.run / "logs/inspection.json", inspection)
        detail = inspection.detail
    except (Exception, KeyboardInterrupt) as error:
        detail = f"{type(error).__name__}: {error}"
        LOGGER.exception("本 attempt 失败; 不会原地重试")
    finally:
        try:
            cleanup_ok = cleanup(spec)
            archive_framework(spec)
        except Exception as error:
            cleanup_ok = False
            detail += f"\n归档/清理异常: {type(error).__name__}: {error}"
            LOGGER.exception("归档/清理异常")
        finished = (
            result == 0 and inspection is not None and inspection.terminal_success and cleanup_ok
        )
        state = state.model_copy(
            update={
                "status": Status.FINISHED if finished else Status.FAILED,
                "updated_at": timestamp(),
                "returncode": result,
                "detail": detail,
                "cleanup_ok": cleanup_ok,
            }
        )
        records(spec, state)
    if state.status is not Status.FINISHED:
        fail(f"run 失败: {spec.run}; 请查询日志并 retry 新 run。")


def worker(args: argparse.Namespace) -> None:
    """Consume one durable intent under the experiment's serial execution lock."""
    spec = read_spec(args.run.resolve())
    with locked(spec.experiment / ".execution.lock"):
        execute(spec)


def queue(args: argparse.Namespace) -> None:
    """Run each pending slot once, serially; stop on a failed or uncertain attempt."""
    experiment = args.experiment.resolve()
    with locked(experiment / ".queue.lock"):
        planned = latest_specs(experiment, args.phase)
        if len(planned) != len(slots(args.phase)):
            fail("准备的 slot 不完整; 不启动部分 batch。")
        if args.phase == "batch":
            load_approval(experiment, planned[0].source)
        acknowledged = acknowledged_failures(
            args.phase,
            {path.resolve() for path in args.acknowledged_failure},
            {spec.run for spec in planned if read_state(spec.run).status is Status.FAILED},
        )
        for spec in planned:
            state = read_state(spec.run)
            if state.status is Status.FINISHED:
                continue
            if spec.run in acknowledged:
                LOGGER.error("保留已审查的失败实验, 不重跑、不计作成功: %s", spec.run)
                continue
            can_launch(state)
            validate_inputs(spec)
            state = state.model_copy(
                update={
                    "status": Status.LAUNCHING,
                    "updated_at": timestamp(),
                    "detail": "已记录启动 intent, 等待 gpuctl 单卡租约。",
                }
            )
            records(spec, state)
            try:
                result = run_process(
                    ["bash", str(spec.run / "scripts/launch.sh")],
                    spec.run / "logs/queue.log",
                    cwd=spec.run,
                    timeout=LEASE_WAIT_SECONDS + spec.max_run_seconds + 4 * GRACE_SECONDS,
                )
            except BaseException as error:
                final = read_state(spec.run)
                if final.status is Status.LAUNCHING:
                    records(
                        spec,
                        final.model_copy(
                            update={
                                "status": Status.FAILED,
                                "updated_at": timestamp(),
                                "detail": f"启动/租约异常: {type(error).__name__}: {error}",
                            }
                        ),
                    )
                raise
            final = read_state(spec.run)
            if result != 0 or final.status is not Status.FINISHED:
                if final.status is Status.LAUNCHING:
                    records(
                        spec,
                        final.model_copy(
                            update={
                                "status": Status.FAILED,
                                "updated_at": timestamp(),
                                "returncode": result,
                                "detail": "gpuctl 未完成 worker 启动。",
                            }
                        ),
                    )
                fail(f"run 未正常完成: {spec.run}; 不继续队列。")


def approve(args: argparse.Namespace) -> None:
    """Admit batch only after trusted evidence and explicit operator code audit."""
    spec = read_spec(args.run.resolve())
    state = read_state(spec.run)
    if spec.slot.phase != "pilot" or state.status is not Status.FINISHED:
        fail("只有正常完成的 pilot 可以审查。")
    if state.returncode != 0 or not state.cleanup_ok:
        fail("pilot 退出/清理异常。")
    checkpoint(spec.source, spec.tested_commit)
    review = PilotReview.model_validate_json(args.review.read_bytes())
    inspection = inspect_spec(spec)
    baseline, candidate = verify_ab(inspection, review)
    verify_revision_inputs(spec, baseline.revision)
    verify_revision_inputs(spec, candidate.revision)
    approval = Approval(
        pilot=spec.run,
        tested_commit=spec.tested_commit,
        objective_sha256=spec.objective_sha256,
        review=review,
        baseline=baseline,
        candidate=candidate,
        gain=candidate.value / baseline.value - 1,
    )
    with locked(spec.experiment / ".queue.lock"):
        target = spec.experiment / "pilot-approval.json"
        if target.exists():
            fail("已有 pilot approval; 不覆盖历史审查。")
        save(spec.run / "scripts/pilot-review.json", review)
        patch = command_bytes(
            ["git", "diff", baseline.revision, candidate.revision],
            cwd=spec.run / "project",
        )
        (spec.run / "scripts/pilot-ab.diff").write_bytes(patch)
        save(target, approval)
    sys.stdout.write(approval.model_dump_json(indent=2) + "\n")


def retry(args: argparse.Namespace) -> None:
    """Create a new independent attempt; never overwrite/replay the original."""
    previous = read_spec(args.run.resolve())
    with locked(previous.experiment / ".queue.lock"):
        if read_state(previous.run).status not in {Status.FINISHED, Status.FAILED}:
            fail("非终态 attempt 不可 retry; 先停止并核对资源。")
        latest = latest_specs(previous.experiment, previous.slot.phase)
        if previous not in latest:
            fail("此 slot 已有更新 attempt; 不得从旧 attempt 分叉。")
        if previous.slot.phase == "batch":
            load_approval(previous.experiment, previous.source)
        spec = prepare_run(previous.source, previous.experiment, previous.slot, previous)
        sys.stdout.write(str(spec.run) + "\n")


def query(args: argparse.Namespace) -> None:
    """Read prior logs/status or derive a fresh canonical inspection without GPU work."""
    if args.action == "status":
        for spec in all_specs(args.experiment.resolve()):
            sys.stdout.write(
                json.dumps(
                    {
                        "run": str(spec.run),
                        "slot": spec.slot.model_dump(mode="json"),
                        "attempt": spec.attempt,
                        "state": read_state(spec.run).model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return
    spec = read_spec(args.run.resolve())
    if args.action == "inspect":
        value = inspect_spec(spec)
        sys.stdout.write(value.model_dump_json(indent=2) + "\n")
    else:
        path = spec.run / "logs" / args.log
        if path.parent != spec.run / "logs":
            fail("--log 只能指定 logs/ 直接子文件。")
        sys.stdout.write(
            "\n".join(path.read_text(errors="replace").splitlines()[-args.tail :]) + "\n"
        )


def parser() -> argparse.ArgumentParser:
    """Expose the bounded operator workflow, not a generic experiment engine."""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    sub = commands.add_parser("prepare")
    sub.add_argument("phase", choices=("pilot", "batch"))
    sub.add_argument("--experiment", type=Path)
    sub.set_defaults(function=prepare)
    for name, function in (("queue", queue), ("status", query)):
        sub = commands.add_parser(name)
        sub.add_argument("--experiment", type=Path, required=True)
        if name == "queue":
            sub.add_argument("--phase", choices=("pilot", "batch"), required=True)
            sub.add_argument("--acknowledged-failure", type=Path, action="append", default=[])
        sub.set_defaults(function=function)
    for name, function in (
        ("worker", worker),
        ("inspect", query),
        ("logs", query),
        ("approve-pilot", approve),
        ("retry", retry),
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
        if name == "approve-pilot":
            sub.add_argument("--review", type=Path, required=True)
        if name == "logs":
            sub.add_argument("--log", default="vibesys.log")
            sub.add_argument("--tail", type=int, default=80)
        sub.set_defaults(function=function)
    return result


def interrupt(signum: int, frame: object) -> None:
    """Route SIGTERM through the same bounded cleanup as keyboard interruption."""
    del signum, frame
    raise KeyboardInterrupt


def main() -> None:
    """Run the operator CLI and return nonzero on any failed boundary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, interrupt)
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
