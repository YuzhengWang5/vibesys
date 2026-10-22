"""Stop a Scenario C diagnostic after trusted measurements and first-issue review."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from entrypoints.production import production_runs
from headless import HeadlessRenderer
from vibesys.api import (
    ComputeBackend,
    OrchestrationDescriptor,
    ProfilerKind,
    RunRequest,
    RunStatus,
    load_config,
)
from vibesys.api.request import load_task, make_run_environment_spec

if TYPE_CHECKING:
    from vibesys.api import CoreEvent, RunHandle


@dataclass
class PilotProgress:
    """Require trusted accuracy and performance before accepting normal flow."""

    accuracy_passed: bool = False
    benchmark_value: float | None = None
    judge_passed: bool = False
    next_workstream_started: bool = False

    def observe(self, event: CoreEvent) -> bool:
        """Accept either benchmark-before-review or benchmark-during-assessment order."""
        row = event.model_dump(mode="json")
        data = row.get("data") or {}
        if data.get("kind") == "gate_finished" and row.get("status") == "completed":
            if data.get("gate") in ("accuracy", "validation"):
                self.accuracy_passed = True
            if data.get("gate") == "benchmark":
                value = data.get("value")
                if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                    self.benchmark_value = float(value)
        execution = row.get("execution_id") or ""
        if data.get("kind") == "invocation_finished" and execution.startswith("inv:judge:"):
            result = data.get("result")
            if isinstance(result, str):
                result = json.loads(result)
            if isinstance(result, dict):
                reply = result.get("reply")
                if isinstance(reply, dict) and reply.get("verdict") == "pass":
                    self.judge_passed = True
        if (
            data.get("kind") == "invocation_started"
            and execution.startswith("inv:perf-evaluator:")
            and self.accuracy_passed
            and self.judge_passed
        ):
            self.next_workstream_started = True
        return self.next_workstream_started and self.benchmark_value is not None


async def _settle(handle: RunHandle) -> None:
    handle.stop()
    try:
        async with asyncio.timeout(180):
            await handle.result()
    except TimeoutError:
        handle.cancel()
        async with asyncio.timeout(60):
            await handle.result()


async def run(args: argparse.Namespace) -> int:
    """Run exactly one bounded pilot through the normal public run interface."""
    bundle = load_task(args.project, "default")
    request = RunRequest(
        project_root=args.project,
        input_bundle=bundle,
        objective=bundle.objective,
        config=load_config(args.config),
        orchestration=OrchestrationDescriptor(
            id="plain",
            config_version=1,
            options={
                "max_rounds": 2,
                "max_attempts_per_issue": 1,
                "max_issues_per_perf_eval": 1,
                "max_issues_per_round": 1,
                "tracker": {"backend": "local"},
            },
        ),
        run_id=args.run_id,
        profiler_kind=ProfilerKind.NONE,
        agent_backend="cli",
        cli_provider="copilot",
        allow_unvetted_model=True,
        backend=ComputeBackend.CUDA,
        run_environment=make_run_environment_spec(),
    )
    handle = production_runs().start(request)
    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, handle.stop)
    progress = PilotProgress()
    renderer = HeadlessRenderer()
    stopped_at_goal = False
    deadline_reached = False
    settled = False
    try:
        try:
            async with asyncio.timeout(args.timeout_seconds):
                async for event in handle.events():
                    renderer.handle(event)
                    if progress.observe(event) and not stopped_at_goal:
                        stopped_at_goal = True
                        handle.stop()
                result = await handle.result()
        except TimeoutError:
            deadline_reached = True
            await _settle(handle)
            result = await handle.result()
        settled = True
    finally:
        if not settled:
            await _settle(handle)
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(number)
    verified = stopped_at_goal and result.status is RunStatus.STOPPED and not deadline_reached
    args.status_file.write_text(
        json.dumps(
            {
                "goal_reached": verified,
                "accuracy_passed": progress.accuracy_passed,
                "benchmark_value": progress.benchmark_value,
                "judge_passed": progress.judge_passed,
                "next_workstream_started": progress.next_workstream_started,
                "stopped_at_goal": stopped_at_goal,
                "deadline_reached": deadline_reached,
                "result": result.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if verified else 1


def main() -> int:
    """Parse the pinned project and status paths for one GPU-controlled run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5400)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
