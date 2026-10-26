"""Run and observe one GPU-controlled strict Scenario C experiment through the public API."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from entrypoints.production import production_runs
from headless import HeadlessRenderer
from vibesys.api import (
    ComputeBackend,
    OrchestrationDescriptor,
    ProfilerKind,
    RunRequest,
    load_config,
)
from vibesys.api.request import load_task, make_run_environment_spec

if TYPE_CHECKING:
    from vibesys.api import RunHandle


async def settle(handle: RunHandle) -> None:
    """A failed observer still stops the run and waits for owned-resource cleanup."""
    handle.stop()
    try:
        async with asyncio.timeout(300):
            await handle.result()
    except TimeoutError:
        handle.cancel()
        async with asyncio.timeout(60):
            await handle.result()


def write_json(path: Path, value: object) -> None:
    """Publish whole snapshots so an external monitor never reads a half-written record."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> int:
    """Persist semantic events and final facts without parsing renderer output."""
    options = json.loads(args.options.read_text())
    if not options.pop("strict_sequence"):
        message = "this observer requires strict_sequence=true"
        raise ValueError(message)
    maximum_seconds = options.pop("max_run_seconds")
    bundle = load_task(args.project, "default")
    benchmark = bundle.benchmark_result
    if benchmark is None:
        message = "strict Scenario C requires a declared benchmark result"
        raise ValueError(message)
    config = load_config(args.config)
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"max_run_seconds": maximum_seconds})}
    )
    request = RunRequest(
        project_root=args.project,
        input_bundle=bundle,
        objective=bundle.objective,
        config=config,
        orchestration=OrchestrationDescriptor(
            id="multi-agent",
            config_version=1,
            options={
                **options,
                "strict_sequence": True,
                "interface": "service",
                "modality": "text_generation",
                "metric_space": {
                    "objectives": [{"name": benchmark.metric, "direction": "max"}],
                },
            },
        ),
        run_id=args.run_id,
        profiler_kind=ProfilerKind.AUTO,
        agent_backend="cli",
        cli_provider="copilot",
        allow_unvetted_model=True,
        backend=ComputeBackend.CUDA,
        run_environment=make_run_environment_spec(),
    )
    write_json(
        args.logs / "request.json",
        {
            "run_id": args.run_id,
            "orchestration": request.orchestration.model_dump(mode="json"),
            "max_run_seconds": maximum_seconds,
            "provider": "copilot",
            "model": config.model.name,
            "reasoning": config.thinking.level,
        },
    )
    handle = production_runs().start(request)
    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, handle.stop)
    renderer = HeadlessRenderer()
    finished = False
    try:
        with (args.logs / "events.jsonl").open("w") as events:
            async for event in handle.events():
                events.write(event.model_dump_json() + "\n")
                events.flush()
                renderer.handle(event)
                write_json(args.logs / "latest-event.json", event.model_dump(mode="json"))
            result = await handle.result()
        finished = True
        write_json(args.logs / "result.json", result.model_dump(mode="json"))
        write_json(args.logs / "final-view.json", handle.session.view().model_dump(mode="json"))
    finally:
        if not finished:
            await settle(handle)
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(number)
    return 0 if result.succeeded else 1


def main() -> int:
    """Parse paths fixed in this run's script snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
