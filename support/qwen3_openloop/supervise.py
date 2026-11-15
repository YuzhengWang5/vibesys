"""Own one framework process and persist external health/token observations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from qwen3_openloop.accounting import summarize
from vs_project.api import Project
from vs_sim.api import InheritedStdioLauncher, Signals, SystemClock

LOGGER = logging.getLogger(__name__)
MAX_ROUNDS = 20
MAX_IDLE_SECONDS = 12000


def save(path: Path, payload: dict) -> None:
    """Atomically publish a complete snapshot for readers."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_events(path: Path) -> list[dict]:
    """Read only complete JSONL records; a concurrent final write may be partial."""
    if not path.exists():
        return []
    contents = path.read_bytes()
    complete = contents[: contents.rfind(b"\n") + 1]
    return [json.loads(line) for line in complete.splitlines()]


def _publish(run: Path, report: dict, status: dict, seen_errors: set[int]) -> None:
    save(run / "logs/token-costs.json", report)
    save(run / "logs/monitor-status.json", status)
    for error in report["errors"]:
        if error["sequence"] not in seen_errors:
            seen_errors.add(error["sequence"])
            with (run / "logs/alerts.jsonl").open("a") as stream:
                stream.write(json.dumps(error, ensure_ascii=False) + "\n")
            LOGGER.error("Framework fault: %s", error)


async def execute(
    run: Path, run_id: str, *, acknowledged_missing_usage: frozenset[str] = frozenset()
) -> int:
    """Watch every 30 seconds; surface faults while framework-owned retries run."""
    os.environ["VIBESYS_STATE_HOME"] = str(run / "machine-local")
    project = Project.open(run / "project")
    core_log = project.state.log_directory(run_id) / "core-events.jsonl"
    child = await InheritedStdioLauncher().start(["bash", str(run / "scripts/launch.sh")])
    pending = asyncio.create_task(child.wait())
    clock = SystemClock()
    last_progress = clock.now()
    sequence = -1
    seen_errors: set[int] = set()
    fatal = False
    if acknowledged_missing_usage:
        LOGGER.warning(
            "Operator acknowledged missing CLI usage; costs remain incomplete: %s",
            sorted(acknowledged_missing_usage),
        )
    try:
        while True:
            events = read_events(core_log)
            report = summarize(events, acknowledged_missing_usage=acknowledged_missing_usage)
            if events and events[-1]["sequence"] != sequence:
                sequence = events[-1]["sequence"]
                last_progress = clock.now()
            status = {
                "framework_exited": pending.done(),
                "last_event_sequence": sequence,
                "idle_seconds": clock.now() - last_progress,
                "current_optimization": report["current_optimization"],
                "rounds_finished": len(report["rounds"]),
                "missing_usage": report["missing_usage"],
                "acknowledged_missing_usage": report["acknowledged_missing_usage"],
                "unacknowledged_missing_usage": report["unacknowledged_missing_usage"],
                "usage_complete": report["usage_complete"],
                "latest_gate": next(
                    (
                        event
                        for event in reversed(events)
                        if event["type"] in ("gate_started", "gate_finished")
                    ),
                    None,
                ),
            }
            _publish(run, report, status, seen_errors)
            if report["unacknowledged_missing_usage"] or len(report["rounds"]) > MAX_ROUNDS:
                LOGGER.error("Accounting/budget invariant failed: %s", status)
                fatal = True
            if clock.now() - last_progress > MAX_IDLE_SECONDS:
                LOGGER.error("No framework event within longest evaluation budget")
                fatal = True
            if fatal and not pending.done():
                child.send_signal(Signals.SIGINT)
                break
            if pending.done():
                break
            await asyncio.wait({pending}, timeout=30)
        code = await asyncio.wait_for(asyncio.shield(pending), timeout=600)
        events = read_events(core_log)
        save(
            run / "logs/token-costs.json",
            summarize(events, acknowledged_missing_usage=acknowledged_missing_usage),
        )
        save(run / "logs/monitor-exit.json", {"framework_returncode": code, "monitor_fault": fatal})
        return 2 if fatal else code
    finally:
        if not pending.done():
            child.send_signal(Signals.SIGINT)
            try:
                await asyncio.wait_for(asyncio.shield(pending), timeout=600)
            except TimeoutError:
                child.send_signal(Signals.SIGTERM)
                await asyncio.wait_for(asyncio.shield(pending), timeout=180)


def main() -> None:
    """Supervise a prepared run; every failure remains visible to the operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--acknowledge-missing-usage",
        action="append",
        default=[],
        metavar="INVOCATION_ID",
        help="Explicitly accept this invocation's unknown cost on recovery; never fills it with zero.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(
        asyncio.run(
            execute(
                args.run.resolve(),
                args.run_id,
                acknowledged_missing_usage=frozenset(args.acknowledge_missing_usage),
            )
        )
    )


if __name__ == "__main__":
    main()
