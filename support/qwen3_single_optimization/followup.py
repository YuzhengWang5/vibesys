"""Prepare eight isolated follow-up candidates and dispatch one immutable run."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from qwen3_single_optimization.controller import (
    GRACE_SECONDS,
    LEASE_WAIT_SECONDS,
    checkpoint,
    interrupt,
    locked,
    prepare_run,
    read_spec,
    read_state,
    records,
    run_process,
    timestamp,
    validate_inputs,
)
from qwen3_single_optimization.core import Optimization, Slot, Status, can_launch, fail

__all__ = ["followup_slots", "main"]


def followup_slots() -> tuple[tuple[Slot, str], ...]:
    """Return two full-decode trials and three guided trials per assigned treatment."""
    graph = Optimization.CUDA_GRAPH
    speculation = Optimization.SPECULATIVE_DECODING
    return (
        *(
            (Slot(phase="batch", optimization=graph, repeat=repeat), "full-decode")
            for repeat in (1, 2)
        ),
        *(
            (Slot(phase="batch", optimization=graph, repeat=repeat), "guided")
            for repeat in (3, 4, 5)
        ),
        *(
            (Slot(phase="batch", optimization=speculation, repeat=repeat), "guided")
            for repeat in (1, 2, 3)
        ),
    )


def prepare(experiment: Path) -> None:
    """Prepare inputs from a clean checkpoint without inheriting learned implementations."""
    source = Path(os.environ["QWEN3_SOURCE"]).resolve()
    checkpoint(source)
    with locked(experiment / ".prepare.lock"):
        if any((experiment / "runs").glob("*/scripts/spec.json")):
            fail("follow-up inputs already prepared; do not overwrite or duplicate")
        for slot, variant in followup_slots():
            spec = prepare_run(source, experiment, slot, objective_variant=variant)
            sys.stdout.write(str(spec.run) + "\n")


def dispatch(run: Path) -> None:
    """Record intent before gpuctl, then consume it using the existing bounded worker."""
    spec = read_spec(run.resolve())
    if (spec.slot, spec.objective_variant) not in followup_slots():
        fail("run is not one of the declared follow-up treatments")
    with locked(spec.run / ".dispatch.lock"):
        state = read_state(spec.run)
        can_launch(state)
        validate_inputs(spec)
        records(
            spec,
            state.model_copy(
                update={
                    "status": Status.LAUNCHING,
                    "updated_at": timestamp(),
                    "detail": "follow-up single-run dispatch waiting for gpuctl lease",
                }
            ),
        )
        try:
            result = run_process(
                ["bash", str(spec.run / "scripts/launch.sh")],
                spec.run / "logs/dispatch.log",
                cwd=spec.run,
                timeout=LEASE_WAIT_SECONDS + spec.max_run_seconds + 4 * GRACE_SECONDS,
            )
        except BaseException as error:
            mark_unstarted_failure(spec.run, f"dispatch error: {type(error).__name__}: {error}")
            raise
        final = read_state(spec.run)
        if result != 0 or final.status is not Status.FINISHED:
            mark_unstarted_failure(spec.run, f"gpuctl/worker exit={result}")
            fail(f"run did not finish successfully: {spec.run}; retain evidence for review")


def mark_unstarted_failure(run: Path, detail: str) -> None:
    """Only settle an unconsumed launch; never overwrite worker terminal facts."""
    spec = read_spec(run)
    state = read_state(run)
    if state.status is Status.LAUNCHING:
        records(
            spec,
            state.model_copy(
                update={"status": Status.FAILED, "updated_at": timestamp(), "detail": detail}
            ),
        )


def main() -> None:
    """Expose prepare/dispatch; runtime, evaluation and provider policy are unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dispatch"))
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.experiment is None:
            parser.error("prepare requires --experiment")
        prepare(args.experiment.resolve())
    else:
        if args.run is None:
            parser.error("dispatch requires --run")
        dispatch(args.run.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGTERM, interrupt)
    main()
