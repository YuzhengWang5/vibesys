"""The pilot stops only after trusted measurements and subsequent normal flow."""

from __future__ import annotations

import pytest
from scripts.scenario_c_pilot import PilotProgress

from vibesys.api import CoreEvent


def _event(kind: str, **fields: object) -> CoreEvent:
    return CoreEvent.model_validate({"timestamp": "2026-10-21T00:00:00Z", "type": kind, **fields})


def _gate(gate: str, *, status: str = "completed", value: float | None = None) -> CoreEvent:
    return _event(
        "gate_finished",
        status=status,
        data={"kind": "gate_finished", "gate": gate, "value": value},
    )


def _performance_start() -> CoreEvent:
    return _event(
        "invocation_started",
        execution_id="inv:perf-evaluator:perf-1",
        data={"kind": "invocation_started", "system_prompt": "", "user_prompt": ""},
    )


def _judge_pass() -> CoreEvent:
    return _event(
        "invocation_finished",
        execution_id="inv:judge:queue%2F000001.issue:0",
        data={
            "kind": "invocation_finished",
            "result": '{"reply":{"verdict":"pass"}}',
        },
    )


@pytest.mark.parametrize("accuracy_gate", ["validation", "accuracy"])
def test_pilot_requires_both_trusted_gates_and_next_workstream(accuracy_gate: str) -> None:
    progress = PilotProgress()
    assert not progress.observe(_performance_start())
    assert not progress.observe(_gate(accuracy_gate, status="failed"))
    assert not progress.observe(_gate("benchmark", value=23.5))
    assert not progress.observe(_performance_start())
    assert not progress.observe(_gate(accuracy_gate))
    assert not progress.observe(_performance_start())
    assert not progress.observe(_judge_pass())
    assert progress.observe(_performance_start())
    assert progress.benchmark_value == 23.5


@pytest.mark.parametrize("value", [None, 0.0, -1.0])
def test_pilot_rejects_missing_and_nonpositive_metrics(value: float | None) -> None:
    progress = PilotProgress()
    assert not progress.observe(_gate("validation"))
    assert not progress.observe(_judge_pass())
    assert not progress.observe(_gate("benchmark", value=value))
    assert not progress.observe(_performance_start())


def test_pilot_rejects_failed_benchmark_with_positive_metric() -> None:
    progress = PilotProgress()
    assert not progress.observe(_gate("validation"))
    assert not progress.observe(_judge_pass())
    assert not progress.observe(_gate("benchmark", status="failed", value=25.0))
    assert not progress.observe(_performance_start())


def test_pilot_stops_during_first_assessment_when_benchmark_finishes_there() -> None:
    progress = PilotProgress()
    assert not progress.observe(_gate("validation"))
    assert not progress.observe(_judge_pass())
    assert not progress.observe(_performance_start())
    assert progress.observe(_gate("benchmark", value=27.0))
    assert progress.next_workstream_started
