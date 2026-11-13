"""Public event attribution keeps retries and design costs without double counts."""

import runpy
from pathlib import Path

import pytest

API = runpy.run_path(str(Path(__file__).parents[1] / "accounting.py"))


def invocation(index: int, role: str, *, hypothesis: str | None = None) -> list[dict]:
    common = {
        "execution_id": str(index),
        "agent_kind": role,
        "timestamp": str(index),
        "sequence": index,
    }
    return [
        {**common, "type": "agent_execution_started", "data": {}},
        {
            **common,
            "type": "turn_usage_recorded",
            "data": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 40,
                "reasoning_output_tokens": 5,
            },
        },
        {
            **common,
            "type": "agent_execution_finished",
            "status": "completed",
            "data": {
                "reply": {"value": {} if hypothesis is None else {"hypothesis_id": hypothesis}},
            },
        },
    ]


def test_planning_and_resumed_retries_follow_specific_optimization() -> None:
    events = [
        *invocation(0, "multi-orchestrator"),
        *invocation(1, "multi-orchestrator", hypothesis="flashinfer-attention"),
        *invocation(2, "multi-implementer"),
        *invocation(3, "multi-implementer"),
        *invocation(4, "multi-judge"),
        *invocation(5, "multi-orchestrator", hypothesis="fusion-norm"),
        *invocation(6, "multi-implementer"),
    ]
    report = API["summarize"](events)
    assert report["by_optimization"]["flashinfer"]["turns"] == 5
    assert report["by_optimization"]["flashinfer"]["input_tokens"] == 500
    assert report["by_optimization"]["fusion"]["turns"] == 2
    assert not report["missing_usage"]
    assert API["summarize"](events) == report


def test_unknown_counts_are_not_zero_cost_and_unmatched_planning_stays_separate() -> None:
    events = invocation(0, "multi-implementer")
    events[1]["data"] = dict.fromkeys(events[1]["data"], 0)
    events.extend(invocation(1, "multi-orchestrator"))
    report = API["summarize"](events)
    assert report["missing_usage"] == ["0"]
    assert "baseline" not in report["by_optimization"]
    assert report["by_optimization"]["unassigned-planning"]["turns"] == 1


def test_unknown_stage_is_not_guessed() -> None:
    with pytest.raises(ValueError, match="unattributable"):
        API["optimization"]("arbitrary-name")


def test_agent_can_revisit_optimizations_and_choose_dynamic_batching_first() -> None:
    events = [
        *invocation(0, "multi-orchestrator", hypothesis="dynamic-batching-admission"),
        *invocation(1, "multi-implementer"),
        *invocation(2, "multi-orchestrator", hypothesis="prompt-ngram-source"),
        *invocation(3, "multi-implementer"),
        *invocation(4, "multi-orchestrator", hypothesis="dynamic-batching-tune"),
        *invocation(5, "multi-judge"),
    ]
    report = API["summarize"](events)
    assert report["by_optimization"]["dynamic-batching"]["input_tokens"] == 400
    assert report["by_optimization"]["prompt-ngram"]["input_tokens"] == 200
    assert not report["missing_usage"]
