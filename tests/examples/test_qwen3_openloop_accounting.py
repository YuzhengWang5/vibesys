"""Missing failed-turn usage remains explicit and does not erase measured calls."""

from hypothesis import given
from hypothesis import strategies as st
from qwen3_openloop.accounting import COUNTERS, summarize

MEASURED = {
    "input_tokens": 100,
    "output_tokens": 10,
    "cache_read_input_tokens": 40,
    "cache_creation_input_tokens": 10,
    "reasoning_output_tokens": 2,
}


def invocation(identity: str, usage: dict, *, failed: bool) -> list[dict]:
    return [
        {
            "type": "agent_execution_started",
            "execution_id": identity,
            "agent_kind": "multi-implementer",
            "timestamp": "2026-11-14T20:00:00Z",
        },
        {"type": "turn_usage_recorded", "execution_id": identity, "data": usage},
        {
            "type": "agent_execution_finished",
            "execution_id": identity,
            "timestamp": "2026-11-14T20:01:00Z",
            "sequence": 3,
            "status": "failed" if failed else "completed",
            "data": {"error": "interrupted" if failed else None},
        },
    ]


@given(missing=st.sets(st.sampled_from(COUNTERS), min_size=1))
def test_missing_failed_turn_counters_are_flagged_without_losing_known_usage(
    missing: set[str],
) -> None:
    incomplete = {key: None if key in missing else value for key, value in MEASURED.items()}
    report = summarize(
        invocation("measured", MEASURED, failed=False)
        + invocation("interrupted", incomplete, failed=True)
    )

    assert report["missing_usage"] == ["interrupted"]
    assert report["by_optimization"]["baseline"] == {"turns": 1, **MEASURED}
    assert report["invocations"][-1]["usage"] is None
    assert report["invocations"][-1]["status"] == "failed"
    assert report["errors"][-1]["detail"] == "interrupted"


@given(missing=st.sets(st.sampled_from(COUNTERS), min_size=1))
def test_acknowledgement_keeps_cost_unknown_and_does_not_accept_a_new_missing_call(
    missing: set[str],
) -> None:
    incomplete = {key: None if key in missing else value for key, value in MEASURED.items()}
    events = invocation("measured", MEASURED, failed=False) + invocation(
        "interrupted", incomplete, failed=True
    )
    strict = summarize(events)
    accepted = summarize(events, acknowledged_missing_usage=frozenset({"interrupted"}))
    assert strict["unacknowledged_missing_usage"] == ["interrupted"]
    assert accepted["unacknowledged_missing_usage"] == []
    assert accepted["acknowledged_missing_usage"] == ["interrupted"]
    assert accepted["missing_usage"] == ["interrupted"]
    assert not accepted["usage_complete"]
    assert accepted["invocations"] == strict["invocations"]
    assert accepted["by_optimization"] == strict["by_optimization"]

    resumed = summarize(
        events + invocation("new-failure", incomplete, failed=True),
        acknowledged_missing_usage=frozenset({"interrupted"}),
    )
    assert resumed["unacknowledged_missing_usage"] == ["new-failure"]
    assert resumed["missing_usage"] == ["interrupted", "new-failure"]
    assert resumed["by_optimization"]["baseline"] == {"turns": 1, **MEASURED}
