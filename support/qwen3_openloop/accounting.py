"""Pure projections of public core events into measured optimization token costs."""

from __future__ import annotations

from collections import defaultdict

STAGES = (
    "baseline",
    "flashinfer",
    "fusion",
    "continuous-batching",
    "dynamic-batching",
    "prompt-ngram",
    "cuda-graph",
)
COUNTERS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
)


def optimization(hypothesis_id: str) -> str:
    """Require a stable declared prefix instead of inferring from arbitrary prose."""
    for stage in STAGES:
        if hypothesis_id.startswith(stage + "-"):
            return stage
    message = f"unattributable hypothesis id: {hypothesis_id!r}"
    raise ValueError(message)


def _reply_value(data: dict) -> dict:
    reply = data.get("reply")
    return reply.get("value", {}) if isinstance(reply, dict) else {}


def _identity(event: dict) -> str:
    identity = event.get("execution_id")
    if not isinstance(identity, str) or not identity:
        message = f"{event['type']} has no invocation identity"
        raise ValueError(message)
    return identity


def _totals(records: dict[str, dict]) -> tuple[dict, list[str]]:
    totals: dict[str, dict] = defaultdict(lambda: {"turns": 0, **dict.fromkeys(COUNTERS, 0)})
    missing = []
    for record in records.values():
        if record["usage"] is None:
            if record["status"] != "active":
                missing.append(record["execution_id"])
            continue
        bucket = totals[record["optimization"]]
        bucket["turns"] += 1
        for key in COUNTERS:
            bucket[key] += record["usage"][key]
    return dict(totals), missing


def _assign_planning(records: dict[str, dict], pending: set[str], stage: str) -> None:
    for invocation in pending:
        records[invocation]["optimization"] = stage
    pending.clear()


def summarize(
    events: list[dict], *, acknowledged_missing_usage: frozenset[str] = frozenset()
) -> dict:
    """Attribute every invocation once, including planning, judge, and retries.

    Planning before a design is backfilled to that design. Planning after the
    final design remains unassigned; it must not be silently charged elsewhere.
    Acknowledgements permit operator recovery, never supply missing counts.
    """
    stage = "baseline"
    records: dict[str, dict] = {}
    pending: set[str] = set()
    errors = []
    rounds = []
    for event in events:
        kind = event["type"]
        data = event.get("data") or {}
        if kind == "agent_execution_started":
            identity = _identity(event)
            role = event["agent_kind"]
            planning = role in ("multi-orchestrator", "multi-profiler")
            records[identity] = {
                "execution_id": identity,
                "role": role,
                "optimization": "unassigned-planning" if planning else stage,
                "started_at": event["timestamp"],
                "usage": None,
                "status": "active",
            }
            if planning:
                pending.add(identity)
        elif kind == "turn_usage_recorded":
            identity = _identity(event)
            record = records[identity]
            usage = {key: data[key] for key in COUNTERS}
            known = all(value is not None for value in usage.values())
            record["usage"] = (
                usage if known and usage["input_tokens"] + usage["output_tokens"] > 0 else None
            )
        elif kind == "agent_execution_finished":
            identity = _identity(event)
            record = records[identity]
            record["status"] = event.get("status")
            record["finished_at"] = event["timestamp"]
            value = _reply_value(data)
            if "hypothesis_id" in value:
                stage = optimization(value["hypothesis_id"])
                _assign_planning(records, pending, stage)
            if data.get("error"):
                errors.append(
                    {"sequence": event["sequence"], "kind": kind, "detail": data["error"]}
                )
        elif kind == "gate_finished" and event.get("status") in ("failed", "error"):
            errors.append({"sequence": event["sequence"], "kind": kind, "detail": data})
        elif kind == "round_finished":
            rounds.append({"sequence": event["sequence"], "optimization": stage, **data})
    totals, missing = _totals(records)
    return {
        "by_optimization": totals,
        "invocations": list(records.values()),
        "missing_usage": missing,
        "acknowledged_missing_usage": [
            identity for identity in missing if identity in acknowledged_missing_usage
        ],
        "unacknowledged_missing_usage": [
            identity for identity in missing if identity not in acknowledged_missing_usage
        ],
        "usage_complete": not missing
        and all(record["status"] != "active" for record in records.values()),
        "errors": errors,
        "rounds": rounds,
        "current_optimization": stage,
        "accounting": "input includes cached input; reasoning is a subset of output; do not add twice",
    }
