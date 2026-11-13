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


def summarize(events: list[dict]) -> dict:
    """Attribute every invocation once, including planning, judge, and retries.

    Planning before a design is backfilled to that design. Planning after the
    final design remains unassigned; it must not be silently charged elsewhere.
    """
    stage = "baseline"
    records: dict[str, dict] = {}
    pending: set[str] = set()
    errors = []
    rounds = []
    for event in events:
        kind = event["type"]
        data = event.get("data") or {}
        identity = event.get("execution_id")
        if kind == "agent_execution_started":
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
            record = records[identity]
            usage = {key: data[key] for key in COUNTERS}
            record["usage"] = usage if usage["input_tokens"] + usage["output_tokens"] > 0 else None
        elif kind == "agent_execution_finished":
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
        "errors": errors,
        "rounds": rounds,
        "current_optimization": stage,
        "accounting": "input includes cached input; reasoning is a subset of output; do not add twice",
    }
