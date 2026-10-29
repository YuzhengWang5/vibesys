"""Pure experiment contracts and fail-closed pilot admission.

The public surface is the models, ``slots``, ``verify_ab`` and ``can_launch``.
External records reject unknown keys. Semantic single-factor auditing remains
an explicit operator attestation, never an inference from an agent's prose.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class Optimization(StrEnum):
    """The experiment's closed set of independent treatments."""

    CUDA_GRAPH = "cuda-graph"
    SPECULATIVE_DECODING = "speculative-decoding"
    FLASHINFER = "flashinfer"
    OPERATOR_FUSION = "operator-fusion"


class Status(StrEnum):
    """Durable attempt lifecycle; an uncertain launch is never replayed."""

    PREPARED = "prepared"
    LAUNCHING = "launching"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class Contract(BaseModel):
    """Strict external JSON boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def fail(message: str) -> NoReturn:
    """Raise a fail-closed operator boundary error with actionable detail."""
    raise ValueError(message)


class Slot(Contract):
    """One of the fixed pilot or twenty batch positions."""

    phase: Literal["pilot", "batch"]
    optimization: Optimization
    repeat: int = Field(ge=1, le=5)


class RunSpec(Contract):
    """Immutable attempt identity and exact source/input pins."""

    schema_version: Literal[1] = 1
    slot: Slot
    attempt: int = Field(ge=1)
    retry_of: Path | None = None
    source: Path
    experiment: Path
    run: Path
    run_id: str
    tested_commit: str
    branch: str
    candidate_commit: str
    objective_sha256: str
    trusted_sha256: dict[str, str]
    script_sha256: dict[str, str]
    python: Path
    model: Path
    max_run_seconds: int = Field(default=21600, ge=1)
    outer_loop: Literal["agent"] = "agent"
    provider: Literal["copilot"] = "copilot"
    agent_model: Literal["gpt-6.1-sol"] = "gpt-6.1-sol"
    max_rounds: Literal[5] = 5
    max_in_flight: Literal[1] = 1


class RunState(Contract):
    """Mutable operator record, written before each external launch."""

    status: Status = Status.PREPARED
    updated_at: str
    cuda_visible_devices: str | None = None
    returncode: int | None = None
    cleanup_ok: bool = False
    detail: str = "尚未启动; 没有测量结论。"


class Measurement(Contract):
    """One accepted trusted benchmark, paired with accuracy for its revision."""

    request_id: str
    evidence_id: str
    revision: str
    purpose: Literal["baseline", "official"]
    evaluator_digest: str
    workload_digest: str
    environment_digest: str
    value: FiniteFloat = Field(gt=0)


class Inspection(Contract):
    """A read-only projection of canonical core and receipt-ledger state."""

    run_id: str
    terminal_success: bool
    selected_revision: str | None
    core_revision: int
    measurements: tuple[Measurement, ...] = ()
    detail: str


class PilotReview(Contract):
    """Operator-owned evidence selection and code-path audit, not agent input."""

    baseline_revision: str
    candidate_revision: str
    baseline_request_id: str
    candidate_request_id: str
    baseline_evidence_id: str
    candidate_evidence_id: str
    reviewer: str = Field(min_length=1)
    audit: str = Field(min_length=20)
    eager_bf16_ar_baseline: Literal[True]
    only_assigned_optimization: Literal[True]
    production_logits_shared_forward: Literal[True]
    trusted_evaluation_unchanged: Literal[True]


class Approval(Contract):
    """Frozen admission record bound to a finished pilot's evidence and code."""

    pilot: Path
    tested_commit: str
    objective_sha256: str
    review: PilotReview
    baseline: Measurement
    candidate: Measurement
    gain: FiniteFloat = Field(gt=0)


def slots(phase: Literal["pilot", "batch"]) -> tuple[Slot, ...]:
    """Return one pilot or exactly five independent repeats per treatment."""
    if phase == "pilot":
        return (Slot(phase="pilot", optimization=Optimization.CUDA_GRAPH, repeat=1),)
    return tuple(
        Slot(phase="batch", optimization=optimization, repeat=repeat)
        for optimization in Optimization
        for repeat in range(1, 6)
    )


def can_launch(state: RunState) -> None:
    """Reject duplicate or uncertain execution; recovery requires a new attempt."""
    if state.status is not Status.PREPARED:
        fail(f"{state.status}: 不可重跑原 attempt; 查询证据后使用 retry 新建 run。")


def acknowledged_failures(
    phase: Literal["pilot", "batch"],
    requested: set[Path],
    failed: set[Path],
) -> set[Path]:
    """Only explicit existing batch failures may be skipped, never an unknown launch."""
    if requested and phase != "batch":
        fail("pilot 失败不能跳过。")
    if not requested <= failed:
        fail("--acknowledged-failure 必须指向本队列已失败的最新 attempt。")
    return requested


def verify_ab(inspection: Inspection, review: PilotReview) -> tuple[Measurement, Measurement]:
    """Require normal final adoption and genuinely comparable trusted A/B gain."""
    if not inspection.terminal_success:
        fail("pilot core 未正常成功终态/采纳验证失败; 不开放 batch。")
    if inspection.selected_revision != review.candidate_revision:
        fail("review candidate_revision 不是 core 最终验证采纳的 revision。")
    selected = []
    for role in ("baseline", "candidate"):
        matching = tuple(
            item
            for item in inspection.measurements
            if item.revision == getattr(review, f"{role}_revision")
            and item.request_id == getattr(review, f"{role}_request_id")
            and item.evidence_id == getattr(review, f"{role}_evidence_id")
        )
        if len(matching) != 1:
            fail(f"{role}: 缺少唯一的可信 accuracy+benchmark 证据。")
        selected.append(matching[0])
    baseline, candidate = selected
    if baseline.revision == candidate.revision:
        fail("A/B 必须使用不同实现 revision。")
    for field in ("evaluator_digest", "workload_digest", "environment_digest"):
        if getattr(baseline, field) != getattr(candidate, field):
            fail(f"A/B 的 {field} 不同, 不能对比。")
    if candidate.value <= baseline.value:
        fail("未观察到唯一优化的正收益; 不开放 batch。")
    return baseline, candidate


__all__ = [
    "Approval",
    "Inspection",
    "Measurement",
    "Optimization",
    "PilotReview",
    "RunSpec",
    "RunState",
    "Slot",
    "Status",
    "acknowledged_failures",
    "can_launch",
    "fail",
    "slots",
    "verify_ab",
]
