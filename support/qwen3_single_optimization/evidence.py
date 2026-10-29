"""Read canonical framework state through public Project and runtime APIs.

``inspect_run`` never starts an evaluator, agent, Docker container or GPU task.
Layout and codecs are the framework's; missing/quarantined/incompatible state
raises instead of being guessed from exit status or log text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qwen3_single_optimization.core import Inspection, Measurement, fail
from vibesys.api.catalog import MULTI
from vs_core.api import EvidenceKind, ObservationStatus, RunStatus
from vs_evaluation.api import EvidenceOutcome
from vs_project.api import Project, StoredEnvelope
from vs_runtime.api.core import ReceiptEvidenceLedger, ReceiptStore
from vs_sim.api import OsThreads

if TYPE_CHECKING:
    from pathlib import Path


def inspect_run(project_path: Path, run_id: str) -> Inspection:
    """Read typed core and trusted receipts; retain only accuracy-proven metrics."""
    project = Project.open(project_path)
    manifest = project.state.load_run(run_id)
    if manifest.orchestration.id != MULTI.plugin.id:
        fail(f"非本控制器的 outer-loop agent: {manifest.orchestration.id}")
    stored = project.stored_record(run_id)
    if not isinstance(stored, StoredEnvelope):
        fail("core record 缺失或 quarantine; 不推断成功。")
    projector = MULTI.projector
    if projector is None:
        fail("multi-agent registration 未提供 canonical record codec。")
    record = projector.record_type.decode(stored, projector.operations)
    core = record.envelope.core
    if core.run.run_id.root != run_id:
        fail("core run_id 与记录不符。")
    ledger = ReceiptEvidenceLedger(
        ReceiptStore(project.state.local_namespace(run_id, "receipts"), OsThreads())
    )
    readings = []
    for ref in core.evaluation.evidence:
        if (
            ref.kind is not EvidenceKind.BENCHMARK
            or ref.status is not ObservationStatus.SUCCEEDED
            or ref.provenance != "trusted"
            or ref.purpose not in {"baseline", "official"}
            or ref.acceptance_receipt is None
            or ref.candidate.git_commit is None
            or core.evaluation.accuracy_proof(ref.candidate) is None
        ):
            continue
        entry = ledger.lookup(ref.key)
        if entry is None or entry.purpose != ref.purpose:
            fail(f"证据 ledger 缺失或 purpose 不符: {ref.key}")
        if entry.evidence.outcome is EvidenceOutcome.FAILED:
            fail(f"core/ledger 成败不一致: {ref.key}")
        metrics = [item for item in entry.evidence.metrics if item.name == "median_tok_per_sec"]
        if len(metrics) != 1:
            fail(f"缺少唯一 median_tok_per_sec: {ref.key}")
        readings.append(
            Measurement(
                request_id=ref.source_request.root,
                evidence_id=ref.evidence_id.root,
                revision=ref.candidate.git_commit,
                purpose=ref.purpose,
                evaluator_digest=ref.evaluator_digest,
                workload_digest=ref.workload_digest,
                environment_digest=ref.environment_digest,
                value=metrics[0].value,
            )
        )
    result = core.run.result
    adoption = core.settlement.adoption
    successful = (
        core.run.status is RunStatus.TERMINAL
        and result is not None
        and result.outcome == "success"
        and adoption is not None
        and adoption.verified
        and adoption.failure is None
        and all(job.terminal and job.released for job in core.evaluation.jobs)
        and all(job.terminal and job.released for job in core.evaluation.registered_jobs)
        and not core.settlement.pending
    )
    selected = None if adoption is None else adoption.selection.revision.git_commit
    return Inspection(
        run_id=run_id,
        terminal_success=successful,
        selected_revision=selected,
        core_revision=core.revision,
        measurements=tuple(readings),
        detail=f"core={core.run.status}; result={result}; adoption={adoption}",
    )


__all__ = ["inspect_run"]
