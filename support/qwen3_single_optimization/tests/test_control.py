"""CPU-only contracts: objective isolation and evidence-gated admission."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from qwen3_single_optimization.controller import framework_args
from qwen3_single_optimization.core import (
    Inspection,
    Measurement,
    Optimization,
    PilotReview,
    RunSpec,
    RunState,
    Status,
    can_launch,
    slots,
    verify_ab,
)

from vs_prompts.api import TemplateRenderer

TEMPLATES = Path(__file__).parents[1] / "templates"


def example_review() -> PilotReview:
    return PilotReview(
        baseline_revision="a" * 40,
        candidate_revision="b" * 40,
        baseline_request_id="baseline-request",
        candidate_request_id="candidate-request",
        baseline_evidence_id="baseline-evidence",
        candidate_evidence_id="candidate-evidence",
        reviewer="operator",
        audit="审查两份不可变 revision 的真实生产路径, 唯一因素改变且基线是 eager BF16 AR。",
        eager_bf16_ar_baseline=True,
        only_assigned_optimization=True,
        production_logits_shared_forward=True,
        trusted_evaluation_unchanged=True,
    )


def example_inspection(baseline: float = 10.0, candidate: float = 12.0) -> Inspection:
    review = example_review()
    common = {
        "purpose": "official",
        "evaluator_digest": "same-checker",
        "workload_digest": "same-workload",
        "environment_digest": "same-device-and-env",
    }
    return Inspection(
        run_id="pilot",
        terminal_success=True,
        selected_revision=review.candidate_revision,
        core_revision=5,
        detail="已验证采纳与资源终态。",
        measurements=(
            Measurement(
                **common,
                revision=review.baseline_revision,
                request_id=review.baseline_request_id,
                evidence_id=review.baseline_evidence_id,
                value=baseline,
            ),
            Measurement(
                **common,
                revision=review.candidate_revision,
                request_id=review.candidate_request_id,
                evidence_id=review.candidate_evidence_id,
                value=candidate,
            ),
        ),
    )


@pytest.mark.parametrize("optimization", Optimization)
def test_objective_names_only_the_assigned_treatment(optimization: Optimization) -> None:
    rendered = TemplateRenderer(TEMPLATES).render_template(
        "OBJECTIVE.md.j2", optimization=optimization.value
    )
    assert optimization.value in rendered
    assert all(item.value not in rendered for item in Optimization if item != optimization)
    assert "K=16" not in rendered
    assert "生产 forward" in rendered
    assert "eager BF16" in rendered
    assert "HF 旁路" in rendered


def test_batch_repeats_one_macro_goal_without_old_combination() -> None:
    assert len(slots("pilot")) == 1
    batch = slots("batch")
    assert len(batch) == 20
    for optimization in Optimization:
        assert [item.repeat for item in batch if item.optimization == optimization] == list(
            range(1, 6)
        )
    renderer = TemplateRenderer(TEMPLATES)
    for document in ("root", "reference", "benchmark"):
        rendered = renderer.render_template("candidate-doc.md.j2", document=document)
        assert not any(item.value in rendered for item in Optimization)
        assert "K16" not in rendered
        assert "use_cuda_graph" not in rendered


@given(
    baseline=st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
)
def test_trusted_positive_gain_is_required(baseline: float, delta: float) -> None:
    review = example_review()
    left, right = verify_ab(example_inspection(baseline, baseline + delta), review)
    assert right.value > left.value
    with pytest.raises(ValueError, match="正收益"):
        verify_ab(example_inspection(baseline + delta, baseline), review)
    with pytest.raises(ValueError, match="正收益"):
        verify_ab(example_inspection(baseline, baseline), review)


@pytest.mark.parametrize(
    "change",
    [
        {"terminal_success": False},
        {"selected_revision": "c" * 40},
        {"measurements": ()},
    ],
)
def test_zero_exit_cannot_replace_real_framework_success(change: dict) -> None:
    with pytest.raises(ValueError, match=r"终态|revision|可信"):
        verify_ab(example_inspection().model_copy(update=change), example_review())


@pytest.mark.parametrize("field", ["evaluator_digest", "workload_digest", "environment_digest"])
def test_incomparable_a_b_is_rejected(field: str) -> None:
    inspection = example_inspection()
    first, second = inspection.measurements
    inspection = inspection.model_copy(
        update={"measurements": (first, second.model_copy(update={field: "different"}))}
    )
    with pytest.raises(ValueError, match=field):
        verify_ab(inspection, example_review())


def test_missing_or_duplicate_evidence_never_grants_admission() -> None:
    inspection = example_inspection()
    duplicate = inspection.model_copy(
        update={"measurements": (*inspection.measurements, inspection.measurements[0])}
    )
    with pytest.raises(ValueError, match="唯一"):
        verify_ab(duplicate, example_review())
    with pytest.raises(ValidationError):
        PilotReview.model_validate(
            {**example_review().model_dump(), "only_assigned_optimization": False}
        )
    with pytest.raises(ValidationError):
        PilotReview.model_validate({**example_review().model_dump(), "relax_accuracy": True})


@pytest.mark.parametrize("status", Status)
def test_only_prepared_attempt_can_launch(status: Status) -> None:
    state = RunState(status=status, updated_at="fixed-input")
    if status is Status.PREPARED:
        can_launch(state)
    else:
        with pytest.raises(ValueError, match="retry"):
            can_launch(state)


def test_fixed_five_round_public_launch_contract() -> None:
    spec = RunSpec(
        slot=slots("pilot")[0],
        attempt=1,
        source=Path("/source"),
        experiment=Path("/experiment"),
        run=Path("/experiment/runs/pilot"),
        run_id="pilot",
        tested_commit="a" * 40,
        branch="test",
        candidate_commit="b" * 40,
        objective_sha256="hash",
        trusted_sha256={},
        script_sha256={},
        python=Path("/python"),
        model=Path("/model"),
    )
    args = framework_args(spec)
    for flag, value in {
        "--outer-loop": "agent",
        "--cli-provider": "copilot",
        "--model": "gpt-6.1-sol",
        "--max-rounds": "5",
        "--max-in-flight": "1",
        "--evaluation-slots": "1",
    }.items():
        assert args[args.index(flag) + 1] == value
    assert "--no-skills" not in args
    with pytest.raises(ValidationError):
        RunSpec.model_validate({**spec.model_dump(), "max_rounds": 24})
