"""CPU-only contracts: objective isolation and evidence-gated admission."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from qwen3_single_optimization.controller import framework_args, queue_lock, verify_original_gate
from qwen3_single_optimization.core import (
    Inspection,
    Measurement,
    Optimization,
    PilotReview,
    RunSpec,
    RunState,
    Status,
    acknowledged_failures,
    can_launch,
    can_replace_approval,
    can_retry,
    slots,
    verify_ab,
)
from tests.support.task_project import write_task

from entrypoints import cli
from entrypoints.cli.loops import validated_run_request
from vibesys.api import agent_spec_from_config
from vibesys.api.catalog import MULTI
from vs_prompts.api import TemplateRenderer
from vs_sim.api.testing import ScriptedProbe

TEMPLATES = Path(__file__).parents[1] / "templates"
ROOT = Path(__file__).resolve().parents[3]
TASK = Path(".vibesys/tasks/default")


def test_original_gate_export_needs_no_calibration_or_observation_interface(tmp_path: Path) -> None:
    bundle = ROOT / "examples/model-serving/qwen3-32b-code-edit"
    for relative in ("accuracy_checker/checker.py", "vibesys.input.toml"):
        destination = tmp_path / TASK / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((bundle / TASK / relative).read_bytes())
    verify_original_gate(tmp_path)
    checker = tmp_path / TASK / "accuracy_checker/checker.py"
    checker.write_bytes(checker.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checker hash"):
        verify_original_gate(tmp_path)


def test_bundle_documents_match_the_single_optimization_templates() -> None:
    bundle = ROOT / "examples/model-serving/qwen3-32b-code-edit"
    renderer = TemplateRenderer(TEMPLATES)
    assert (bundle / TASK / "OBJECTIVE.md").read_text() == renderer.render_template(
        "OBJECTIVE.md.j2", optimization=Optimization.CUDA_GRAPH.value
    )
    for document, relative in (
        ("root", Path("README.md")),
        ("reference", TASK / "reference/README.md"),
        ("benchmark", TASK / "benchmark/README.md"),
    ):
        assert (bundle / relative).read_text() == renderer.render_template(
            "candidate-doc.md.j2", document=document
        )


@pytest.mark.parametrize(
    "extra", ['"--seed", "42"', '"--num-samples", "50"', '"--max-tokens", "8"']
)
def test_export_rejects_accuracy_default_overrides(tmp_path: Path, extra: str) -> None:
    bundle = ROOT / "examples/model-serving/qwen3-32b-code-edit"
    checker = tmp_path / TASK / "accuracy_checker/checker.py"
    checker.parent.mkdir(parents=True)
    checker.write_bytes((bundle / TASK / "accuracy_checker/checker.py").read_bytes())
    manifest = (bundle / TASK / "vibesys.input.toml").read_text()
    (tmp_path / TASK / "vibesys.input.toml").write_text(
        manifest.replace(
            '"--tokenizer-path", "/model"]', f'"--tokenizer-path", "/model", {extra}]', 1
        )
    )
    with pytest.raises(ValueError, match="默认参数"):
        verify_original_gate(tmp_path)


def test_parallel_treatments_exclude_duplicates_and_full_queue(tmp_path: Path) -> None:
    with (
        queue_lock(tmp_path, Optimization.CUDA_GRAPH),
        queue_lock(tmp_path, Optimization.FLASHINFER),
    ):
        with (
            pytest.raises(BlockingIOError, match="unavailable"),
            queue_lock(tmp_path, Optimization.CUDA_GRAPH),
        ):
            pytest.fail("duplicate queue admitted")
        with pytest.raises(BlockingIOError, match="unavailable"), queue_lock(tmp_path, None):
            pytest.fail("full queue admitted beside treatments")
    with (
        queue_lock(tmp_path, None),
        pytest.raises(BlockingIOError, match="unavailable"),
        queue_lock(tmp_path, Optimization.CUDA_GRAPH),
    ):
        pytest.fail("treatment queue admitted beside full queue")
    with queue_lock(tmp_path, Optimization.CUDA_GRAPH):
        pass


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
    assert "五次重复" not in rendered
    assert "自然产生" not in rendered
    assert "生产 forward" in rendered
    assert "eager BF16" in rendered
    assert "真实 Hugging Face" in rendered
    assert "/v1/logprobs" not in rendered
    assert "return_token_ids" not in rendered
    assert "A/B" not in rendered


def test_batch_repeats_one_macro_goal_without_old_combination() -> None:
    assert len(slots("pilot")) == 1
    batch = slots("batch")
    assert len(batch) == 12
    for optimization in Optimization:
        assert [item.repeat for item in batch if item.optimization == optimization] == list(
            range(1, 4)
        )
    assert [(item.repeat, item.optimization) for item in batch] == [
        (repeat, optimization) for repeat in range(1, 4) for optimization in Optimization
    ]
    renderer = TemplateRenderer(TEMPLATES)
    for document in ("root", "reference", "benchmark"):
        rendered = renderer.render_template("candidate-doc.md.j2", document=document)
        assert not any(item.value in rendered for item in Optimization)
        assert "K16" not in rendered
        assert "use_cuda_graph" not in rendered
        assert "/v1/logprobs" not in rendered
        assert "return_token_ids" not in rendered


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


@pytest.mark.parametrize("status", Status)
@pytest.mark.parametrize("changed", [False, True])
def test_retry_never_replays_active_or_same_checkpoint_prepared_attempts(
    status: Status, *, changed: bool
) -> None:
    state = RunState(status=status, updated_at="fixed-input")
    current = "new" if changed else "old"
    if status in {Status.FINISHED, Status.FAILED} or (status is Status.PREPARED and changed):
        can_retry(state, "old", current)
    else:
        with pytest.raises(ValueError, match="不可 retry"):
            can_retry(state, "old", current)


@pytest.mark.parametrize("previous", [None, "old", "new"])
@pytest.mark.parametrize("acknowledged", [None, "old", "new", "wrong"])
def test_reapproval_requires_explicit_obsolete_checkpoint(
    previous: str | None, acknowledged: str | None
) -> None:
    allowed = (previous is None and acknowledged is None) or (
        previous == "old" and acknowledged == "old"
    )
    if allowed:
        can_replace_approval(previous, "new", acknowledged)
    else:
        with pytest.raises(ValueError, match=r"approval|checkpoint"):
            can_replace_approval(previous, "new", acknowledged)


def test_fixed_five_round_public_launch_contract(tmp_path: Path) -> None:
    spec = RunSpec(
        slot=slots("pilot")[0],
        attempt=1,
        source=Path("/source"),
        experiment=Path("/experiment"),
        run=tmp_path / "pilot",
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
    write_task(
        spec.run / "project",
        'version = 1\nbackends = ["cuda"]\n[agent]\ndomain = "llm-serving"\n'
        '[accuracy]\ncommand = ["python", "checker.py"]\n'
        '[benchmark]\ncommand = ["python", "benchmark.py"]\n',
    )
    config_path = tmp_path / "agent.toml"
    config_path.write_text('[model]\nname = "overridden-by-global-cli-model"\n')
    invocation = cli.parse_cli_invocation([*args[3:], "--config", str(config_path)])
    probe = ScriptedProbe(lambda _argv: None)
    request, config = validated_run_request(invocation.args, probe=probe)
    assert request.orchestration.id == MULTI.plugin.id
    assert MULTI.plugin.agents
    assert config.agent.roles == {}
    default = agent_spec_from_config(
        config,
        backend=request.agent_backend,
        provider=request.cli_provider,
        allow_unvetted_model=request.allow_unvetted_model,
    )
    assert default.provider == "copilot"
    assert default.model == "gpt-6.1-sol"
    assert all(role.id not in config.agent.roles for role in MULTI.plugin.agents)
    assert probe.calls == []
    with pytest.raises(ValidationError):
        RunSpec.model_validate({**spec.model_dump(), "max_rounds": 24})


def test_acknowledgement_never_skips_pilot_or_unfinished_trials() -> None:
    failed = Path("/failed")
    unfinished = Path("/unknown")
    assert acknowledged_failures("batch", {failed}, {failed}) == {failed}
    assert acknowledged_failures("batch", set(), {failed}) == set()
    with pytest.raises(ValueError, match="pilot"):
        acknowledged_failures("pilot", {failed}, {failed})
    with pytest.raises(ValueError, match="最新 attempt"):
        acknowledged_failures("batch", {unfinished}, {failed})
