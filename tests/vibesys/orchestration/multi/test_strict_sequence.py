"""Strict rounds finish one reviewed, measured intervention or fail without advancing."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.vibesys.orchestration.multi._host import (
    IMPLEMENTER,
    JUDGE,
    PLANNER,
    MultiHost,
    config,
    implemented,
    reviewed,
)
from tests.vibesys.search_faults import assert_flow_closed
from tests.vibesys.serial_search._replies import plan_reply
from tests.vibesys.vibesys_flows import across_restarts

from vibesys.hypothesis import OrchestratorPlan
from vs_core.api import Finish, Retained, RevisionRef
from vs_sim.api.testing import examples


def design(host: MultiHost, identifier: str, **changes: object) -> None:
    """Answer the strict designer with an explicit, falsifiable single intervention."""
    host.drain_until_turn(PLANNER)
    host.run.reply_raw(
        host.run.expect_turn(role=PLANNER),
        plan_reply(
            identifier,
            invariants="All mechanisms except this plan remain unchanged.",
            activation_evidence="The single changed production path is exercised.",
            falsification_criteria="Official correctness fails or the path is inactive.",
            minimum_acceptance_criteria="Complete the path and measure official throughput.",
            **changes,
        ),
    )


def implement(host: MultiHost, candidate: str, outcome: str = "nominated") -> None:
    host.drain_until_turn(IMPLEMENTER)
    host.run.reply(
        host.run.expect_turn(role=IMPLEMENTER),
        implemented(outcome, next_step="repair same mechanism" if outcome == "continue" else ""),
        checkpoint=Retained(revision=RevisionRef.of_git_commit(candidate)),
    )


def judge(host: MultiHost, *, passed: bool = True) -> None:
    host.drain_until_turn(JUDGE)
    host.run.reply(host.run.expect_turn(role=JUDGE), reviewed(passed=passed))


@across_restarts
def test_unfinished_work_is_repaired_inside_one_round_without_redesign(*, restarting: bool) -> None:
    host = MultiHost.start(
        config(strict_sequence=True, max_retries_per_round=3), restarting=restarting
    )
    design(host, "one")
    implement(host, "partial", "continue")
    implement(host, "rejected")
    judge(host, passed=False)
    implement(host, "complete")
    judge(host)
    host.drain()

    assert len(host.opened) == 1
    assert host.measured == ["complete"]
    assert host.record is not None
    assert len(host.record.hypotheses) == 1
    assert len(host.record.hypotheses[0].rounds) == 1
    finish = host.run.finished
    assert isinstance(finish, Finish)
    assert finish.result.outcome == "success"
    assert_flow_closed(host)


@across_restarts
@settings(max_examples=examples(pr=4, nightly=12))
@given(turns=st.integers(1, 4))
def test_exhaustion_is_a_run_failure_not_another_round_or_baseline_success(
    *, restarting: bool, turns: int
) -> None:
    host = MultiHost.start(
        config(strict_sequence=True, max_rounds=3, max_retries_per_round=turns),
        restarting=restarting,
    )
    design(host, "one")
    for number in range(turns):
        implement(host, f"partial-{number}", "continue")
    host.drain()

    assert len(host.opened) == 1
    assert host.measured == []
    finish = host.run.finished
    assert isinstance(finish, Finish)
    assert finish.result.outcome != "success"
    assert "strict_sequence" in str(finish.result)
    assert_flow_closed(host)


@across_restarts
def test_a_later_failure_is_not_hidden_by_an_earlier_success(*, restarting: bool) -> None:
    host = MultiHost.start(config(strict_sequence=True, max_rounds=3), restarting=restarting)
    design(host, "one")
    implement(host, "complete")
    judge(host)
    design(host, "two")
    implement(host, "incomplete", "continue")
    host.drain()

    assert len(host.opened) == 2
    assert host.measured == ["complete"]
    finish = host.run.finished
    assert isinstance(finish, Finish)
    assert finish.result.outcome != "success"
    assert_flow_closed(host)


@across_restarts
def test_next_round_uses_last_completed_revision_not_best_or_input(*, restarting: bool) -> None:
    host = MultiHost.start(
        config(strict_sequence=True, max_rounds=3),
        restarting=restarting,
        throughputs={"fast": 120.0, "slower": 90.0},
    )
    for identifier in ("fast", "slower"):
        design(host, identifier)
        implement(host, identifier)
        judge(host)
    host.drain_until_turn(PLANNER)
    prompt = host.run.expect_turn(role=PLANNER).context
    assert getattr(prompt, "base_revision", None) == RevisionRef.of_git_commit("slower")
    design(host, "third", base_round=0)
    host.drain_until_turn(PLANNER)
    correction = host.run.expect_turn(role=PLANNER).context
    assert "strict_sequence requires base_round 2" in str(getattr(correction, "problem", None))


@across_restarts
def test_unavailable_validation_fails_instead_of_approving_or_advancing(
    *, restarting: bool
) -> None:
    host = MultiHost.start(
        config(strict_sequence=True, max_rounds=2),
        restarting=restarting,
        validating=True,
        validation="errored",
    )
    design(host, "one")
    host.drain_until_turn(IMPLEMENTER)
    host.run.reply(
        host.run.expect_turn(role=IMPLEMENTER),
        implemented(validation_recipe_artifact="validation/recipes.json"),
        checkpoint=Retained(revision=RevisionRef.of_git_commit("candidate")),
    )
    judge(host)
    host.drain()
    assert len(host.opened) == 1
    assert not host.measured
    finish = host.run.finished
    assert isinstance(finish, Finish)
    assert finish.result.outcome != "success"
    assert_flow_closed(host)


@pytest.mark.parametrize(
    "field",
    [
        "hypothesis",
        "task",
        "pass_criteria",
        "invariants",
        "activation_evidence",
        "falsification_criteria",
        "minimum_acceptance_criteria",
    ],
)
def test_strict_plan_requires_a_testable_intervention(field: str) -> None:
    host = MultiHost.start(config(strict_sequence=True), restarting=False)
    host.drain_until_turn(PLANNER)
    reply = OrchestratorPlan.model_validate_json(
        plan_reply(
            "incomplete",
            invariants="fixed",
            activation_evidence="active",
            falsification_criteria="inactive",
            minimum_acceptance_criteria="measured",
        )
    )
    host.run.reply(host.run.expect_turn(role=PLANNER), reply.model_copy(update={field: " "}))
    host.drain_until_turn(PLANNER)
    assert f"strict_sequence requires nonblank {field}" in str(
        getattr(host.run.expect_turn(role=PLANNER).context, "problem", None)
    )
