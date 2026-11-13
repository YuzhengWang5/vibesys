"""Teacher-forced gates tolerate near ties without accepting wrong probabilities."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

# test-isolation: vs-bench publishes numerical modules directly, not through an api package.
from vs_bench.logprobs import LogprobTolerance, TokenLogprobs, compare_logprobs, summarize_logits


@given(st.floats(min_value=-100, max_value=100, allow_nan=False))
def test_logprob_gate_ignores_additive_logit_offsets(offset: float) -> None:
    logits = np.array([[4, 3.999, 2, 1, 0, -1]], dtype=float)
    reference = summarize_logits(logits, [0])
    assert compare_logprobs(reference, summarize_logits(logits + offset, [0])).passed


def test_mutual_top5_checks_both_directions_and_allows_five_percent() -> None:
    logits = np.array([[5, 4, 3, 2, 1, 0]], dtype=float)
    reference = summarize_logits(logits, [0])
    reversed_winner = summarize_logits(np.array([[5, 4, 3, 2, 1, 6]]), [0])
    tolerance = LogprobTolerance(max_reference_top1_delta=10)
    accepted = compare_logprobs(reference * 20, reference * 19 + reversed_winner, tolerance)
    assert accepted.passed
    assert accepted.top5_failed_positions == [19]
    assert not compare_logprobs(
        reference * 20, reference * 18 + reversed_winner * 2, tolerance
    ).passed
    missing_reference = summarize_logits(np.array([[-1, 5, 4, 3, 2, 1]]), [0])
    assert not compare_logprobs(reference, missing_reference, tolerance).passed


def test_near_ties_pass_but_normalized_reference_top1_error_is_not_averaged() -> None:
    logits = np.array([[5, 4.99, 3, 2, 1, 0]])
    reference = summarize_logits(logits, [0])
    assert compare_logprobs(
        reference, summarize_logits(logits + np.array([[-0.02, 0.02, 0, 0, 0, 0]]), [0])
    ).passed
    changed = summarize_logits(logits * 0.2, [0])
    result = compare_logprobs(reference * 20, reference * 19 + changed)
    assert not result.passed
    assert result.logprob_failed_positions == [19]


@pytest.mark.parametrize(
    ("logits", "ids"),
    [
        (np.zeros((0, 6)), []),
        (np.zeros((1, 4)), [0]),
        (np.array([[0, 1, 2, 3, 4, np.nan]]), [0]),
        (np.zeros((1, 6)), [6]),
        (np.zeros((1, 6)), []),
    ],
)
def test_invalid_observations_fail_explicitly(logits: np.ndarray, ids: list[int]) -> None:
    with pytest.raises(ValueError, match=r"shape|finite|reference_ids"):
        summarize_logits(logits, ids)


def test_missing_positions_and_inconsistent_queried_logprobs_are_errors() -> None:
    reference = summarize_logits(np.array([[5, 4, 3, 2, 1, 0]]), [0])
    with pytest.raises(ValueError, match="position counts"):
        compare_logprobs(reference, [])
    actual = reference[0].model_copy(update={"reference_token_logprob": -5})
    with pytest.raises(ValueError, match="disagrees"):
        compare_logprobs(reference, [actual])
    with pytest.raises(ValidationError, match="distinct"):
        TokenLogprobs(
            top_token_ids=[0] * 5,
            top_logprobs=[-3] * 5,
            reference_token_logprob=-3,
        )
