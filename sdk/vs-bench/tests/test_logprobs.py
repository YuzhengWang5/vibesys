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


STRICT = LogprobTolerance(top_k=2, min_mutual_top5_rate=1, max_reference_top1_delta=0.02)


@pytest.mark.parametrize(
    ("observed", "passed"),
    [(-0.52, True), (-0.519999999, True), (-0.520000001, False), (-0.52001, False)],
)
def test_strict_decimal_boundary_has_no_margin(observed: float, *, passed: bool) -> None:
    reference = TokenLogprobs(
        top_token_ids=[0, 1, 2, 3, 4],
        top_logprobs=[-0.5, -2, -3, -4, -5],
        reference_token_logprob=-0.5,
    )
    candidate = reference.model_copy(
        update={
            "top_logprobs": [observed, -2, -3, -4, -5],
            "reference_token_logprob": observed,
        }
    )
    assert compare_logprobs([reference], [candidate], STRICT).passed is passed


@pytest.mark.parametrize(
    ("expected_ids", "actual_ids"),
    [([0, 1, 2, 3, 4], [1, 2, 0, 3, 4]), ([0, 1, 2, 3, 4], [2, 0, 1, 3, 4])],
)
def test_strict_mutual_top2_rejects_either_direction(
    expected_ids: list[int], actual_ids: list[int]
) -> None:
    expected = TokenLogprobs(
        top_token_ids=expected_ids,
        top_logprobs=[-2] * 5,
        reference_token_logprob=-2,
    )
    actual = expected.model_copy(update={"top_token_ids": actual_ids})
    result = compare_logprobs([expected] * 306, [expected] * 305 + [actual], STRICT)
    assert not result.passed
    assert result.top_k == 2
    assert result.top5_failed_positions == [305]


@given(st.integers(min_value=6, max_value=500))
def test_summary_normalizes_entire_vocabulary(vocabulary: int) -> None:
    row = summarize_logits(np.zeros((1, vocabulary)), [vocabulary - 1])[0]
    assert row.reference_token_logprob == pytest.approx(-np.log(vocabulary))
    assert sum(np.exp(row.top_logprobs)) == pytest.approx(5 / vocabulary)


@pytest.mark.parametrize(
    "values",
    [[-2, -3, -2, -4, -5], [-2, -3, -4, -5], [-2, -3, -4, -5, np.inf]],
)
def test_top_observations_reject_unsorted_missing_and_nonfinite(values: list[float]) -> None:
    with pytest.raises(ValidationError):
        TokenLogprobs(
            top_token_ids=[0, 1, 2, 3, 4],
            top_logprobs=values,
            reference_token_logprob=-2,
        )


def test_queried_probability_cannot_hide_a_threshold_exceedance() -> None:
    reference = TokenLogprobs(
        top_token_ids=[0, 1, 2, 3, 4],
        top_logprobs=[-0.5, -2, -3, -4, -5],
        reference_token_logprob=-0.5,
    )
    forged = reference.model_copy(
        update={
            "top_logprobs": [-0.520001, -2, -3, -4, -5],
            "reference_token_logprob": -0.52,
        }
    )
    with pytest.raises(ValueError, match="disagrees"):
        compare_logprobs([reference], [forged], STRICT)
