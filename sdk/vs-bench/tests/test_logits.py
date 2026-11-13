"""Numerical gates distinguish rounding from incorrect inference."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vs_bench.logits import compare_logits


@given(st.floats(min_value=-100, max_value=100, allow_nan=False))
def test_additive_offsets_preserve_the_distribution(offset: float) -> None:
    reference = np.array([[0.5, 1.0, -2.0, 3.0], [-4.0, 2.0, 1.0, 0.1]])
    assert compare_logits(reference, reference + offset).passed


def test_bfloat16_scale_roundoff_and_near_ties_are_accepted() -> None:
    reference = np.array([[3.0, 2.999, -1.0, 0.0]])
    actual = reference + np.array([[-0.002, 0.002, 0.001, -0.001]])
    assert reference.argmax() != actual.argmax()
    assert compare_logits(reference, actual).passed


@pytest.mark.parametrize("corruption", ["permutation", "stale-kv", "temperature", "winner"])
def test_wrong_engine_results_cannot_hide_in_an_average(corruption: str) -> None:
    reference = np.tile(np.array([[0.5, 1.0, -2.0, 3.0]]), (100, 1))
    actual = reference.copy()
    match corruption:
        case "permutation":
            actual[-1] = actual[-1, ::-1]
        case "stale-kv":
            actual[-1] = reference[0] + np.array([0, 1, 0, -1])
        case "temperature":
            actual[-1] *= 0.5
        case "winner":
            actual[-1, 0] = 4.0
    result = compare_logits(reference, actual)
    assert not result.passed
    assert result.failed_rows == [99]


@pytest.mark.parametrize(
    "actual",
    [
        np.zeros((2, 4)),
        np.ones((1, 4)),
        np.array([[0, 1, np.nan, 2]]),
        np.array([[0, 1, np.inf, 2]]),
        np.zeros((1, 3)),
    ],
)
def test_malformed_nonfinite_and_constant_responses_fail_closed(actual: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"shape|finite|constant"):
        compare_logits(np.array([[0, 1, 2, 3]]), actual)


@pytest.mark.parametrize("reference", [np.zeros((0, 4)), np.ones((1, 1)), np.ones(4)])
def test_invalid_reference_shapes_are_not_qualification_evidence(reference: np.ndarray) -> None:
    with pytest.raises(ValueError, match="reference logits must have shape"):
        compare_logits(reference, reference)
