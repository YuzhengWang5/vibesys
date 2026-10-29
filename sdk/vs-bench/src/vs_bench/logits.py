"""Full-vocabulary, row-wise numerical qualification for inference engines.

The public contracts reject malformed/nonfinite data. Comparisons tolerate
BF16 roundoff and additive logit offsets, but never average away a bad row.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LogitComparison", "LogitTolerance", "compare_logits"]

_MATRIX_DIMENSIONS = 2
_MIN_VOCABULARY = 2


class LogitTolerance(BaseModel):
    """Fixed qualification policy, applied independently to every token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_cosine: float = Field(default=0.999, ge=0, le=1)
    max_normalized_rmse: float = Field(default=0.045, gt=0)
    max_probability_tv: float = Field(default=0.035, gt=0, le=1)
    max_winner_regret: float = Field(default=0.25, ge=0)


class LogitComparison(BaseModel):
    """Worst-row diagnostics and the conjunction of all numerical gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    rows: int
    min_cosine: float
    max_normalized_rmse: float
    max_probability_tv: float
    max_winner_regret: float
    failed_rows: list[int]


_DEFAULT_TOLERANCE = LogitTolerance()


def compare_logits(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance: LogitTolerance = _DEFAULT_TOLERANCE,
) -> LogitComparison:
    """Compare [positions, vocabulary] arrays; raise ValueError on bad data."""
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if (
        expected.ndim != _MATRIX_DIMENSIONS
        or expected.shape[0] == 0
        or expected.shape[1] < _MIN_VOCABULARY
    ):
        message = "reference logits must have shape [nonempty positions, vocabulary >= 2]"
        raise ValueError(message)
    if actual.shape != expected.shape:
        message = f"candidate logits shape {actual.shape} != reference {expected.shape}"
        raise ValueError(message)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        message = "logits must contain only finite numbers"
        raise ValueError(message)
    centered_reference = expected - expected.mean(axis=1, keepdims=True)
    centered_candidate = actual - actual.mean(axis=1, keepdims=True)
    reference_norm = np.linalg.norm(centered_reference, axis=1)
    candidate_norm = np.linalg.norm(centered_candidate, axis=1)
    if np.any(reference_norm == 0) or np.any(candidate_norm == 0):
        message = "constant logits cannot qualify an inference engine"
        raise ValueError(message)
    cosine = np.sum(centered_reference * centered_candidate, axis=1) / (
        reference_norm * candidate_norm
    )
    normalized_rmse = np.linalg.norm(centered_reference - centered_candidate, axis=1) / (
        reference_norm
    )
    reference_probability = np.exp(expected - expected.max(axis=1, keepdims=True))
    reference_probability /= reference_probability.sum(axis=1, keepdims=True)
    candidate_probability = np.exp(actual - actual.max(axis=1, keepdims=True))
    candidate_probability /= candidate_probability.sum(axis=1, keepdims=True)
    tv = np.abs(reference_probability - candidate_probability).sum(axis=1) / 2
    winners = actual.argmax(axis=1)
    regret = expected.max(axis=1) - expected[np.arange(len(expected)), winners]
    failed = (
        (cosine < tolerance.min_cosine)
        | (normalized_rmse > tolerance.max_normalized_rmse)
        | (tv > tolerance.max_probability_tv)
        | (regret > tolerance.max_winner_regret)
    )
    return LogitComparison(
        passed=not bool(failed.any()),
        rows=len(expected),
        min_cosine=float(cosine.min()),
        max_normalized_rmse=float(normalized_rmse.max()),
        max_probability_tv=float(tv.max()),
        max_winner_regret=float(regret.max()),
        failed_rows=np.flatnonzero(failed).tolist(),
    )
