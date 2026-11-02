"""Small teacher-forced inference checks, following vLLM's mutual top-k rule."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, model_validator

__all__ = [
    "LogprobComparison",
    "LogprobTolerance",
    "TokenLogprobs",
    "compare_logprobs",
    "summarize_logits",
]

TOP_K = 5
_MATRIX_DIMENSIONS = 2


class TokenLogprobs(BaseModel):
    """Sorted top-five IDs/probabilities and the requested reference token's logprob."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    top_token_ids: list[StrictInt] = Field(min_length=TOP_K, max_length=TOP_K)
    top_logprobs: list[FiniteFloat] = Field(min_length=TOP_K, max_length=TOP_K)
    reference_token_logprob: FiniteFloat

    @model_validator(mode="after")
    def valid_distribution(self) -> TokenLogprobs:
        """Require an ordered, finite subset of one normalized vocabulary distribution."""
        if len(set(self.top_token_ids)) != TOP_K or min(self.top_token_ids) < 0:
            message = "top_token_ids must be distinct nonnegative vocabulary IDs"
            raise ValueError(message)
        if self.top_logprobs != sorted(self.top_logprobs, reverse=True):
            message = "top_logprobs must be sorted descending"
            raise ValueError(message)
        if max(self.top_logprobs) > 0 or self.reference_token_logprob > 0:
            message = "logprobs must be nonpositive"
            raise ValueError(message)
        if sum(math.exp(value) for value in self.top_logprobs) > 1 + 1e-5:
            message = "top-five probability mass exceeds one"
            raise ValueError(message)
        return self


class LogprobTolerance(BaseModel):
    """Fixed mutual top-k policy over five reported full-vocabulary observations.

    The historical ``top5`` rate field names remain compatible; ``top_k`` selects
    the subset actually compared. No numerical margin is added to the limit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_mutual_top5_rate: float = Field(default=0.95, gt=0, le=1)
    max_reference_top1_delta: FiniteFloat = Field(default=0.2, ge=0)
    top_k: int = Field(default=TOP_K, ge=1, le=TOP_K, strict=True)


class LogprobComparison(BaseModel):
    """Historical top5-named fields describe the selected ``top_k`` policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    top_k: int = Field(default=TOP_K, ge=1, le=TOP_K, strict=True)
    positions: int
    mutual_top5_rate: float
    max_reference_top1_delta: float
    top5_failed_positions: list[int]
    logprob_failed_positions: list[int]


_DEFAULT_TOLERANCE = LogprobTolerance()


def summarize_logits(logits: np.ndarray, reference_ids: list[int]) -> list[TokenLogprobs]:
    """Normalize full vocabulary logits in FP64 and retain only required observations."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != _MATRIX_DIMENSIONS or not len(values) or values.shape[1] < TOP_K:
        message = "logits must have shape [nonempty positions, vocabulary >= 5]"
        raise ValueError(message)
    if not np.isfinite(values).all():
        message = "logits must be finite"
        raise ValueError(message)
    if len(reference_ids) != len(values) or any(
        token < 0 or token >= values.shape[1] for token in reference_ids
    ):
        message = "reference_ids must contain one in-vocabulary token per position"
        raise ValueError(message)
    shifted = values - values.max(axis=1, keepdims=True)
    logprobs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    top_ids = np.argsort(-logprobs, axis=1, kind="stable")[:, :TOP_K]
    return [
        TokenLogprobs(
            top_token_ids=ids.tolist(),
            top_logprobs=row[ids].tolist(),
            reference_token_logprob=float(row[reference_id]),
        )
        for ids, row, reference_id in zip(top_ids, logprobs, reference_ids, strict=True)
    ]


def compare_logprobs(
    reference: list[TokenLogprobs],
    candidate: list[TokenLogprobs],
    tolerance: LogprobTolerance = _DEFAULT_TOLERANCE,
) -> LogprobComparison:
    """Check all aligned teacher-forced positions; reject incomplete or inconsistent data."""
    if not reference or len(reference) != len(candidate):
        message = "reference/candidate must have equal nonzero position counts"
        raise ValueError(message)
    top5_failed = []
    deltas = []
    logprob_failed = []
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        if expected.reference_token_logprob != expected.top_logprobs[0]:
            message = "reference_token_logprob must be the reference top-one logprob"
            raise ValueError(message)
        reference_id = expected.top_token_ids[0]
        if reference_id in actual.top_token_ids:
            reported = actual.top_logprobs[actual.top_token_ids.index(reference_id)]
            if reported != actual.reference_token_logprob:
                message = "candidate reference_token_logprob disagrees with its top-five"
                raise ValueError(message)
        if (
            reference_id not in actual.top_token_ids[: tolerance.top_k]
            or actual.top_token_ids[0] not in expected.top_token_ids[: tolerance.top_k]
        ):
            top5_failed.append(index)
        # Compare the decimal values carried by JSON, not a widened binary-float
        # limit. In particular -0.50 versus -0.52 is exactly the 0.02 boundary.
        delta = abs(
            Decimal(str(expected.reference_token_logprob))
            - Decimal(str(actual.reference_token_logprob))
        )
        deltas.append(float(delta))
        if delta > Decimal(str(tolerance.max_reference_top1_delta)):
            logprob_failed.append(index)
    rate = 1 - len(top5_failed) / len(reference)
    return LogprobComparison(
        passed=rate >= tolerance.min_mutual_top5_rate and not logprob_failed,
        top_k=tolerance.top_k,
        positions=len(reference),
        mutual_top5_rate=rate,
        max_reference_top1_delta=max(deltas),
        top5_failed_positions=top5_failed,
        logprob_failed_positions=logprob_failed,
    )
