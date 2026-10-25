"""Pure sanitization for free-form text crossing operator-facing boundaries."""

from __future__ import annotations

import re

from vs_redaction._patterns import SECRET_NAME
from vs_redaction._stream import DiagnosticRedactor

_SECRET_ASSIGNMENT = re.compile(rf"(\b{SECRET_NAME}\b\s*[:=]\s*)([^\s,;]+)")
_BEARER_TOKEN = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")


def redact_diagnostic_text(text: str) -> str:
    """Redact common credential assignments before they cross a UI boundary."""
    text = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    return _BEARER_TOKEN.sub(r"\1[REDACTED]", text)


def redact_diagnostic_tail(text: str, *, max_chars: int) -> str:
    """Return a bounded diagnostic tail without truncating before redaction.

    Redacting the complete input first is part of the contract: slicing an
    untrusted diagnostic can remove the beginning of a credential key and
    make the remaining assignment impossible for a later boundary to detect.
    """
    if max_chars < 0:
        message = "max_chars must be non-negative"
        raise ValueError(message)
    if max_chars == 0:
        return ""
    return redact_diagnostic_text(text)[-max_chars:]


def redact_diagnostic_excerpt(text: str, *, max_chars: int) -> str:
    """Redact before bounding, keeping both the initial error and final context."""
    if max_chars < 0:
        message = "max_chars must be non-negative"
        raise ValueError(message)
    redacted = redact_diagnostic_text(text)
    if len(redacted) <= max_chars:
        return redacted
    separator = "\n[...]\n"
    available = max_chars - len(separator)
    if available <= 1:
        return redacted[:max_chars]
    tail = available // 2
    return redacted[: available - tail] + separator + redacted[-tail:]


__all__ = [
    "DiagnosticRedactor",
    "redact_diagnostic_excerpt",
    "redact_diagnostic_tail",
    "redact_diagnostic_text",
]
