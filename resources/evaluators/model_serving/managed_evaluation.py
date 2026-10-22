"""Run an HTTP evaluator with its candidate service in one isolated container.

This stdlib-only entrypoint is copied into the read-only task input. Candidate
code supplies run_server.sh; neither the checker nor the benchmark is modified.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, suppress
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType
    from typing import TextIO

_LOG = logging.getLogger("managed-evaluation")
_POLL_SECONDS = 0.1
_STOP_SECONDS = 5
_MAX_PORT = 65535


class EvaluationError(RuntimeError):
    """The service could not support the requested evaluation."""


class Cancellation:
    """Record signals without interrupting process acquisition or cleanup."""

    def __init__(self) -> None:
        """Start with no cancellation requested."""
        self.signum: int | None = None

    def record(self, signum: int, frame: FrameType | None) -> None:
        """Remember the first signal; cleanup remains one-shot."""
        del frame
        if self.signum is None:
            self.signum = signum

    def check(self) -> None:
        """Stop at a point where every acquired process has a cleanup owner."""
        if self.signum is not None:
            message = f"evaluation interrupted by signal {self.signum}"
            raise EvaluationError(message)


def _status(process: subprocess.Popen[bytes]) -> int | None:
    # Keep the leader unreaped until its entire group has been stopped, so its
    # process-group ID cannot be recycled while cleanup still owns it.
    info = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def _stop(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + _STOP_SECONDS
    while _status(process) is None and time.monotonic() < deadline:
        time.sleep(_POLL_SECONDS)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=_STOP_SECONDS)


def _write_service_log(stream: TextIO) -> None:
    stream.flush()
    stream.seek(0)
    sys.stderr.write("\n__VIBESYS_SERVICE_LOG_BEGIN__\n")
    shutil.copyfileobj(stream, sys.stderr)
    sys.stderr.write("\n__VIBESYS_SERVICE_LOG_END__\n")
    sys.stderr.flush()


def _wait_ready(
    service: subprocess.Popen[bytes],
    url: str,
    timeout: float,
    cancellation: Cancellation,
) -> None:
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = "no health response"
    while True:
        cancellation.check()
        status = _status(service)
        if status is not None:
            message = f"candidate service exited before readiness: status={status}"
            raise EvaluationError(message)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            message = f"candidate service readiness timed out: {url}; {last_error}"
            raise EvaluationError(message)
        try:
            with opener.open(url, timeout=min(2.0, remaining)) as response:
                if response.status == HTTPStatus.OK:
                    _LOG.info("candidate service ready: %s", url)
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))


def _evaluate(
    command: list[str],
    service: subprocess.Popen[bytes],
    cancellation: Cancellation,
    cleanup: ExitStack,
) -> int:
    # lint-waiver: LW-031900 [S603]; The trusted task supplies an argv vector. A shell
    # > wrapper would weaken quoting; this script runs only in candidate isolation.
    evaluator = subprocess.Popen(  # noqa: S603
        command, stdin=subprocess.DEVNULL, start_new_session=True
    )
    cleanup.callback(_stop, evaluator)
    while True:
        cancellation.check()
        status = _status(evaluator)
        if status is not None:
            _LOG.info("evaluator exited: status=%s", status)
            return status if status >= 0 else 128 - status
        service_status = _status(service)
        if service_status is not None:
            message = f"candidate service exited during evaluation: status={service_status}"
            raise EvaluationError(message)
        time.sleep(_POLL_SECONDS)


def run_evaluation(
    command: list[str],
    *,
    port: int,
    startup_timeout: float,
    cancellation: Cancellation,
) -> int:
    """Own the service, readiness, evaluator, logs, and cleanup for one stage."""
    launcher = Path.cwd() / "run_server.sh"
    if not launcher.is_file():
        message = f"candidate startup entrypoint is missing: {launcher}"
        raise EvaluationError(message)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    url = f"http://127.0.0.1:{port}"
    with (
        tempfile.TemporaryDirectory(prefix="vibesys-service-") as temporary,
        (Path(temporary) / "service.log").open("w+", encoding="utf-8", errors="replace") as log,
        ExitStack() as cleanup,
    ):
        cleanup.callback(_write_service_log, log)
        service_command = ["bash", str(launcher), "--model-dir", "/model", "--port", str(port)]
        _LOG.info("starting candidate service: %s", service_command)
        # lint-waiver: LW-031901 [S603]; Executing the candidate's fixed startup entry
        # > is intentional. Shell-free argv preserves quoting and container confinement.
        service = subprocess.Popen(  # noqa: S603
            service_command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        cleanup.callback(_stop, service)
        _wait_ready(service, f"{url}/health", startup_timeout, cancellation)
        return _evaluate([*command, "--url", url], service, cancellation, cleanup)


def main(argv: list[str] | None = None) -> int:
    """Forward evaluator arguments unchanged, except for its managed local URL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an evaluator command is required after --")
    if not 1 <= args.port <= _MAX_PORT:
        parser.error("--port must be between 1 and 65535")
    if not 0 < args.startup_timeout_seconds < float("inf"):
        parser.error("--startup-timeout-seconds must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="[managed-evaluation] %(message)s")
    cancellation = Cancellation()
    with ExitStack() as signals:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(number, cancellation.record)
            signals.callback(signal.signal, number, previous)
        try:
            return run_evaluation(
                command,
                port=args.port,
                startup_timeout=args.startup_timeout_seconds,
                cancellation=cancellation,
            )
        except (EvaluationError, OSError, subprocess.TimeoutExpired) as error:
            _LOG.error("%s", error)
            return 128 + cancellation.signum if cancellation.signum is not None else 1


if __name__ == "__main__":
    sys.exit(main())
