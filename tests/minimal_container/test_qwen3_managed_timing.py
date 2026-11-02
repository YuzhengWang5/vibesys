"""用有界 CPU 子进程验证 wrapper 的清理后计时; 不使用 Docker 或 GPU。"""

import json
import shlex
import sys
from pathlib import Path

import pytest

from vs_sim.api import DISCARD, OsProcesses, ProcessOutcome, ProcessSpec

WRAPPER = (
    Path(__file__).resolve().parents[2]
    / "examples/model-serving/qwen3-32b-code-edit/.vibesys/tasks/default/managed_evaluation.py"
)
MARKER = "__VIBESYS_MANAGED_TIMING__ "
SERVER = """\
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir")
parser.add_argument("--port", type=int)
args = parser.parse_args()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""


def run_wrapper(workspace: Path, args: list[str]) -> ProcessOutcome:
    result = OsProcesses().run(
        ProcessSpec(argv=(sys.executable, str(WRAPPER), *args), cwd=workspace),
        timeout=20,
        output=DISCARD,
    )
    assert isinstance(result, ProcessOutcome), result
    return result


def timing_record(result: ProcessOutcome) -> dict:
    lines = [line for line in result.stderr.decode().splitlines() if line.startswith(MARKER)]
    assert len(lines) == 1, result.stderr
    return json.loads(lines[0][len(MARKER) :])


def test_missing_launcher_emits_failed_preflight_timing(tmp_path: Path) -> None:
    result = run_wrapper(
        tmp_path,
        ["--", sys.executable, "-c", "raise RuntimeError('未执行')"],
    )
    record = timing_record(result)
    assert result.returncode == record["returncode"] == 1
    assert record["phase"] == "preflight"
    assert record["client_subprocess_seconds"] is None
    assert record["startup_seconds"] is None


@pytest.mark.parametrize("client_code", [0, 7])
def test_client_exit_is_preserved_and_total_includes_cleanup(
    tmp_path: Path,
    client_code: int,
    unused_tcp_port: int,
) -> None:
    (tmp_path / "server.py").write_text(SERVER)
    (tmp_path / "run_server.sh").write_text(
        f'exec {shlex.join([sys.executable, str(tmp_path / "server.py")])} "$@"\n'
    )
    (tmp_path / "client.py").write_text(
        f"import sys\nprint('原始客户端输出')\nsys.exit({client_code})\n"
    )
    result = run_wrapper(
        tmp_path,
        [
            "--port",
            str(unused_tcp_port),
            "--startup-timeout-seconds",
            "5",
            "--",
            sys.executable,
            str(tmp_path / "client.py"),
        ],
    )
    record = timing_record(result)
    assert result.returncode == record["returncode"] == record["client_returncode"] == client_code
    assert record["phase"] == ("completed" if client_code == 0 else "client")
    assert result.stdout.decode() == "原始客户端输出\n"
    assert record["checker_seconds"] == record["client_subprocess_seconds"]
    assert record["managed_seconds"] >= (
        record["client_subprocess_seconds"] + record["startup_seconds"]
    )
    assert not list(tmp_path.glob(".vibesys-service-*"))
    assert b"__VIBESYS_SERVICE_LOG_END__" in result.stderr


def test_startup_failure_records_phase_without_inventing_ready_time(
    tmp_path: Path,
    unused_tcp_port: int,
) -> None:
    (tmp_path / "run_server.sh").write_text("exit 9\n")
    result = run_wrapper(
        tmp_path,
        ["--port", str(unused_tcp_port), "--", "unused-client"],
    )
    record = timing_record(result)
    assert result.returncode == record["returncode"] == 1
    assert record["phase"] == "startup"
    assert record["startup_seconds"] is None
    assert record["client_returncode"] is None
    assert b"__VIBESYS_SERVICE_LOG_END__" in result.stderr
