"""Exercise the read-only serving wrapper in fresh CPU-only containers."""

from __future__ import annotations

import os
import textwrap
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tests.slurm_cluster.cluster import docker

if TYPE_CHECKING:
    import subprocess

pytestmark = pytest.mark.minimal_container

_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER = _ROOT / "resources/evaluators/model_serving/managed_evaluation.py"
_IMAGE_ENV = "VIBESYS_SERVICE_TEST_IMAGE"
_SERVER = """\
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
        self.wfile.write(b"ready")
    def log_message(self, *args):
        pass
print("candidate service started", flush=True)
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""
_CLIENT = """\
import argparse, json, pathlib, sys, urllib.request
parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--output-json")
parser.add_argument("--exit-code", type=int, default=0)
args = parser.parse_args()
with urllib.request.urlopen(args.url + "/health", timeout=5) as response:
    assert response.read() == b"ready"
if args.output_json:
    pathlib.Path(args.output_json).write_text(json.dumps({"score": 42}))
print("evaluator completed", flush=True)
sys.exit(args.exit_code)
"""


@pytest.fixture
def serving_input(tmp_path: Path) -> Path:
    (tmp_path / "server.py").write_text(_SERVER)
    (tmp_path / "client.py").write_text(_CLIENT)
    (tmp_path / "run_server.sh").write_text('exec python /workspace/server.py "$@"\n')
    return tmp_path


def _container(workspace: Path, script: str) -> subprocess.CompletedProcess[str]:
    image = os.environ.get(_IMAGE_ENV)
    if image is None:
        pytest.skip(f"set {_IMAGE_ENV} to an existing Python container image")
    return docker(
        "run",
        "--rm",
        "--runtime",
        "runc",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=void",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-v",
        f"{workspace}:/workspace:ro",
        "-v",
        f"{_WRAPPER}:/opt/managed_evaluation.py:ro",
        "-w",
        "/workspace",
        "--entrypoint",
        "python",
        image,
        "-c",
        textwrap.dedent(script),
        check=False,
        timeout=60,
    )


def test_fresh_evaluation_requires_a_managed_service(serving_input: Path) -> None:
    result = _container(
        serving_input,
        """\
        import subprocess, sys
        direct = subprocess.run([sys.executable, "client.py"], capture_output=True)
        assert direct.returncode != 0
        assert b"Connection refused" in direct.stderr
        for stage in ("accuracy", "benchmark"):
            managed = subprocess.run([
                sys.executable, "/opt/managed_evaluation.py", "--",
                sys.executable, "client.py", "--output-json", "/tmp/" + stage + ".json"
            ], capture_output=True)
            assert managed.returncode == 0, managed.stderr.decode()
            assert b"evaluator completed" in managed.stdout
            assert b"candidate service ready" in managed.stderr
            assert b"candidate service started" in managed.stderr
            import json
            with open("/tmp/" + stage + ".json") as result:
                assert json.load(result) == {"score": 42}
        """,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("exit_code", [1, 7])
def test_evaluator_failure_is_preserved_and_service_stops(
    serving_input: Path, exit_code: int
) -> None:
    script = """\
        import socket, subprocess, sys
        managed = subprocess.run([
            sys.executable, "/opt/managed_evaluation.py", "--",
            sys.executable, "client.py", "--exit-code", EXIT_CODE
        ], capture_output=True)
        assert managed.returncode == int(EXIT_CODE), managed.stderr.decode()
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 8000))
        assert b"candidate service started" in managed.stderr
        """
    result = _container(serving_input, script.replace("EXIT_CODE", repr(str(exit_code))))
    assert result.returncode == 0, result.stderr


def test_startup_failure_keeps_service_log_and_never_runs_client(serving_input: Path) -> None:
    (serving_input / "run_server.sh").write_text("echo deliberate-startup-error >&2\nexit 7\n")
    result = _container(
        serving_input,
        """\
        import subprocess, sys
        result = subprocess.run([
            sys.executable, "/opt/managed_evaluation.py", "--",
            sys.executable, "client.py"
        ], capture_output=True)
        assert result.returncode == 1
        assert b"exited before readiness: status=7" in result.stderr
        assert b"deliberate-startup-error" in result.stderr
        assert b"evaluator completed" not in result.stdout
        """,
    )
    assert result.returncode == 0, result.stderr


def test_missing_startup_entrypoint_is_an_explicit_failure(serving_input: Path) -> None:
    (serving_input / "run_server.sh").unlink()
    result = _container(
        serving_input,
        """\
        import subprocess, sys
        result = subprocess.run([
            sys.executable, "/opt/managed_evaluation.py", "--",
            sys.executable, "client.py"
        ], capture_output=True)
        assert result.returncode == 1
        assert b"startup entrypoint is missing" in result.stderr
        assert b"evaluator completed" not in result.stdout
        """,
    )
    assert result.returncode == 0, result.stderr


def test_cancellation_stops_the_service_and_evaluator(serving_input: Path) -> None:
    result = _container(
        serving_input,
        """\
        import socket, subprocess, sys
        client = "import os, signal; os.kill(os.getppid(), signal.SIGTERM); signal.pause()"
        result = subprocess.run([
            sys.executable, "/opt/managed_evaluation.py", "--",
            sys.executable, "-c", client
        ], capture_output=True)
        assert result.returncode == 143, result.stderr.decode()
        assert b"interrupted by signal 15" in result.stderr
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 8000))
        """,
    )
    assert result.returncode == 0, result.stderr


def test_managed_input_wraps_both_original_evaluators() -> None:
    path = _ROOT / "support/scenario-c-repro/managed-input.toml"
    config = tomllib.loads(path.read_text())
    for stage, evaluator in (("accuracy", "checker.py"), ("benchmark", "benchmark.py")):
        command = config[stage]["command"]
        assert command[:3] == [
            "python",
            "${PROJECT_ROOT}/.vibesys/tasks/default/managed_evaluation.py",
            "--",
        ]
        assert any(item.endswith(evaluator) for item in command[3:])
    assert config["benchmark"]["result"]["metric"] == "median_tok_per_sec"
