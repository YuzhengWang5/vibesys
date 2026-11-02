"""Verify container visibility with the real production interpreter and UID."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from tests.slurm_cluster.cluster import docker

from vs_agent.api import CONTAINER_PYTHON, container_import_roots
from vs_agent.api.testing import FakeDockerBuildRunner
from vs_runtime.api.infrastructure import (
    DockerEnvironment,
    DockerEnvironmentConfig,
    RunEnvironmentRequest,
)
from vs_runtime.api.testing import idle_environment_system
from vs_sandbox.api import HostResourceAccess, ProjectPathPolicy
from vs_sandbox.api.testing import FakeComputeBackend
from vs_sim.api.testing import ManualClock, SimThreads

pytestmark = pytest.mark.minimal_container


def test_real_agent_uid_cannot_read_operator_tools_or_history_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = os.environ.get("VIBESYS_VISIBILITY_TEST_IMAGE")
    if image is None:
        pytest.skip("set VIBESYS_VISIBILITY_TEST_IMAGE to an existing agent image")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GH_TOKEN", "test-token")
    workspace = tmp_path / "workspace"
    history = tmp_path / "history"
    for root in (workspace, history):
        for name in ("local", "runs"):
            directory = root / ".vibesys/state" / name
            directory.mkdir(parents=True)
            (directory / "secret.txt").write_text("private")
        (root / "candidate.py").write_text("VALUE = 1")
    backend = FakeComputeBackend(threads=SimThreads())
    request = RunEnvironmentRequest(
        log_dir=tmp_path / "logs",
        workspace=workspace,
        git_history_root=history,
        ref_dir=None,
        backend=backend,
        agent_backend="cli",
        cli_provider="copilot",
        run_id=f"visibility-test-{uuid4()}",
        framework_root=Path(__file__).resolve().parents[2],
        project_path_policy=ProjectPathPolicy(
            read_only_paths=(".vibesys",),
            hidden_paths=(".vibesys/state/local", ".vibesys/state/runs"),
        ),
        clock=ManualClock(),
        system=idle_environment_system(),
    )
    DockerEnvironment(DockerEnvironmentConfig(build_runner=FakeDockerBuildRunner())).build_editor(
        request
    )
    mounts = ["-v", f"{workspace}:/workspace"]
    for resource in backend.creations[-1].resources:
        target = resource.agent_path or str(resource.path)
        suffix = ":ro" if resource.access is HostResourceAccess.READ_ONLY else ""
        mounts.extend(("-v", f"{resource.path}:{target}{suffix}"))
    script = """
import os
from pathlib import Path
assert os.getuid() != 0
assert not Path('/opt/vibesys/support/qwen3_gate_validation/server.py').exists()
assert not Path('/opt/vibesys/support/qwen3_single_optimization/controller.py').exists()
for alias in ('/workspace', '/opt/vibesys-history'):
    assert (Path(alias) / 'candidate.py').read_text() == 'VALUE = 1'
    for name in ('local', 'runs'):
        assert not (Path(alias) / '.vibesys/state' / name / 'secret.txt').exists()
        assert not list((Path(alias) / '.vibesys/state' / name).iterdir())
import vs_evaluation.agent_core_mcp
import vs_evaluation.queue_status_mcp
print('visibility and MCP imports passed', flush=True)
"""
    result = docker(
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
        "--label",
        f"vibesys.run-id={request.run_id}",
        "-e",
        f"PYTHONPATH={os.pathsep.join(container_import_roots())}",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=void",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        *mounts,
        "--entrypoint",
        CONTAINER_PYTHON,
        image,
        "-c",
        script,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "visibility and MCP imports passed" in result.stdout
