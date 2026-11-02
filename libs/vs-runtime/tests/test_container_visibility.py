"""Every container alias preserves the host's project visibility policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("history_alias", [False, True])
@pytest.mark.parametrize("directory", [False, True])
def test_container_aliases_hide_private_state_and_operator_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    history_alias: bool,
    directory: bool,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GH_TOKEN", "test-token")
    workspace = tmp_path / "workspace"
    history = tmp_path / "history"
    for root in (workspace, history):
        root.mkdir()
        (root / "trusted.txt").write_text("trusted")
        if directory:
            (root / "private").mkdir()
            (root / "private/secret.txt").write_text("private")
        else:
            (root / "private").write_text("private")
    framework = tmp_path / "framework"
    (framework / "support").mkdir(parents=True)
    (framework / "support/operator.py").write_text("OPERATOR_SECRET = True")
    (framework / "libs").mkdir()
    backend = FakeComputeBackend(threads=SimThreads())
    request = RunEnvironmentRequest(
        log_dir=tmp_path / "logs",
        workspace=workspace,
        git_history_root=history if history_alias else None,
        ref_dir=None,
        backend=backend,
        agent_backend="cli",
        cli_provider="copilot",
        run_id="visibility",
        framework_root=framework,
        project_path_policy=ProjectPathPolicy(
            hidden_paths=("private",), read_only_paths=("trusted.txt",)
        ),
        clock=ManualClock(),
        system=idle_environment_system(),
    )
    environment = DockerEnvironment(DockerEnvironmentConfig(build_runner=FakeDockerBuildRunner()))

    environment.build_editor(request)

    resources = {resource.agent_path: resource for resource in backend.creations[-1].resources}
    aliases = ["/workspace"]
    if history_alias:
        aliases.append("/opt/vibesys-history")
    for alias in aliases:
        hidden = resources[f"{alias}/private"]
        assert hidden.access is HostResourceAccess.READ_ONLY
        assert hidden.path.is_dir() is directory
        if directory:
            assert not list(hidden.path.iterdir())
        else:
            assert hidden.path.read_text() == ""
        trusted = resources[f"{alias}/trusted.txt"]
        assert trusted.access is HostResourceAccess.READ_ONLY
        assert trusted.path.read_text() == "trusted"
    assert resources["/opt/vibesys"].path == framework
    operator_tools = resources["/opt/vibesys/support"]
    assert operator_tools.access is HostResourceAccess.READ_ONLY
    assert operator_tools.path.is_dir()
    assert not list(operator_tools.path.iterdir())
