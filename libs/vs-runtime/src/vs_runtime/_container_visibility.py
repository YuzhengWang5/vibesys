"""Apply project visibility to every container alias and hide host-only tooling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vs_agent.api import CONTAINER_FRAMEWORK_ROOT
from vs_sandbox.api import (
    HostResource,
    ProjectPathPolicy,
    host_resource_for_mount,
)
from vs_sandbox.api.docker_workspace import docker_project_path_resources

if TYPE_CHECKING:
    from pathlib import Path


def project_visibility_resources(
    policy: ProjectPathPolicy,
    workspace: Path,
    history_root: Path | None,
    *,
    mask_root: Path,
    container_root: str,
) -> tuple[HostResource, ...]:
    """Protect the editable workspace and its read-only history alias equally."""
    aliases = [(workspace, container_root, "workspace")]
    if history_root is not None:
        aliases.append((history_root, "/opt/vibesys-history", "history"))
    return tuple(
        resource
        for host, container, name in aliases
        for resource in docker_project_path_resources(
            policy,
            host,
            mask_root=mask_root / name,
            container_root=container,
        )
    )


def framework_resources(framework_root: Path, *, mask_root: Path) -> tuple[HostResource, ...]:
    """Expose framework code without the checkout's host-only support tools."""
    root = host_resource_for_mount(str(framework_root), CONTAINER_FRAMEWORK_ROOT, read_only=True)
    if not (framework_root / "support").exists():
        return (root,)
    return (
        root,
        *docker_project_path_resources(
            ProjectPathPolicy(hidden_paths=("support",)),
            framework_root,
            mask_root=mask_root,
            container_root=CONTAINER_FRAMEWORK_ROOT,
        ),
    )
