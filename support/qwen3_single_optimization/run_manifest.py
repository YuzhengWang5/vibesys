"""Run the original immutable accuracy or benchmark command for fixed-code replication."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

__all__ = ["main", "manifest_command"]


def manifest_command(
    manifest: dict, project: Path, stage: str, output: Path
) -> tuple[list[str], int]:
    """Resolve only the original PROJECT_ROOT placeholder; preserve evaluator arguments."""
    section = manifest[stage]
    command = [argument.replace("${PROJECT_ROOT}", str(project)) for argument in section["command"]]
    if stage == "benchmark":
        command.extend([section["result"]["json_argument"], str(output)])
    return command, section["timeout_seconds"]


def main() -> None:
    """Record exact argv and propagate timeout/nonzero exits without a success fallback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage", choices=("accuracy", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    manifest = tomllib.loads((project / ".vibesys/tasks/default/vibesys.input.toml").read_text())
    command, timeout = manifest_command(manifest, project, args.stage, args.output)
    sys.stderr.write(json.dumps({"stage": args.stage, "argv": command, "timeout": timeout}) + "\n")
    # Pinned manifest argv is trusted; shell execution or a duplicate command whitelist is weaker.
    result = subprocess.run(command, cwd=project, timeout=timeout, check=False)  # noqa: S603
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
