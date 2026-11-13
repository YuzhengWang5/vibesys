"""Fail fast on an unimplemented candidate before resolving GPU dependencies."""

import os
import sys
from pathlib import Path


def main() -> int:
    """Run the unchanged numerical gate in its committed uv environment."""
    launcher = Path.cwd() / "run_server.sh"
    if not launcher.is_file():
        print(f"[accuracy] candidate startup entrypoint is missing: {launcher}", file=sys.stderr)
        return 1
    checker = Path(__file__).with_name("checker.py")
    os.execvp("uv", ["uv", "run", "python", str(checker), "--managed", *sys.argv[1:]])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
