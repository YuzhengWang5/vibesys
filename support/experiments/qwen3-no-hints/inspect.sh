#!/usr/bin/env bash
set -euo pipefail

python3 - "$@" <<'PY'
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--now", type=int, required=True)
parser.add_argument("--running", choices=("yes", "no"), required=True)
parser.add_argument("--phase", default="monitoring")
args = parser.parse_args()
root = args.run_dir.resolve()
logs = root / "logs"
files = sorted((logs / "vibesys-state").glob("projects/*/runs/*/logs/core-events.jsonl"))
failures = []
last_event = None
run_id = None
resume_file = root / "scripts/resume_run_id"
if resume_file.exists():
    run_id = resume_file.read_text().strip() or None
last_progress = int((logs / "launcher.log").stat().st_mtime) if (logs / "launcher.log").exists() else args.now
for path in files:
    last_progress = max(last_progress, int(path.stat().st_mtime))
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue  # A writer may still own the last record.
        event = json.loads(line)
        run_id = event["run_id"]
        last_event = event["type"]
        if event["type"] == "gate_finished" and event["status"] == "failed":
            data = event["data"]
            if "candidate startup entrypoint is missing" in (data.get("output_tail") or ""):
                continue  # The deliberately empty initial implementation is not a framework failure.
            failures.append({"sequence": event["sequence"], "gate": data["gate"]})
terminal = None
if run_id:
    path = root / "project" / ".vibesys/state/runs" / run_id / "publications/publications.json"
    if path.exists():
        publications = json.loads(path.read_text())["publications"]
        for publication in publications:
            event = publication["event"]
            if event["kind"] == "run_ended":
                terminal = event["result"]
status = {
    "now": args.now,
    "phase": args.phase,
    "running": args.running == "yes",
    "run_id": run_id,
    "last_event": last_event,
    "failure_count": len(failures),
    "recent_failures": failures[-8:],
    "seconds_without_events": max(0, args.now - last_progress),
    "terminal": terminal,
}
destination = logs / "supervisor-health.json"
pending = destination.with_suffix(".tmp")
pending.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
pending.replace(destination)
print(json.dumps(status, ensure_ascii=False))
PY
