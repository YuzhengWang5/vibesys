"""All Copilot roles exclude shared-account history on fresh and resumed turns."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agentshim.testing import FakeExecutor, FakeRun

from vs_agent.api import (
    AgentClient,
    AgentExecutionPolicy,
    AgentSessionKey,
    AgentSessionSpec,
    AgentTurnRequest,
    SessionScope,
)
from vs_agent.api.testing import fake_agentshim_launcher
from vs_sim.api.testing import ManualClock, SimThreads

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("role", ["implementer", "judge", "perf-evaluator"])
def test_cross_session_tools_are_excluded_on_every_turn(tmp_path: Path, role: str) -> None:
    run = FakeRun(
        stdout=[
            json.dumps({"type": "session.start", "data": {"sessionId": "isolated"}}),
            json.dumps({"type": "assistant.message", "data": {"content": "done"}}),
        ]
    )
    executor = FakeExecutor(run, binaries={"copilot": "/usr/bin/copilot"})
    threads = SimThreads()
    launcher = fake_agentshim_launcher(provider="copilot", executor=executor, threads=threads)
    client = AgentClient(launcher, provider="copilot", clock=ManualClock(), threads=threads)
    spec = AgentSessionSpec(
        role=role,
        provider="copilot",
        workspace=tmp_path,
        policy=AgentExecutionPolicy(require_enforcement=False),
    )
    key = AgentSessionKey(SessionScope.MEMBER, role)
    try:
        for _ in range(2):
            result = client.run(
                session_spec=spec,
                turn=AgentTurnRequest(message="done"),
                session_key=key,
            )
            assert result.text == "done"
    finally:
        client.close()
    turns = [request for request in executor.requests if "--output-format" in request.argv]
    assert len(turns) == 2
    assert "--resume" not in turns[0].argv
    assert "--resume" in turns[1].argv
    for request in turns:
        for tool in ("session_store_sql", "store_memory", "vote_memory"):
            assert f"--excluded-tools={tool}" in request.argv
            assert f"--deny-tool={tool}" in request.argv
