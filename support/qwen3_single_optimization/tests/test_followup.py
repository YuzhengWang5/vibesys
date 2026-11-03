"""Follow-up objective variants preserve treatment boundaries and independent budgets."""

from pathlib import Path

import pytest
from qwen3_single_optimization.core import Optimization, Slot
from qwen3_single_optimization.followup import followup_slots
from qwen3_single_optimization.run_manifest import manifest_command

from vs_prompts.api import TemplateRenderer

TEMPLATES = Path(__file__).parents[1] / "templates"


def test_fixed_replication_preserves_manifest_workload_and_timeout() -> None:
    manifest = {
        "benchmark": {
            "command": ["python", "${PROJECT_ROOT}/managed.py", "--num-samples", "50"],
            "timeout_seconds": 4500,
            "result": {"json_argument": "--output-json"},
        }
    }
    command, timeout = manifest_command(
        manifest, Path("/candidate"), "benchmark", Path("/run/a.json")
    )
    assert command == [
        "python",
        "/candidate/managed.py",
        "--num-samples",
        "50",
        "--output-json",
        "/run/a.json",
    ]
    assert timeout == 4500


def test_followup_plan_has_two_full_decode_and_three_per_guided_direction() -> None:
    plan = followup_slots()
    assert len(plan) == 8
    assert len({slot for slot, _ in plan}) == len(plan)
    assert sum(variant == "full-decode" for _, variant in plan) == 2
    for direction in (Optimization.CUDA_GRAPH, Optimization.SPECULATIVE_DECODING):
        assert (
            sum(slot.optimization == direction and variant == "guided" for slot, variant in plan)
            == 3
        )


@pytest.mark.parametrize(("slot", "variant"), followup_slots())
def test_followup_objective_enforces_exact_route_without_history(slot: Slot, variant: str) -> None:
    rendered = TemplateRenderer(TEMPLATES).render_template(
        "OBJECTIVE.md.j2", optimization=slot.optimization.value, objective_variant=variant
    )
    assert "HF BF16" in rendered
    assert "/v1/logprobs" in rendered
    assert "/mnt_ssd/" not in rendered
    assert "20261031" not in rendered
    assert "来源说明" not in rendered
    assert all(other.value not in rendered for other in Optimization if other != slot.optimization)
    if variant == "full-decode":
        assert "一次完整模型 graph replay" in rendered
        assert "lm_head logits" in rendered
        assert "每个 decoder 层有一个独立 graph" not in rendered
    elif slot.optimization is Optimization.CUDA_GRAPH:
        assert "捕获止于下一层 RoPE 之前" in rendered
        assert "一次完整模型 graph replay" not in rendered
    else:
        assert "连续两次" in rendered
        assert "重新匹配成功后" in rendered
        assert "不转为 sticky cursor" in rendered
