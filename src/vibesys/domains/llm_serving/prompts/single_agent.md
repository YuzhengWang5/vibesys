## LLM-serving combined-round invariants

Trace every claim through the actual request-to-model-to-stream path. For
batching, slot/KV/mask/position changes, retain deterministic concurrent mixed-
length correctness including one request finishing while others continue. For
kernel/layout work, name the removed production operator and its frequency,
bytes, or launches; configuration alone is not activation.

{% if workspace_sources %}
Build on the pinned mutable checkout(s): {% for source in workspace_sources %}`{{ source.dest }}/` ({{ source.name }}){% if not loop.last %}, {% endif %}{% endfor %}. Inspect and adapt their relevant paths before replacing components; if replacement is necessary, record what you inspected and why.
{% endif %}

Follow the input-owned streaming contract and verify decoded text, completion
counts, EOS/stop behavior, and usage accounting. UTF-8 characters may span
multiple model tokens; permit empty fragment records or buffered text when the
input allows them, without manufacturing text or token boundaries. Use the task
evaluator's token-counting rule, not a universal SSE-record count. Fabricating
counts or replaying a completed output/token trajectory to a later arrival is
reward hacking; live exact cohorts may share one active model execution.

Inventory hot-loop telemetry synchronization and measure observer effect. A
materially perturbed profile is qualitative and cannot establish an Amdahl
share. Do not preserve Python/FastAPI boundaries without a contract reason.

As your own judge, do not let yourself cheat: reject prerecorded/constant output, evaluator-specific
branches, weakened checks, omitted failures, or any steady-state response that
uses completed replay instead of the request's declared model execution.
