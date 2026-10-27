# qwen3-32b-code-edit — predicted-outputs benchmark

This input bundle measures **single-batch tok/s** of an OpenAI-style predicted-outputs server for `Qwen/Qwen3-32B` on the **code-debug** subset of [`m-a-p/CodeEditorBench`](https://huggingface.co/datasets/m-a-p/CodeEditorBench). Every request includes the buggy original code in `prediction.content`.

**Single-batch only.** Concurrency 1. The metric the orchestrator tracks is `median_tok_per_sec` from `.vibesys/tasks/default/benchmark/benchmark.py`.

## Dataset

CodeEditorBench's "code debug" rows provide `incorrect_solutions` (the buggy code) and `solutions` (the corrected code).

`benchmark/benchmark.py` records per-sample timing, output quality and token-level alignment statistics.

## Layout

```
qwen3-32b-code-edit/
├── README.md                          # this file
├── pyproject.toml                     # uv environment for the checker and benchmark
└── .vibesys/tasks/default/
    ├── OBJECTIVE.md                   # the goal handed to the VibeSys agent loop
    ├── vibesys.input.toml             # manifest: checker and benchmark commands
    ├── managed_evaluation.py          # owns service startup, readiness and cleanup
    ├── benchmark/
    │   ├── benchmark.py               # /v1/completions driver, single-batch
    │   └── README.md
    ├── accuracy_checker/
    │   ├── checker.py                 # quality gate: prevents echo-input bypass
    │   └── README.md
    └── reference/
        ├── README.md                  # how to mount the target model
        └── meta.json                  # pinned model ids
```

## Request envelope

The benchmark drives an OpenAI-compatible `/v1/completions` endpoint. Every request body is:

```json
{
  "prompt": "<chat-templated instruction containing the buggy code>",
  "prediction": {"type": "content", "content": "<the buggy original code, verbatim>"},
  "max_tokens": 512,
  "temperature": 0,
  "stream": true,
  "prompt_is_preformatted": true
}
```

The `prediction` field follows the OpenAI predicted-outputs request format. The server contract is defined in `OBJECTIVE.md`.

## How to run

### VibeSys and managed local evaluation

Implement a foreground `run_server.sh --model-dir /model --port PORT` in the
candidate project. Mount the target weights read-only at `/model`. The service
must return HTTP 200 from `/health` only after the real model is ready.

Both manifest commands run `managed_evaluation.py`: it launches the submitted
candidate, waits up to 900 seconds for readiness, runs the original client,
then stops the service and its descendants. Accuracy and benchmark each use a
fresh container; a service started in an agent's container is not available
in a formal evaluation container. Stop self-test services before submitting
formal evaluation to avoid loading two models onto the allocated GPU.

Startup errors, readiness timeout and service exit fail the evaluation with
the service log retained on stderr. Client stdout and `--output-json` are
preserved. Startup time is outside the request throughput measurement.

For example, in the prepared candidate environment:

```bash
python .vibesys/tasks/default/managed_evaluation.py -- \
    uv run python .vibesys/tasks/default/accuracy_checker/checker.py \
    --tokenizer-path /model --seed 42
```

### External servers and baseline measurements

To measure an already running external server instead, call the client
directly. This does not use the managed manifest commands:

```bash
cd examples/model-serving/qwen3-32b-code-edit
uv run python .vibesys/tasks/default/benchmark/benchmark.py \
    --url http://localhost:8000 --model qwen3-32b \
    --num-samples 100 \
    --output-json /tmp/code_edit_baseline.json
```

The bench prints the headline `Primary metric: median_tok_per_sec = ...` line and writes per-sample alignment + quality stats to the output JSON.

## Accuracy gate

The unchanged checker requires no request errors and at least 50% of steady
outputs to reach gold-solution similarity 0.50. This is a loose similarity
gate, not full semantic correctness or a complete anti-echo check.
The benchmark separately records verbatim echo and improvement-over-input
statistics; retain those statistics alongside throughput.
