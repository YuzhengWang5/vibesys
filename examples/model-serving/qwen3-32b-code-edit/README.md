# qwen3-32b-code-edit — predicted-outputs benchmark

This input bundle measures **single-batch tok/s** of an OpenAI-style [predicted-outputs](https://developers.openai.com/api/docs/guides/predicted-outputs) server for `Qwen/Qwen3-32B` on the **code-debug** subset of [`m-a-p/CodeEditorBench`](https://huggingface.co/datasets/m-a-p/CodeEditorBench). The benchmark sends the buggy original code as `prediction.content` on every request. The pinned objective determines which single optimization the candidate may implement.

**Single-batch only.** Concurrency 1. The metric the orchestrator tracks is `median_tok_per_sec` from `.vibesys/tasks/default/benchmark/benchmark.py`.

## Why this dataset

CodeEditorBench's "code debug" rows hand the model an `incorrect_solutions` (the buggy code) and ask it to emit `solutions` (the same code with the bug fixed). On Python3 rows, **median character overlap between the two is 99.6%**, and 75% of matched runs are >16 chars long. That is exactly the regime predicted outputs is designed for: the model's output is dominated by long verbatim copies of a known prediction with a few small islands of new tokens.

`benchmark/benchmark.py` records, per sample, the token-level alignment between the actual model output and the prediction (matched-run lengths, longest run, total matched/diverged tokens). Those numbers are what an analytical headroom calculation would feed on.

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
    │   ├── checker.py                 # teacher-forced top-five/logprob gate
    │   ├── reference_logprobs.json    # pinned independent native HF reference
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

The `prediction` field is the OpenAI predicted-outputs format. Servers that don't implement it (vLLM, SGLang today) just ignore it; a server that does implement it is the configuration this benchmark scores.

## How to run

### VibeSys and managed local evaluation

Implement a foreground `run_server.sh --model-dir /model --port PORT` in the
candidate project. Mount the target weights read-only at `/model`. The service
must return HTTP 200 from `/health` only after the real model is ready.

Both manifest commands use `managed_evaluation.py`: it launches the submitted
candidate, waits up to 900 seconds for readiness, runs the client,
then stops the service and its descendants. Accuracy uses a pinned, independently
computed HF reference without loading a second model. Accuracy and benchmark each use a
fresh container; a service started in an agent's container is not available
in a formal evaluation container. Stop self-test services before submitting
formal evaluation to avoid loading two models onto the allocated GPU.

Startup errors, readiness timeout and service exit fail the evaluation with
the service log retained on stderr. Client stdout and `--output-json` are
preserved. Startup time is outside the request throughput measurement.

For example, in the prepared candidate environment:

```bash
uv run python .vibesys/tasks/default/accuracy_checker/checker.py --managed \
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

The checker compares teacher-forced production top-five and reference-top-one
logprobs with native Hugging Face BF16 inference. Its numerical tolerances,
KV-cache coverage and required observation interface are specified in
`accuracy_checker/README.md`. The accuracy manifest owns service startup,
cleanup. Native HF reference generation is performed once, not per candidate.
The fixed policy requires all 306 observations from six 50-continuation-token
plans to meet mutual top2 and reference-top1 logprob error <=0.02, without
any calibration margin. Independent vLLM diagnostics never change that limit.
The benchmark separately records verbatim echo and improvement-over-input
statistics; retain those statistics alongside throughput.
