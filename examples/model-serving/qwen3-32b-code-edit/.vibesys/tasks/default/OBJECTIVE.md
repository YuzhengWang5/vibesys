Maximize **single-batch end-to-end completion throughput** (median completion tokens per second) for **code-debug** edits driven by `.vibesys/tasks/default/benchmark/benchmark.py`. The target model is `Qwen/Qwen3-32B`. Build an OpenAI-style **predicted-outputs** server, but introduce its optimizations as separate, measured interventions in the mandatory order below.

**Headline metric**: `median_tok_per_sec` from `.vibesys/tasks/default/benchmark/benchmark.py`'s JSON output ("Primary metric: median_tok_per_sec = ..."). Single-batch (concurrency 1), tokens counted by re-tokenizing the concatenated server response. This is the only number `perf_metric` should record.

## Prompt-only experiment: one optimization at a time

The experiment tests whether task instructions alone can make the designer,
implementer and judge maintain one optimization mechanism per hypothesis and
candidate. These sequencing rules are **not enforced by new framework code**.
Follow them explicitly in the existing plan, implementation, review and result
fields; do not add new reply-schema fields or change framework policy.

For this task, the staged requirements below take precedence over serving-skills
recommendations that CUDA Graphs or fused FlashInfer attention must be present
in the first runnable version. Those two capabilities are deliberately deferred
until stages S2 and S1 respectively. This is a temporary experimental exemption from those
implementation recommendations, not from model correctness, evaluation,
resource ownership, or service lifecycle requirements. Do not reject an S0/S1
candidate just because it has no CUDA Graphs, or an S0 candidate just because
it has no FlashInfer. Prediction-driven verification is deferred until S3.

### Mandatory sequence

Complete or explicitly close each stage before proposing the next. A stage may
need several rounds of narrow repairs, but one round must not implement or
measure two stages. The first successful service is a prerequisite baseline,
not a bundle of the three optimizations.

| Stage | Single intervention | Required production path | Still forbidden |
| --- | --- | --- | --- |
| S0: runnable control | Establish the minimum correct, measurable AR service; no performance optimization yet | BF16 target-model greedy autoregression, eager PyTorch execution and PyTorch attention; concurrency 1 | Prediction-based multi-token verification, CUDA Graph capture/replay, FlashInfer, custom fused kernels, compiler/backend tuning, quantization, and unrelated performance changes |
| S1: FlashInfer attention | Replace the current attention implementation with eager FlashInfer batched attention only | Preserve S0 greedy autoregression, BF16 model equations and request behavior; use_cuda_graph=False, no capture/replay | Prediction-based multi-token verification, CUDA Graphs, simultaneous KV-write fusion, norm/MLP fusion, GEMM-provider changes, or any independent optimization |
| S2: CUDA Graphs | Capture and replay the existing AR prefill/decode computation only | Preserve the accepted S1 attention backend, greedy AR algorithm, dtype and arithmetic; enable use_cuda_graph=True for retained FlashInfer attention | Prediction-based multi-token verification, attention-provider replacement, new fusion kernels, new prefill widths, graph-width tuning sweeps, and any independent optimization |
| S3: speculative decoding | Add prediction-driven fixed-K verification only | Preserve the retained attention backend and graph execution mode; K=16 verification, necessary KV rollback and the fixed verifier graph required by that algorithm | Attention-provider replacement, auxiliary-kernel fusion, GEMM-provider changes, graph-width tuning sweeps, lookup-policy changes beyond the specified algorithm, or any independent optimization |
| S4 and later | Choose one additional optimization mechanism per hypothesis and candidate | Start from a named trusted retained parent and keep all other mechanisms fixed | Bundles, parameter sweeps, speculative implementation of future ideas, or independent improvements hidden as prerequisites |

S0-S2 must accept the normal `prediction` request envelope, but intentionally
do not use its content for drafting: generate the answer with the real target
model one token at a time. This is the staged exception to the eventual
predicted-outputs algorithm. It is not permission to echo predictions, replay
completed answers, alter prompts, or bypass the target. Keep this control
implementation recoverable for the S1 attention comparison and retain the
accepted S2 AR control for the S3 speculation comparison.

For S1, library-required layout and planning adaptations belong to the attention
replacement; using FlashInfer to optimize unrelated KV writes or normalizations
does not. Keep all execution eager, including FlashInfer planning and attention.
Do not implement S1/S2/S3 together under a title such as "bootstrap the complete fast engine".
For S2, fixed-address buffers and capture-safe orchestration strictly necessary
for replay are part of that intervention, but replacing attention or fusing
operators is not. If the existing computation cannot be captured correctly,
record the blocker rather than replacing attention or importing S3 into S2.
For S3, replace only the generation mechanism described below. If the incumbent
uses graphs, adding the fixed width [seed] + K=16 verifier graph and its necessary
buffers belongs to speculation; preserve existing AR graph shapes and attention,
do not sweep widths or independently retune graph coverage. If an earlier stage
was closed without adoption, retain and disclose that absence instead.

Keep the target identity, BF16 weights/KV, request semantics, evaluator, workload,
concurrency, and unrelated execution choices fixed throughout S0-S3. Do not
pre-implement future stages as inactive branches, selectors or optional paths
inside an earlier candidate. Installing a dependency does not prove activation,
and merely disabling a bundled feature for measurement does not make its
implementation a separate intervention.

### Per-round decision and evidence

Before editing, the designer must state in its existing hypothesis/task fields:

1. The stage, exact parent revision and most recent trusted control measurement.
2. One causal hypothesis, one production mechanism to change, and which
   mechanisms must remain unchanged.
3. The smallest implementation scope, expected effect, falsifier, and the
   official evaluation needed to accept or reject this intervention.

The implementer must compare the complete candidate diff with that parent and
report the actual changed mechanism, necessary supporting edits, and any
deviation. One mechanism may span several files. Conversely, changes to several
independent mechanisms are not one optimization just because they share a file,
title or launch command. For example, fusing normalization, rotary embedding and
activation are separate additional interventions, not one "auxiliary fusion"
hypothesis. Repair correctness, packaging or startup only as needed for the
current stage; do not add performance changes while repairing it.

The judge must check the stage and the whole diff as well as correctness.
Reject a candidate that enables a future stage or combines independent
optimizations, even if it is fast and passes accuracy. Request a narrower repair
within the current stage instead. A performance profile is evidence gathering,
not permission to implement the next idea in the same round.

Use the configured unchanged official accuracy and benchmark to establish S0
and evaluate each later intervention. Keep each submitted source revision and
its actual framework result recoverable. Local tests, predictions, profiler
shares, and missing-input startup failures are not throughput measurements.
Record the measured parent-to-candidate delta; call small or unreplicated
changes uncertain rather than promising stable speedups. A valid S0 measurement
has no speedup over the missing input service: missing is not zero throughput.

Keep a compact stage ledger in the candidate's own documentation using the
existing result fields as appropriate:

```text
round | stage | parent revision | single intervention | submitted revision
      | official accuracy | official tok/s | delta versus control
      | keep/reject/blocked | next stage or narrow repair
```

Advance only after the current stage has a recorded official outcome, or after
a concrete implementation/validation blocker has been explicitly closed under
the existing retry budget. Do not repeatedly submit an unchanged failing
candidate or revalidate an unchanged diagnostic result as new optimization.
On failure or no defensible gain, preserve the trusted incumbent and report
what was not established. If a stage is closed without adoption, later stages
start from that incumbent, must disclose the absent feature, and must not
quietly reintroduce it together with another intervention. A final version
without working prediction-driven verification is a partial experiment, not
completion of the eventual predicted-outputs objective.

Do not obtain implementations, plans, results or conversations from another
experiment. Use only this task's public inputs and this run's own evidence.
Do not modify this objective, the evaluator, correctness thresholds or
benchmark to make a stage pass.

## Server contract

- Streaming `POST /v1/completions`. Bench sends `prompt_is_preformatted: True` — tokenize the prompt verbatim, no second chat template pass.
- SSE text chunks need not correspond one-to-one to model tokens. Buffer partial UTF-8 characters or use empty fragment records without corrupting otherwise-valid Unicode or inventing text to make records non-empty. Speculatively accepted tokens may share a chunk. The unchanged benchmark re-tokenizes the concatenated response for `output_tokens`; SSE chunk count is reported separately, not used as the token count.
- Accept the predicted-outputs envelope `{"prediction": {"type": "content", "content": "<text>"}}`. From S3 onward, tokenize `prediction.content` once at request-time; it is a draft hint, **not** part of the prompt. In S0-S2, accept the envelope without using the hint.
- Disable thinking mode (`enable_thinking=False`); Qwen3's `<think>...</think>` would blow past `max_tokens`.
- Use eager execution in S0/S1. Introduce FlashInfer batched attention with `use_cuda_graph=False` only in S1; introduce AR prefill/decode CUDA Graphs and `use_cuda_graph=True` only in S2. Introduce fixed-K speculative verification and its required verifier graph only in S3.

## Speculation algorithm: introduce only in S3

Stateless n-gram lookup (prompt-lookup decoding). The drafter is the prediction
itself: no drafter model and no cursor between cycles. Tokenize the prediction
once per request. Each cycle, take the last 2-8 emitted tokens as a needle and
find its rightmost occurrence in the prediction. Propose the next K tokens
after the match; verify in one forward over `[seed] + draft`; accept the longest
matching prefix and emit prefix plus a bonus token, which seeds the next lookup.
Use one verify forward per cycle, not an extra decode pass. On no match,
including the first cycle, emit one AR token and retry. Keep K=16 fixed.

S1 changes only attention to eager FlashInfer. S2 captures the accepted AR
prefill/decode paths without introducing speculation. S3 introduces this fixed-K
algorithm using the retained attention backend and execution mode, adding only
the fixed verifier graph needed when graph execution was adopted. Preserve correct
logit/draft alignment, accepted-prefix KV commitment, rejection rollback, EOS
and generation-budget handling in every stage.

## Expected throughput

The original H100 BF16 expectation for the completed K=16 speculative server is
approximately 200-400 median tok/s versus approximately 30 tok/s vanilla AR,
conditional on a 90-95% per-token match assumption. This experiment uses an
A100: those numbers are neither measured A100 controls nor stage acceptance
thresholds. Measure the actual S0 baseline and each single intervention.
Do not enable future stages early to reach a forecast. The sequence and honest
per-intervention results take priority over an unsubstantiated headline gain.

## Official evaluation service lifecycle

Implement a candidate-owned `run_server.sh` at the project root. It must accept
`--model-dir /model --port PORT`, prepare its dependencies from the committed
project, and run the candidate service in the foreground until stopped.
`GET /health` must return HTTP 200 only after the real model is ready to serve.
Do not daemonize the launcher or depend on a server left by an earlier command.

The framework runs accuracy and benchmark in separate fresh containers at the
submitted candidate revision. Each configured command invokes the read-only
`managed_evaluation.py` that starts `run_server.sh`, waits up to 900 seconds for
readiness, executes the original evaluator, and stops the service and its
descendants. The checker and benchmark correctness rules are unchanged.

Stop any GPU self-test service before submitting a framework evaluation, so the
single allocated GPU does not hold two copies of Qwen3-32B. For local use, call
the configured managed commands too; do not start a second service alongside
them. A missing launcher, startup failure, or service exit is an evaluation
failure with the service log retained, not a valid performance measurement.
