## Modality: text generation (causal LM)

{% if interface == "inprocess" %}
**Accuracy-checker interface** (always required): the input-declared entry
module must expose an importable `VibeServeModel` compatibility class with
`from_pretrained(model_dir, device, dtype)` and
`generate(input_ids, max_new_tokens=N)`. The production server may use a
different language or runtime behind that adapter.

{% endif %}
**Decode invariants** (verify on whichever endpoint the orchestrator scoped in): EOS must not appear in emitted text; stop-string truncation must run before emission; `completion_tokens` must count only emitted text, not raw sampled tokens.

**API contract**: the specific endpoints and request/response shapes to verify are whatever the orchestrator's `pass_criteria` for this round specifies. Do NOT flag "missing" endpoints that the orchestrator did not scope in. If a round only scopes `/v1/completions`, do not fail it for lacking `/v1/chat/completions` or `/predict`. When you need contract details for a scoped endpoint, consult `serving-systems/references/tooling/openai-api.md`.

The input-declared accuracy contract is always required, even when a plan
omits it. If the checker declares a logits observation endpoint, trace it to
the same production weights, forward, attention and KV path as generation,
with the claimed optimization active. A separate reference-model path or
fabricated observations cannot qualify the production engine. Review the
single-mechanism scope against the actual source diff and require an
unoptimized control and like-for-like production A/B evidence.
