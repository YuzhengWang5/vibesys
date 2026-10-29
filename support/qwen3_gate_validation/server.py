"""Operator-only native HF positive/negative control; never seed candidates."""

import argparse
import json
from collections.abc import Iterator

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--corrupt", action="store_true")
args = parser.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = (
    AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    .to("cuda")
    .eval()
)
eos = model.generation_config.eos_token_id
eos = [eos] if isinstance(eos, int) else eos
app = FastAPI()


@app.get("/health")
def health() -> dict:
    """Report readiness only after weights have loaded."""
    return {"ready": True}


@app.post("/v1/logits")
def logits(body: dict) -> dict:
    """Expose the native cached forward, optionally corrupting decode rows."""
    ids = torch.tensor([body["input_ids"]], device="cuda")
    length = body["prefill_length"]
    with torch.inference_mode():
        result = model(ids[:, :length], use_cache=True)
        cache = result.past_key_values
        rows = [result.logits[0, -1].float().cpu().tolist()]
        for size in body["decode_chunk_sizes"]:
            result = model(ids[:, length : length + size], past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            values = result.logits[0].float()
            if args.corrupt:
                values = values.roll(1, dims=-1)
            rows.extend(values.cpu().tolist())
            length += size
    return {"positions": list(range(body["prefill_length"] - 1, length)), "logits": rows}


@app.post("/v1/completions")
def completions(body: dict) -> StreamingResponse:
    """Stream native greedy decisions from the same model."""

    def events() -> Iterator[str]:
        ids = tokenizer(body["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(
            "cuda"
        )
        generated = []
        with torch.inference_mode():
            cache = None
            for _ in range(body["max_tokens"]):
                result = model(ids, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                token = int(result.logits[0, -1].argmax())
                generated.append(token)
                ids = torch.tensor([[token]], device="cuda")
                if token in eos:
                    break
        text = tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        yield "data: " + json.dumps({"choices": [{"text": text, "token_ids": generated}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


uvicorn.run(app, host="127.0.0.1", port=args.port)
