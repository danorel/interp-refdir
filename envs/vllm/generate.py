"""Bulk generation in the vLLM env. Reads/writes JSONL so it needs nothing from interptemp.

Input rows: {"prompt": "<fully formatted text>"}; output rows: {"prompt", "completion"}.
Usage: uv run --project envs/vllm python envs/vllm/generate.py \
    --model Qwen/Qwen3-8B --input in.jsonl --output out.jsonl --params '{"max_tokens": 512}'
"""

import argparse
import json

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params", default="{}", help="SamplingParams kwargs as JSON")
    ap.add_argument("--llm-kwargs", default="{}", help="LLM(...) kwargs as JSON")
    ap.add_argument("--no-add-bos", action="store_true")
    args = ap.parse_args()

    with open(args.input) as f:
        prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    llm = LLM(model=args.model, **json.loads(args.llm_kwargs))
    # Tokenize ourselves: vLLM would add special tokens to text prompts, and chat templates
    # (Llama/Gemma) already include BOS -> double BOS. Mirrors NnterpModel.encode.
    tok = llm.get_tokenizer()
    bos = None if args.no_add_bos else tok.bos_token
    inputs = [
        TokensPrompt(
            prompt_token_ids=tok.encode(
                p if not bos or p.startswith(bos) else bos + p, add_special_tokens=False
            )
        )
        for p in prompts
    ]
    outs = llm.generate(inputs, SamplingParams(**json.loads(args.params)))
    with open(args.output, "w") as f:
        for p, o in zip(prompts, outs, strict=True):
            f.write(
                json.dumps({"prompt": p, "completion": o.outputs[0].text}, ensure_ascii=False)
                + "\n"
            )


if __name__ == "__main__":
    main()
