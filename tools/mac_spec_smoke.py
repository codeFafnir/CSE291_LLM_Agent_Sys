"""Single-stream speculative decoding smoke test on Apple Silicon (MLX).

Feeds one real Hermes agent prompt (system prompt with tool definitions + user
task) to the target model, first without a drafter and then with a draft model
at several draft lengths. Decoding is greedy, so every run should emit the same
tokens: speculative decoding changes speed, not output.

Usage (run with -I: the trace file is downloaded, untrusted input):
    uv run python -I tools/mac_spec_smoke.py data/raw/hermes/data/train.jsonl
"""

import argparse
import json

from mlx_lm import load, stream_generate


def first_task(path: str, category: str, user_prompt: str | None = None) -> list[dict]:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["category"] == category:
                conv = row["conversations"]
                return [
                    {"role": "system", "content": conv[0]["value"]},
                    {"role": "user", "content": user_prompt or conv[1]["value"]},
                ]
    raise ValueError(f"no task in category {category!r}")


def run(model, tokenizer, prompt_ids, max_tokens, draft=None, k=0):
    tokens, n_from_draft, last = [], 0, None
    for resp in stream_generate(model, tokenizer, prompt_ids, max_tokens=max_tokens,
                                draft_model=draft, num_draft_tokens=k):
        last = resp
        if resp.token in tokenizer.eos_token_ids:  # the final "stop" response repeats EOS
            continue
        tokens.append(resp.token)
        n_from_draft += resp.from_draft
    return tokens, n_from_draft, last


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hermes_jsonl")
    p.add_argument("--target", default="mlx-community/Qwen3-4B-4bit")
    p.add_argument("--draft", default="mlx-community/Qwen3-0.6B-4bit")
    p.add_argument("--category", default="Terminal & Coding")
    p.add_argument("--user-prompt", help="replace the task's user message (e.g. to force a long answer)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4, 6])
    args = p.parse_args()

    model, tokenizer = load(args.target)
    draft, _ = load(args.draft)
    messages = first_task(args.hermes_jsonl, args.category, args.user_prompt)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"target={args.target}  draft={args.draft}  prompt={len(prompt_ids):,} tokens  "
          f"max_tokens={args.max_tokens}\n")

    # Warm-up: the first run pays one-time Metal kernel compilation.
    run(model, tokenizer, prompt_ids[:256], 8)
    run(model, tokenizer, prompt_ids[:256], 8, draft, 2)

    print(f"{'draft k':>8}{'out tok':>9}{'prefill tok/s':>15}{'decode tok/s':>14}{'from draft':>12}"
          f"{'tok/verify':>12}{'peak GB':>9}{'same output':>13}")
    baseline = None
    for k in [0, *args.draft_lengths]:
        tokens, n_draft, last = run(model, tokenizer, prompt_ids, args.max_tokens,
                                    draft if k else None, k)
        baseline = baseline or tokens
        verify_steps = len(tokens) - n_draft  # each verify step emits exactly one target token
        print(f"{k or '-':>8}{len(tokens):>9}{last.prompt_tps:>15,.0f}{last.generation_tps:>14.1f}"
              f"{n_draft / len(tokens):>12.0%}{len(tokens) / verify_steps:>12.2f}"
              f"{last.peak_memory:>9.2f}{str(tokens == baseline):>13}")

    print("\nsample output:", tokenizer.decode(baseline)[:400].replace("\n", " | "))


if __name__ == "__main__":
    main()
