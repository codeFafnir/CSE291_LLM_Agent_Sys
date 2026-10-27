"""Summarize candidate agent-trace datasets for replay: step structure, context
growth, prefix sharing, and (where timestamps exist) tool-gap timing.

Hermes token counts use the Qwen3 tokenizer over message bodies only (chat
template wrappers add a few tokens per message), so treat them as lower bounds.
trace-commons context sizes come from the recorded Anthropic API usage fields.

Usage (run with -I: the data files are downloaded, untrusted input):
    uv run python -I tools/inspect_traces.py hermes data/raw/hermes/data/train.jsonl
    uv run python -I tools/inspect_traces.py trace-commons data/raw/trace-commons/data/train-00000-of-00001.parquet
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

TOKENIZER = "Qwen/Qwen3-4B"


def pct(xs, ps=(50, 90, 99)):
    if not xs:
        return "n/a"
    a = np.asarray(xs, dtype=float)
    parts = [f"p{p}={np.percentile(a, p):,.1f}" for p in ps]
    return f"n={len(a):,}  mean={a.mean():,.1f}  " + "  ".join(parts) + f"  max={a.max():,.1f}"


def share_at_most(xs, limits=(4096, 16384, 32768)):
    a = np.asarray(xs)
    return "  ".join(f"<={lim:,}: {(a <= lim).mean():.0%}" for lim in limits)


def hermes(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    with open(path) as f:
        rows = [json.loads(line) for line in f]

    # Tokenize each distinct message body once; system prompts repeat heavily.
    texts = {m["value"] for r in rows for m in r["conversations"]}
    texts = list(texts)
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    ntok = dict(zip(texts, map(len, ids)))

    sys_hashes, toolsets, categories = Counter(), Counter(), Counter()
    sys_tokens, calls_per_task, tool_calls_per_task = [], [], []
    prompt_tokens, output_tokens, final_ctx = [], [], []
    parallel_turns = 0
    for r in rows:
        conv = r["conversations"]
        categories[r.get("category")] += 1
        toolsets[tuple(sorted(t["function"]["name"] for t in json.loads(r["tools"])))] += 1
        if conv[0]["from"] == "system":
            sys_hashes[hashlib.sha1(conv[0]["value"].encode()).hexdigest()] += 1
            sys_tokens.append(ntok[conv[0]["value"]])
        ctx = calls = tcalls = 0
        for m in conv:
            n = ntok[m["value"]]
            if m["from"] == "gpt":
                prompt_tokens.append(ctx)
                output_tokens.append(n)
                calls += 1
                k = m["value"].count("<tool_call>")
                tcalls += k
                parallel_turns += k > 1
            ctx += n
        calls_per_task.append(calls)
        tool_calls_per_task.append(tcalls)
        final_ctx.append(ctx)

    print(f"tasks: {len(rows):,}")
    print(f"categories: {dict(categories.most_common())}")
    top = sum(c for _, c in sys_hashes.most_common(5))
    print(f"distinct system prompts: {len(sys_hashes):,}  (top-5 cover {top / len(rows):.0%} of tasks)")
    print(f"distinct tool sets: {len(toolsets):,}  largest covers {toolsets.most_common(1)[0][1] / len(rows):.0%}")
    print(f"system prompt tokens:       {pct(sys_tokens)}")
    print(f"LLM calls per task:         {pct(calls_per_task)}")
    print(f"tool calls per task:        {pct(tool_calls_per_task)}")
    print(f"turns with parallel calls:  {parallel_turns / max(1, sum(calls_per_task)):.0%} of LLM calls")
    print(f"prompt tokens per call:     {pct(prompt_tokens)}")
    print(f"   share of calls           {share_at_most(prompt_tokens)}")
    print(f"output tokens per call:     {pct(output_tokens)}")
    print(f"final context per task:     {pct(final_ctx)}")
    print(f"   share of tasks           {share_at_most(final_ctx)}")


def trace_commons(path):
    import pandas as pd

    df = pd.read_parquet(path)
    print(f"sessions by harness: {dict(Counter(df['harness']))}  (analyzing claude_code)")
    df = df[df["harness"] == "claude_code"]

    gaps_by_tool = defaultdict(list)
    calls_per_session, prompt_tokens, cached_share, output_tokens = [], [], [], []
    for trace in df["trace"]:
        events = [json.loads(x) for x in trace]
        pending = {}  # tool_use_id -> (timestamp, tool name)
        usage_by_request = {}
        for e in events:
            if e.get("type") not in ("assistant", "user"):
                continue
            ts = datetime.fromisoformat(e["timestamp"])
            msg = e.get("message") or {}
            if e["type"] == "assistant" and "usage" in msg and e.get("requestId"):
                usage_by_request[e["requestId"]] = msg["usage"]
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if c.get("type") == "tool_use":
                    pending[c["id"]] = (ts, c["name"])
                elif c.get("type") == "tool_result" and c.get("tool_use_id") in pending:
                    start, name = pending.pop(c["tool_use_id"])
                    gaps_by_tool[name].append((ts - start).total_seconds())
        calls_per_session.append(len(usage_by_request))
        for u in usage_by_request.values():
            read, created = u.get("cache_read_input_tokens", 0), u.get("cache_creation_input_tokens", 0)
            total = u.get("input_tokens", 0) + read + created
            prompt_tokens.append(total)
            cached_share.append(read / total if total else 0)
            output_tokens.append(u.get("output_tokens", 0))

    all_gaps = [g for gs in gaps_by_tool.values() for g in gs]
    print(f"LLM calls per session:      {pct(calls_per_session)}")
    print(f"prompt tokens per call:     {pct(prompt_tokens)}")
    print(f"   share of calls           {share_at_most(prompt_tokens)}")
    print(f"cache-read share of prompt: {pct(cached_share, ps=(10, 50, 90))}")
    print(f"output tokens per call:     {pct(output_tokens)}")
    print(f"tool gap seconds (all):     {pct(all_gaps, ps=(50, 90, 99))}")
    print("tool gap seconds by tool (top 10 by count):")
    for name, gs in sorted(gaps_by_tool.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"   {name:<16} {pct(gs, ps=(50, 90))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["hermes", "trace-commons"])
    p.add_argument("path")
    args = p.parse_args()
    {"hermes": hermes, "trace-commons": trace_commons}[args.dataset](args.path)
