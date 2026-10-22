# Concepts primer: LLM serving for this project

This covers the parts of LLM serving that aren't obvious and that this project depends on. Each section
ends with what it means for our experiments. Numbers come from our own runs (see [PLAN.md](PLAN.md)).

---

## 1. Prefill and decode are different workloads

A request runs in two phases:

- **Prefill** processes the whole prompt in one forward pass. Thousands of tokens go through each
  weight matrix together, so it is **compute-bound** and fast per token: about 2,300 tok/s for Qwen3-4B on the M5 Pro.
- **Decode** produces one token per forward pass. Every step reads **all the weights** to produce
  one token per sequence, so it is **memory-bandwidth-bound** and slow per token: about 78 tok/s on the same Mac.

Batching helps decode because one read of the weights serves every sequence in the batch.
At small batch sizes, adding a sequence is nearly free. As the batch grows, decode becomes compute-bound too.

**For us:** agent steps have long prompts and short outputs. In Hermes the median call has a 9.9K-token prompt
and 95 output tokens, so a step is often **mostly prefill**, and speculation only speeds up decode.

## 2. The KV cache, and how to size it

Attention needs the key and value vectors of every earlier token. Storing them, the KV cache, avoids recomputing
the whole history at each step. Its size is:

```
KV bytes per token = 2 (K,V) × layers × kv_heads × head_dim × bytes_per_element
Qwen3-4B:  2 × 36 × 8 × 128 × 2 B = 144 KiB/token  →  a 16K-token context = 2.25 GiB
```

`kv_heads` is smaller than the number of attention heads in models that use **GQA** (grouped-query attention),
which is why GQA models are cheaper to serve. KV grows linearly with context length *and* with the number of
concurrent sequences, so on a 12 GB GPU it is usually what limits concurrency, not the weights.

**For us:** `tools/fit_check.py` does this arithmetic for any model pair. A draft model brings its own
full KV cache: Qwen3-0.6B adds 112 KiB per token on top of the target's 144.

## 3. Paged KV and the preallocated pool: why `nvidia-smi` lies

vLLM does not allocate KV per request. At startup it profiles the model, then **claims
`gpu_memory_utilization` × total memory** (default 0.9) and divides everything left after weights and
activations into fixed-size **blocks** of 16 tokens. That is PagedAttention. The startup log reports the result:

```
GPU KV cache size: 35,xxx tokens
Maximum concurrency for 16,384 tokens per request: 2.1x
```

From then on, `nvidia-smi` shows flat memory whether the cache is empty or full. Real pressure is only
visible through the engine's counters: `vllm:kv_cache_usage_perc`, preemptions, and `num_requests_waiting`.
The proposal's memory-accounting section exists because of this.

**For us:** check the logged KV capacity against `fit_check.py`. Read pressure from `/metrics`, never from
process memory. On the Mac, MLX allocates on demand, so its memory numbers mean something different.

## 4. Continuous batching and three levels of concurrency

Old-style batching waited for a whole batch to finish. **Continuous batching** adds and removes sequences
at every decode step: a finished request leaves and a waiting one joins on the next iteration. vLLM's scheduler does
this every step, limited by `max_num_seqs` (sequences), `max_num_batched_tokens` (tokens per step,
which also chunks long prefills), and free KV blocks. When blocks run out it **preempts** a sequence,
frees its blocks, and later **recomputes** it from scratch. That is the "recomputed tokens" metric.

There are three distinct concurrency numbers. The proposal insists on keeping them apart:

| Level | Who controls it | Example |
|---|---|---|
| Live tasks | the workload | 40 agents in flight, 25 of them waiting on tools |
| Admitted requests | **our harness** (semaphore) | 15 requests sent to vLLM |
| Running sequences | vLLM scheduler | 12 decoding, 3 waiting for KV blocks |

If the harness admits more than the engine can run, the queue moves *inside* vLLM where you can't see it.

## 5. Prefix caching: retained is not pinned

With automatic prefix caching, vLLM hashes each **full block of token ids together with everything before
it**. When a request finishes, its blocks are freed but **keep their contents and hashes**. A later request
that starts with the same tokens reuses them and skips that part of prefill. Freed-but-cached blocks are
**reclaimable**: they are evicted (LRU) the moment someone needs space. That is what "retained state is not
necessarily pinned" means.

Agent histories are append-only. Step *i+1*'s prompt is step *i*'s prompt + output + tool result, so a
cache hit can skip almost all prefill, **if the blocks survived the tool gap**. Long tool gaps plus memory
pressure lead to eviction and full recomputation. That interaction is central to the project.

Things that silently break reuse:
- Anything changing near the start: timestamps in the system prompt, reordered tool lists.
- **Chat-template rewriting.** Some templates strip earlier `<think>` blocks or reformat tool calls, so the
  rendered history no longer matches what the model generated.
- Detokenize-then-retokenize round trips that produce different token ids.
- Substituting recorded trace text for the model's own output during replay.

**For us:** always check `vllm:prefix_cache_hits` against `queries`. Don't assume reuse happened.
Hermes shares 6 system prompts across 3,679 tasks, a large cross-task prefix to reuse.

## 6. Speculative decoding: when it pays, and what it costs

A cheap **drafter** guesses *k* tokens. The **target** checks all *k* in one forward pass and keeps the
longest correct prefix, plus one token of its own. Rejection sampling keeps the output distribution
identical to the target's (**lossless**). If each draft token is accepted with probability α:

```
tokens per verify step  τ = (1 − α^(k+1)) / (1 − α)
speedup ≈ τ / (1 + k·c)      c = cost of one draft token / cost of one target decode step
```

It works when decode is memory-bound, so verifying *k*+1 tokens costs about the same as one, and the drafter is cheap.

What our Mac run showed: τ was 2.2–3.7, a good acceptance rate, but **c ≈ 0.6**. The 0.6B
drafter has 28 layers to the target's 36, and at batch size 1 per-layer overhead dominates. Decode got slower,
and the drafter's prefill of the 6K prompt made the whole step about 30% slower.

The extra costs, which the proposal requires accounting for:
- **Drafter weights and drafter KV** for the whole context. A draft model costs about +78% KV per token here; an EAGLE head about +3%.
- **Lookahead slots**: each running sequence reserves *k* extra KV slots.
- **Verification compute** grows as batch × (*k*+1). At high concurrency decode is already compute-bound,
  so speculation hurts. vLLM's Dynamic SD lowers *k* as the batch grows for this reason.
- **Drafter prefill** on every uncached context.

Drafter families: **draft model** (a small LLM from the same family, same tokenizer), **EAGLE/EAGLE-3**
(a 1-layer head reading the target's hidden states; cheap, but must be trained per target),
**MTP** (built into some models), and **n-gram / suffix** (copies from the prompt or earlier outputs; no
model, no KV, often strong for agents that echo code and tool output).

"Lossless" holds mathematically, not bit-for-bit. Our greedy runs with and without speculation diverged after a
few hundred tokens, because batched verification computes in a different floating-point order and flips near-ties.
In an agent, one flipped token can mean a different tool call. **Compare spec vs. no-spec over
many tasks, never one output.**

## 7. Load generation and goodput

- **Closed-loop** load (N clients, each waits for its reply before sending again) throttles itself:
  when the server slows down, clients send less, which hides queueing. **Open-loop** load (tasks arrive by
  a Poisson or bursty process regardless of server state) shows the real latency under a given offered load.
  Use open-loop, as the proposal does.
- Measure latency from the **scheduled arrival time**, not from when the client got around to sending.
  Otherwise you get *coordinated omission*: a stalled client silently drops the slow samples.
- **Goodput** counts only tasks that succeed within the deadline. Raise the arrival rate until p95 task
  latency crosses the target; the throughput at that knee is the goodput. Compare policies by their
  latency-vs-throughput curves.
- Warm up before measuring (CUDA graphs, Metal shader compilation, the first prefix-cache fill), and discard the warm-up.

## 8. What each replay mode can and can't claim

| Mode | Prompts | Outputs | Valid for | Not valid for |
|---|---|---|---|---|
| Forced/teacher-forced | recorded | recorded text forced | prefill/KV sizing | acceptance rates, prefix reuse, quality |
| Length replay (`ignore_eos`, fixed `max_tokens`) | recorded | free, forced length | throughput under controlled lengths | quality; tokens after EOS are unnatural, which skews acceptance |
| **Self-consistent replay** (our default) | own outputs + recorded tool results | free, capped length | KV, prefix reuse, acceptance, latency | task correctness |
| Live execution | own outputs + real tool results | free | everything, including success | controlled, repeatable timing |

## 9. Measurement hygiene on laptops

- A laptop GPU's clocks depend on power and temperature. Run on AC power in a fixed performance mode, log
  `clocks.sm`, `power.draw`, `temperature.gpu`, and throttle reasons during every run.
- Pin everything a result depends on: model and tokenizer **revision hashes**, chat template, engine version, every flag,
  and the random seed. Save them next to each run's output.
- Repeat runs (≥3) and report confidence intervals. Laptop variance is real.
- Mac unified memory: the GPU can use about 75% of RAM by default (Metal reports 17.8 GiB on the 24 GB M5 Pro;
  `sysctl iogpu.wired_limit_mb` can raise it). It shares that RAM with the OS and everything else running.
