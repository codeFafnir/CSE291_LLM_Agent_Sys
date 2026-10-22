# Project plan

**Question.** Under a fixed single-GPU memory budget, when should memory and compute go to
admitting another ready agent step instead of speculative work for requests already running?

**Primary metric: task goodput.** Successful tasks completed per second, counting only tasks
that finish within a fixed task-latency target. Token throughput can rise while task goodput
falls, for example when the drafter's KV crowds out admissions or prefill is repeated.

Background for every term used here is in [CONCEPTS.md](CONCEPTS.md). NVIDIA bring-up steps
are in [SETUP_NVIDIA.md](SETUP_NVIDIA.md).

---

## 1. What the first measurements showed (2026-10-21)

### Memory fit on a 12 GB RTX 5070 Laptop (`tools/fit_check.py`, 90% utilization, 1 GiB overhead)

| Config | Weights GiB | KV KiB/token | KV tokens | Seqs @ 4K | Seqs @ 16K |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-FP8, no drafter | 4.83 | 144 | 35,508 | 8 | 2 |
| Qwen3-4B-FP8 + Qwen3-0.6B (draft model) | 6.23 | 256 | 14,239 | 3 | 0 |
| Qwen3-4B-FP8 + EAGLE-3 head | 5.24 | 148 | 31,666 | 7 | 1 |
| Same three with FP8 KV cache | | 72 / 128 / 74 | 71K / 28K / 63K | 17 / 6 / 15 | 4 / 1 / 3 |
| Qwen3-4B BF16, no drafter | 7.49 | 144 | 16,147 | 3 | 0 |

- **A separate draft model nearly doubles KV per token.** Qwen3-0.6B has 28 layers × 8 KV heads × 128 dims, so it
  costs 112 KiB per token against the target's 144. This is the project's core tension in one number.
- **An EAGLE-3 head costs about 4 KiB per token** because it is a single layer.
- **On an 8 GB card almost nothing fits** (rerun with `--platform rtx5070-laptop-8g`). The RTX 5070 Laptop
  ships in **8 GB and 12 GB variants**, so step 1 of bring-up is confirming which one we have.

### Workloads (`tools/inspect_traces.py`)

| | Hermes filtered (Apache-2.0) | trace-commons (CC-BY-4.0) |
|---|---|---|
| Tasks | 3,679 | 28 Claude Code sessions |
| Shared prefix | 6 distinct system prompts (2.4K–6.2K tokens) cover all tasks | ~27K-token system prompt + tools |
| LLM calls per task | p50 15, p90 26 | p50 141 |
| Prompt tokens per call | p50 9.9K, p90 39K; 69% ≤ 16K | p50 105K (too large for our GPUs) |
| Output tokens per call | p50 95, p90 786, p99 5.5K | p50 225 |
| Final context per task | p50 18K; **45% of tasks ≤ 16K** | — |
| Timestamps | none | yes: **tool gap p50 1.1 s, p90 7.8 s, p99 125 s** |

**Decision:** Hermes is the replay workload. trace-commons supplies the empirical tool-gap
distribution and is not replayed. Exclude human-in-the-loop tools such as `AskUserQuestion`, and gaps that include
permission prompts, from the tool-time distribution.

### Mac single-stream speculation (`tools/mac_spec_smoke.py`, Qwen3-4B-4bit + Qwen3-0.6B-4bit, MLX)

Two runs: a real agent step (6.2K-token prompt, 39 output tokens) and the same prompt forced to a 512-token answer.

| Draft k | Prefill tok/s | Decode tok/s | Tokens per verify |
|---:|---:|---:|---:|
| none | 2,292–2,406 | 76.2–78.5 | 1.00 |
| 2 | 1,683–1,806 | 74.6–74.9 | 2.17–2.37 |
| 4 | 1,615–1,798 | 64.2–66.6 | 3.00–3.26 |
| 6 | 1,560–1,767 | 52.0 | 3.55–3.71 |

High acceptance did not produce a speedup. The 0.6B drafter costs roughly 60% of a target token
on this GPU, and the drafter also prefills the whole prompt. For a real agent step (6.2K-token prompt,
39 output tokens) speculation made the step about 30% slower. Agent steps are prefill-heavy, so
**prefix reuse matters at least as much as speculation**. Phase 1 must measure both.

---

## 2. Decisions

| Area | Choice | Status |
|---|---|---|
| Primary platform | RTX 5070 Laptop, Linux or WSL2, **vLLM 0.32.0** | Verify VRAM (8 vs 12 GB) and sm_120 kernels first |
| Fallback engine | SGLang 0.5.22 | Only if vLLM blocks us |
| Mac track (MacBook owner) | **vllm-metal 0.32.0** (Homebrew) as the main Mac engine: same scheduler, paged KV, prefix caching and `/metrics` as NVIDIA vLLM. mlx-lm as the single-stream baseline. See [MAC_TRACK.md](MAC_TRACK.md) | Working (2026-10-21). vllm-metal drafts only greedy requests and needs `--no-async-scheduling`; the mlx-lm server can't batch while speculating |
| NVIDIA track (RTX 5070 owners) | See SETUP_NVIDIA.md | Not started |
| Target model | `Qwen/Qwen3-4B-FP8`, thinking disabled | Proposed; fallback `Qwen/Qwen3-8B-AWQ` if tool-call quality is too low |
| AR drafter | `Qwen/Qwen3-0.6B` via `"method": "draft_model"` | Proposed |
| Adaptive control | `AngelSlim/Qwen3-4B_eagle3` + vLLM Dynamic SD (`num_speculative_tokens_per_batch_size`) | vLLM's DSD is only tested with EAGLE/EAGLE-3/DFlash, so the stock adaptive arm uses EAGLE-3 |
| Optional arm | `suffix` / `ngram` speculation (no drafter, no extra KV) | Cheap to add, relevant for agents that copy text |
| KV dtype | BF16 baseline; FP8 KV as a capacity ablation | FP8 KV doubles capacity but may change outputs |
| Context cap | `max_model_len` 16,384; Hermes tasks with final context ≤ 16K | ~1,650 tasks |
| Splits | 70/30 dev/held-out by hash of task id, stratified by category | Freeze before any tuning |
| Replay mode | Self-consistent replay (see below) | Proposed |
| Harness | Plain `asyncio` + OpenAI-compatible HTTP client; engine in its own venv | Proposed |
| Correctness | Executable tasks; candidates BFCL multi-turn, τ²-bench | **Open**: Hermes has no tests |
| Pinning | Exact model revision hashes, tokenizer, chat template, engine version, flags, saved per run | Do it at first real run |

**Self-consistent replay.** Each step's prompt is the task's real history so far: system prompt, user
message, then the **model's own previous outputs** interleaved with the **recorded** tool
responses. Tool delays are injected, either sampled from trace-commons or fixed by the sweep. `max_tokens` is
capped at the recorded output length, and the actual length is recorded. This keeps prefixes
append-only, so prefix-cache reuse behaves like a real agent, and acceptance comes from the model's real
distribution. It cannot measure task correctness because tool results don't react to the model's calls.
That is what the executable benchmark is for. Never substitute recorded assistant text for generated text:
it breaks prefix reuse at every step and makes acceptance numbers meaningless.

---

## 3. Harness architecture

```
 task source ──► arrival process ──► agent executor (one coroutine per live task)
 (Hermes,         (open-loop:          step i: build prompt ─► ADMISSION ─► engine client ─► vLLM
  synthetic)       Poisson/bursty)              ▲              (FIFO sem,     (streams,      (own scheduler,
                                                │               later KV-     timestamps)    KV pool,
                                                │               aware)                       prefix cache,
                                                └── tool gap (sleep, slot released) ◄─────── spec decode)
                                                                                    │
 metrics: per-request timeline (JSONL) + /metrics scraper + nvidia-smi sampler ─────┘
```

There are three separate concurrency limits; log all three:
1. **Live tasks**: agents in flight, including those waiting on a tool.
2. **Admitted requests**: our admission semaphore, which is the knob we control.
3. **Engine running sequences**: bounded by vLLM `max_num_seqs`, `max_num_batched_tokens`, and free
   KV blocks. Set the engine limits loose so queueing happens in our admission layer, where we can see it.
   Also scrape `vllm:num_requests_waiting` to catch queueing inside vLLM.

Phase 2's speculation-aware admission replaces the FIFO semaphore with a policy. It uses our KV-demand
estimate plus the engine's live KV usage, and the engine's allocator stays the final authority.

---

## 4. Milestones

Dates assume the Fall 2026 quarter ends with finals week Dec 7–11; adjust to course deadlines.

| Week | Dates | Deliverable |
|---|---|---|
| W4 | Oct 21–27 | NVIDIA verified (VRAM, sm_120, vLLM serving). vLLM's reported KV capacity checked against `fit_check`. Hermes ≤16K subset and splits frozen. Harness replays one task end to end on Mac (mlx-lm server) and NVIDIA. |
| W5 | Oct 28–Nov 3 | FIFO harness: open-loop arrivals, admission semaphore, tool-gap injection, per-request JSONL, `/metrics` and `nvidia-smi` samplers. Synthetic sweeps **without speculation**. |
| W6 | Nov 4–10 | Speculation arms: none, fixed-K draft model, fixed-K EAGLE-3, DSD. Concurrency sweeps. Hermes replay. |
| W7 | Nov 11–17 | Bottleneck analysis and **decision gate**: does KV capacity limit task goodput when speculating? Yes → Phase 2. No → pivot to verification scheduling or retention, as the proposal says. |
| W8–9 | Nov 18–Dec 1 | Phase 2 speculation-aware admission plus the required controls. |
| W10 | Dec 2–8 | Held-out evaluation, repeated runs with CIs, report. |

## 5. First Phase 1 sweep (single-factor before combinations)

Base point: Qwen3-4B-FP8, 4K context, 256-token output cap, 4 requests per task, 1 s tool gap, shared
system prompt, Poisson arrivals. Vary one factor at a time:

- Context {512, 4K, 16K}; with a draft model, 16K only fits with FP8 KV.
- Output cap {32, 256, 1024}.
- Requests per task {1, 4, 8}, plus 15, the Hermes median.
- Tool gap {0, 0.1, 1, 5} s, then the trace-commons empirical distribution.
- Speculation arm {none, draft k∈{2,3,5}, EAGLE-3 k∈{2,3,5}, DSD}.
- Arrival rate: increase until p95 task latency breaks the target. That knee sets the goodput point.

Record actual output lengths. Report each configuration as a latency–throughput curve, not a single point.

## 6. Risks

| Risk | Mitigation |
|---|---|
| GPU is the 8 GB variant | Qwen3-1.7B target with 0.6B drafter, FP8 KV, and a 4K–8K context cap; or move experiments to the Mac with llama.cpp |
| sm_120 kernel gaps (FP8, FlashInfer, AWQ) | Follow SETUP_NVIDIA.md checks; try `--attention-backend TRITON_ATTN`; fall back to BF16 1.7B or SGLang |
| Laptop thermal and power throttling adds noise | AC power, fixed performance mode, log clocks and power, repeat runs, report CIs |
| WSL2 quirks (pinned memory, RAM cap) | Prefer native Linux; otherwise set `.wslconfig` memory and use a Linux-side filesystem |
| Small model makes poor tool calls | Systems metrics come from replay; correctness from an executable benchmark that may use Qwen3-8B-AWQ |
| Spec vs no-spec outputs diverge numerically | Compare distributions over many tasks, never single outputs |
