# Mac benchmarking track

Hardware: MacBook Pro, Apple M5 Pro, 24 GB unified memory, 16-core GPU (Metal 4), macOS 26.4.
Mac results are reported separately from the NVIDIA results.

## Engines

| Engine | Serves many requests at once | Speculation | Prefix caching | Metrics | Role |
|---|---|---|---|---|---|
| **vllm-metal 0.32.0** (Homebrew) | yes: vLLM scheduler, paged KV | `draft_model`, `ngram`. Greedy requests only, needs `--no-async-scheduling` | yes | Same Prometheus `/metrics` as NVIDIA vLLM, including `spec_decode_*` | **Main engine** |
| mlx-lm 0.32.2 (`uv sync`) | yes (`--decode-concurrency`), but **not while a draft model is loaded** | draft model, one request at a time | LRU prompt cache | `usage.prompt_tokens_details.cached_tokens` only | Single-stream baseline |

vllm-metal shares its version, scheduler, flags and metric names with the vLLM the NVIDIA track uses,
so the two tracks' harness code and metrics line up.

## Setup

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal      # gives the `vllm` command
uv sync                                               # harness environment
git submodule update --init third_party/vllm-metal third_party/mlx-lm third_party/mlx   # engine sources
```

## Before every benchmark run

- **Free memory.** At the first start only 6.3 GB of 24 GB was free, and vllm-metal warned that its KV
  pool was bigger than free memory, which risks swapping. Quit other apps. Something else on this Mac already listens on
  **port 8000**, so the servers here use **8100** and the client uses `127.0.0.1`.
- **Power.** Plug in and turn Low Power Mode off.
- **Swap check.** Record `vm_stat | grep Swapouts` before and after. If it changed, the run swapped; discard it.
- **One configuration per server start.** Restart `vllm serve` between configurations.

## Starting the server

Shared flags: `--served-model-name target --max-model-len 8192 --max-num-seqs 8 --enable-prefix-caching --gpu-memory-utilization 0.4 --port 8100`

```bash
# 1. No speculation (also run once with --no-async-scheduling as the fair control for 3 and 4)
vllm serve mlx-community/Qwen3-4B-4bit <shared flags>

# 3. Draft model (AR drafting)
vllm serve mlx-community/Qwen3-4B-4bit <shared flags> --no-async-scheduling \
  --speculative-config '{"method":"draft_model","model":"mlx-community/Qwen3-0.6B-4bit","num_speculative_tokens":3}'

# 4. N-gram (copies from the prompt; no extra model or KV)
vllm serve mlx-community/Qwen3-4B-4bit <shared flags> --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

The startup log line `KV cache size: N tokens` is the KV capacity. Then run the load test:

```bash
uv run python -I tools/quick_load.py data/raw/hermes/data/train.jsonl --port 8100 --concurrency 1 4 8
curl -s 127.0.0.1:8100/metrics | grep -E "^vllm:(prefix_cache|spec_decode|num_preemptions)"
```

## Long-context bench (`bench/`, in-process, one request at a time)

Same 18 prompts (5.5K–5.7K tokens), greedy decoding and 128 output tokens as the NVIDIA bench, which runs
Qwen3-14B AWQ with an EAGLE3 head. The Mac runs Qwen3-4B 4-bit with a Qwen3-0.6B 4-bit draft model at k=3:
vllm-metal has no EAGLE3 path, and this is the pair the Mac track already measured.

```bash
/opt/homebrew/opt/vllm-metal/libexec/bin/python3.12 bench/run_vllm_metal.py   # -> bench/results/vllm_metal_*.json
uv run python bench/run_mlx.py                                                 # -> bench/results/mlx_*.json
MTL_CAPTURE_ENABLED=1 uv run python bench/profile_mlx.py                      # -> /tmp/llm-traces/mlx/*.gputrace
```

- **Context limit 8192, not 12288.** vllm-metal shrinks its KV pool to what free memory can hold. With apps open it
  fit 8,736 tokens and refused 12,288. The longest prompt plus output is ~5.8K tokens.
- **Peak memory** is the `phys_footprint` of the Python process and its children, sampled every 50 ms. That is
  Activity Monitor's Memory column and includes Metal buffers. vLLM's EngineCore is a child process.
- **Acceptance on MLX** is rebuilt from mlx-lm's per-token `from_draft` flags with vLLM's definitions
  (length = 1 + accepted / rounds, rate = accepted / drafted). Both engines agree: ~3.3 tokens per round, ~76%.
- **Profiling.** `profile_mlx.py` writes Metal frame captures: 2.9 GB for one prefill, 4.3 GB for 16 decode tokens,
  so it captures one prompt. Opening them needs full Xcode (only Command Line Tools are installed now).
  vllm-metal frame capture is not practical at this length: one prefill capture took ~10 min, wrote 6.4 GB in
  over a million files, and the engine died before decode. For vllm-metal timelines, use Instruments'
  Metal System Trace (`xcrun xctrace record --template 'Metal System Trace' --attach <EngineCore pid>`,
  also needs Xcode).

Smoke test (2026-10-27, **not a result**: on battery at 7% with apps open; vllm-metal swapped ~1.7 GB). Medians
over 18 prompts:

| Engine | TTFT p50 | TPOT p50 | Decode tok/s | Tokens per round | Peak memory |
|---|---:|---:|---:|---:|---:|
| vllm-metal, draft k=3 | 2.93 s | 18.1 ms | ~55 | ~3.3 | 9.3–9.7 GB |
| mlx-lm, draft k=3 | 3.03 s | 11.7 ms | ~85 | ~3.3 | 5.5–6.6 GB |

At one request in flight, MLX decodes ~1.5× faster with the same models and the same acceptance. vllm-metal pays
per-step scheduler and paged-KV overhead, plus two preallocated KV pools. Its gains show up with many requests in
flight and prefix reuse (table below), which this bench does not exercise. Rerun both plugged in, with apps
closed, before saving results.

## First results (2026-10-21, single runs, other apps open)

First step of real Hermes tasks (~6.2K-token prompt, 128-token output cap, greedy), 16 requests per level.
Output tokens/sec:

| Config | KV tokens | 1 in flight | 4 in flight | 8 in flight | Accepted drafts |
|---|---:|---:|---:|---:|---:|
| No spec, async on | 30,064 | 67 | 140 | 145 | — |
| No spec, async off (run 1 / run 2) | ~30,000 | 33 / 56 | 135 / 132 | 97 / 151 | — |
| Draft model, k=3 | **12,368** | 28 | 64 | 75 | 66% |
| N-gram, k=3 | 30,016 | 45 | 81 | 60 | 30% |

What it shows:
- **The helper model's memory cost matches the prediction.** KV room fell 2.4× (30K → 12K tokens), which is what
  `tools/fit_check.py` predicted from the model configs.
- **Speculation is slower on this Mac today.** At 4 in flight, the most stable level, no speculation gives
  ~130–140 tok/s, the draft model 64, and n-gram 81. That holds even though the drafter's guesses were 66% accepted.
- **Prefix caching does the heavy lifting.** 97% of prompt tokens were served from cache because tasks share a
  system prompt, so time to first token was about 0.1–0.3 s instead of ~2.6 s.
- **Results are noisy.** Two identical runs differed 1.6× at 1 and at 8 in flight. Every number needs ≥3
  repeats with a closed-app, plugged-in setup before it goes in the report.
- **The GPU saturates around 4–8 in flight.** Above that, requests mostly wait longer.

## Next steps on the Mac track

1. Repeat the four configurations 3× each with apps closed and report means with confidence intervals.
2. Turn `quick_load.py` into the real harness: open-loop arrivals, full multi-step tasks (self-consistent
   replay), tool delays, an admission semaphore, per-request JSONL logs, and a `/metrics` + `vm_stat` sampler.
   Keep it engine-agnostic so the NVIDIA track can reuse it.
3. Phase 1 sweeps from PLAN.md §5, using the context lengths that fit on the Mac.
4. Mac memory accounting: KV numbers from vllm-metal logs and `/metrics`, process memory from `footprint`, swap from
   `vm_stat`, and GPU power/frequency from `sudo powermetrics --samplers gpu_power`.
