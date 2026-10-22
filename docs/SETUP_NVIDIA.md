# NVIDIA bring-up (RTX 5070 Laptop)

**Status: not yet run on the actual machine.** Each step says what to expect. Fill in the table at the
bottom with what you actually see; those values become the pinned configuration.

## 0. Pick the OS

Native Linux (Ubuntu 24.04) is the most reliable choice for vLLM. vLLM does not run natively on Windows, so on
Windows use **WSL2 + Ubuntu 24.04**:

- Install the NVIDIA driver **on Windows only**. Never install a Linux NVIDIA driver inside WSL; the
  Windows driver already exposes the GPU to WSL.
- Keep the repo and the Hugging Face cache on the Linux filesystem (`~/...`), not `/mnt/c`, because
  cross-filesystem I/O is much slower.
- WSL caps RAM at 50% of the host by default. Raise it in `%UserProfile%\.wslconfig`:
  ```
  [wsl2]
  memory=24GB
  ```
- vLLM may warn that pinned memory is unavailable under WSL. That slows CPU↔GPU copies, which matters
  for the KV-offload extension later.

## 1. Verify the GPU (decides whether the plan holds)

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.max_limit --format=csv
nvidia-smi   # look at the process list
```

Expect `compute_cap` **12.0** (Blackwell) and a driver ≥ 570. The RTX 5070 Laptop comes in **8 GB and 12 GB**
versions: `memory.total` around 12,000 MiB means the plan holds; around 8,000 MiB means re-plan
(PLAN.md §6). If a desktop compositor or browser shows up in the process list, the dGPU is driving
the display and that VRAM won't be available to vLLM. Close those apps or switch to hybrid graphics.

## 2. Engine environment, separate from the harness

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/venvs/vllm --python 3.12
source ~/venvs/vllm/bin/activate
uv pip install "vllm==0.32.0" --torch-backend=auto
vllm download-kernels          # FlashInfer kernels; rerun after every vLLM upgrade
```

vLLM pins a specific PyTorch/CUDA build. Keeping it out of the repo's environment means `uv sync` never
fights it. The harness only talks to vLLM over HTTP, so the two environments never need to agree.

## 3. Verify PyTorch has Blackwell kernels

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); \
print(torch.cuda.get_arch_list())"
```

Expect capability `(12, 0)` and `sm_120` in the arch list. If it's missing, the wheel has no RTX 50-series
kernels and you'll get `no kernel image is available` later. Reinstall with an explicit
`--torch-backend=cu129` (or `cu130`, which needs driver ≥ 580).

## 4. First server: no speculation

```bash
vllm serve Qwen/Qwen3-4B-FP8 --served-model-name target \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 64 \
  --enable-prefix-caching --port 8000
```

In the startup log, find **`GPU KV cache size: N tokens`** and **`Maximum concurrency for 16384 tokens
per request`**. Compare them with `uv run tools/fit_check.py Qwen/Qwen3-4B-FP8`, which predicts about 35.5K tokens.
Agreement within ~15% means our memory model is trustworthy. Otherwise adjust `--overhead-gib` until it matches.

If startup crashes with kernel or attention errors, try `--attention-backend TRITON_ATTN`. If it still
fails, try `Qwen/Qwen3-4B` (BF16) to tell an FP8-kernel problem from a general one.

Smoke test (Qwen3 is a hybrid thinking model; we disable thinking per request):

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "target", "max_tokens": 64,
  "messages": [{"role": "user", "content": "Write a haiku about KV caches."}],
  "chat_template_kwargs": {"enable_thinking": false}}' | python3 -m json.tool
```

## 5. Find the metrics we'll depend on

```bash
curl -s localhost:8000/metrics | grep -E "^vllm:(kv_cache|prefix_cache|num_requests|num_preempt|spec_decode)"
```

Metric names change between versions, so write down the exact names this version uses.

## 6. Speculation arms (one server at a time)

Restart `vllm serve` from step 4 with one of these added. Each time, record the KV cache size line
again: **the drop from step 4 is the drafter's memory cost, measured.**

```bash
# A. Separate draft model (AR drafting), fixed k
--speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'

# B. EAGLE-3 head, fixed k
--speculative-config '{"method":"eagle3","model":"AngelSlim/Qwen3-4B_eagle3","num_speculative_tokens":3}'

# C. EAGLE-3 + Dynamic SD (vLLM's stock adaptive control: k shrinks as batch size grows)
--speculative-config '{"method":"eagle3","model":"AngelSlim/Qwen3-4B_eagle3","num_speculative_tokens":3,
  "num_speculative_tokens_per_batch_size":[[1,4,3],[5,16,2],[17,64,0]]}'

# Per-request acceptance in API responses (experimental)
--per-request-spec-decode-metrics summary
```

The EAGLE-3 head was trained against BF16 Qwen3-4B, so acceptance with an FP8 target may be a bit lower.
Measure it before relying on it.

## 7. Quick benchmark, before our harness exists

`vllm bench serve` is open-loop and needs no code. Check `vllm bench serve --help` for the exact flags in this version:

```bash
vllm bench serve --base-url http://localhost:8000 --model Qwen/Qwen3-4B-FP8 --served-model-name target \
  --dataset-name random --random-input-len 4096 --random-output-len 256 --random-prefix-len 2048 \
  --num-prompts 100 --request-rate 2 --seed 0
```

Run it for no speculation and for each speculation arm at request rates 1, 4, and 16. The first plot should show
speculation helping at low load and losing its advantage as load rises. If it doesn't, find out why before building more.

## 8. Log the GPU during every run

```bash
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active \
  --format=csv -lms 500 > runs/<run-name>/gpu.csv
```

Run on AC power in the laptop's maximum performance mode. A run whose SM clock drops partway through
is a throttled run; flag it or repeat it.

## Record what you found

| Item | Value |
|---|---|
| GPU name / memory.total / driver / compute cap | |
| OS (native Linux / WSL2) | |
| torch version / CUDA version | |
| KV tokens, no speculation (vLLM log) vs fit_check ≈ 35,508 | |
| KV tokens with draft_model / eagle3 | |
| Extra flags needed (attention backend, etc.) | |
| Exact metric names for KV usage, prefix hits, spec tokens | |
