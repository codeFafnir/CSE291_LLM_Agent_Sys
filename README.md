# CSE 291 LLM Agent Systems

Course project for CSE 291: Systems for LLMs and AI Agents. This repository measures how a speculative-decoding server behaves while an agent works through terminal tasks.

The agentic experiment runs [Harbor](https://github.com/laude-institute/harbor)'s `terminus-2` agent against `Qwen/Qwen3-14B-AWQ` served by a pinned vLLM fork, with the `RedHatAI/Qwen3-14B-speculator.eagle3` head proposing 3 tokens. Tasks come from `nvidia/Nemotron-Terminal-Synthetic-Tasks`. Summarization is off, so a prompt that crosses the 20,000-token window fails the trial as `ContextLengthExceededError` instead of being shortened.

`non_agentic_pipeline/` is a separate single-request speed bench against the same server. It does not run Harbor.

## Repository layout

| Path | Role |
| --- | --- |
| `agentic_pipeline/bench/serve_vllm.sh` | Start the vLLM server with the fixed knobs below. |
| `agentic_pipeline/bench/serve_sglang.sh` | Same role for SGLang. The recorded comparison used vLLM only. |
| `agentic_pipeline/bench/otel_collector.sh` | OpenTelemetry collector that stores per-request vLLM spans. |
| `agentic_pipeline/experiments/run_baseline.sh` | One Harbor job. Writes the run directory and appends `all_runs.csv`. |
| `agentic_pipeline/experiments/fetch_nemotron_terminal.sh` | Download and extract the dataset once under `/data`. |
| `agentic_pipeline/experiments/select_nemotron_smoke.py` | Copy a seeded easy/medium sample into a flat Harbor task directory. |
| `agentic_pipeline/experiments/bootstrap_ec2.sh` | One-time box setup: `/data`, both forks, Harbor, model weights. |
| `third_party/vllm`, `third_party/sglang` | Serving forks, branch `cse291-agent-evict`. Do not upgrade them. |
| `non_agentic_pipeline/` | SPEED-Bench client. See that directory's scripts for its own entry points. |

## Fixed configuration

These knobs are part of the experiment. Change them only by setting the env vars the scripts already require, and record the new values with the run.

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3-14B-AWQ` |
| Speculator | `RedHatAI/Qwen3-14B-speculator.eagle3`, 3 draft tokens |
| Weight dtype / KV dtype | float16 / float16 |
| Prefix caching | on |
| `MAX_MODEL_LEN` | 20000 |
| `MAX_NUM_SEQS` | 8 |
| `MAX_BATCHED_TOKENS` | 8192 |
| `EVICTION_POLICY` | `lru` |
| `PREEMPTION_POLICY` | `fcfs` |
| GPU memory utilization | 0.90 |
| Agent | Harbor `terminus-2` |
| Max completion tokens | 2048 |
| Thinking | off |
| Summarization | off (`enable_summarize=false`, `proactive_summarization_threshold=0`) |
| Concurrency | 1, then 4, on the same server |

The unquantized Qwen3-14B weights are about 29.5 GB and do not fit a 24 GB GPU. The AWQ checkpoint is the one these runs serve.

Versions recorded on the g6.2xlarge runs: vLLM `0.32.1.dev1+g4b7cdeae4`, SGLang `0.5.23.dev1+gee1aa8663`, Harbor `0.25.0`. Those versions come from the forks and from `uv tool install 'harbor>=0.22'`. Do not bump them to chase a newer release.

## Machine

Runs used one `g6.2xlarge` (NVIDIA L4, 24 GB) in `us-west-2`. A 300 GB EBS volume is mounted at `/data` and holds weights, the dataset, Docker image layers, and results. The instance store is not used.

`agentic_pipeline/experiments/bootstrap_ec2.sh` is the one-time setup. It is safe to re-run. It installs `uv`, builds `/data/envs/vllm` and `/data/envs/sglang` from the forks, installs Harbor, and downloads the model and speculator. Re-check that the install did not change `vllm`, `torch`, `transformers`, or `protobuf` before trusting a box.

Harbor task containers call the server at `host.docker.internal`. That name must resolve to the host (`127.0.0.1` when Docker uses host networking for the published port). Each Nemotron task Dockerfile is built locally with `--force-build`. The `docker_image` field inside `task.toml` points at an internal registry and is not pullable. The base image is `ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest`. Put Docker's data root on `/data`. The root volume is too small for 30 image builds.

## Dataset

The dataset is downloaded once and reused.

```bash
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/data/hf HUGGINGFACE_HUB_CACHE=/data/hf/hub TMPDIR=/data/tmp
bash agentic_pipeline/experiments/fetch_nemotron_terminal.sh
```

That writes `/data/datasets/nemotron-terminal/extracted`. A second run exits immediately when `extracted/.done` exists.

The 30-task set is 20 tasks from `easy_5000` and 10 from `medium_20000`, sampled with seed 291. Harbor needs the task directories as immediate children of the path it is given.

```bash
python agentic_pipeline/experiments/select_nemotron_smoke.py \
  --extracted-dir /data/datasets/nemotron-terminal/extracted \
  --out-dir /data/datasets/nemotron-terminal/set30 \
  --ids-out /data/datasets/nemotron-terminal/set30_ids.txt \
  --easy-count 20 \
  --medium-count 10 \
  --seed 291
```

`run_baseline.sh` reads those two paths for the `nemotron-30` job. Other jobs:

| Job | Tasks |
| --- | --- |
| `nemotron-1` | First id in the 5-task smoke list. Use this to check the pipeline. |
| `nemotron-smoke` | 5 Nemotron tasks. |
| `nemotron-30` | The 20 + 10 set above. |
| `smoke`, `easy30` | SWE-bench Pro ids, not the Nemotron set. |

## Run the experiment

Work from `agentic_pipeline` with the vLLM environment active and `~/.local/bin` on `PATH` so `harbor` resolves.

```bash
source /data/envs/vllm/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/data/hf HUGGINGFACE_HUB_CACHE=/data/hf/hub TMPDIR=/data/tmp
cd agentic_pipeline
```

Start the server and wait until `GET /v1/models` returns. `RUN_ID` is the directory name under `/data/cse291-results`.

```bash
export RUN_ID=serve-vllm-l20000-set30
export MAX_MODEL_LEN=20000 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192
export EVICTION_POLICY=lru PREEMPTION_POLICY=fcfs
bash bench/serve_vllm.sh
```

`serve_vllm.sh` refuses to start unless every knob above is set. It turns on the idle torch profiler, starts the OTEL collector (`cse291-otel`) if it is not already up, and sends spans to `http://127.0.0.1:4317`. Leave this process running.

In another shell, with the same exports plus the server id, run concurrency 1 and then concurrency 4. The driver sends one short chat completion first and exits unless `vllm:spec_decode_num_draft_tokens_total` has moved. A run with the speculator off is discarded.

```bash
export SERVE_RUN_ID=serve-vllm-l20000-set30
export MAX_MODEL_LEN=20000 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192
export EVICTION_POLICY=lru PREEMPTION_POLICY=fcfs

CONCURRENCY=1 bash experiments/run_baseline.sh vllm nemotron-30
CONCURRENCY=4 bash experiments/run_baseline.sh vllm nemotron-30
```

Each invocation picks its own directory:

```text
/data/cse291-results/baseline-vllm-nemotron-30-c<N>-b8-t8192-l20000-lru-<UTC>/
```

Concurrency 1 on this set took about 2 hours. Concurrency 4 took about 55 minutes. Both runs share the server. Stop the server only after the second job has written `run_metrics.csv`.

## What a run writes

`run_baseline.sh` records a start timestamp, scrapes `/metrics` once a second, and runs `nvidia-smi dmon` for the whole Harbor job. After Harbor exits it writes `summary.json`, then `run_csv.py` joins the OTEL spans to the trial windows. The last trajectory is replayed through the server profiler: two warmup requests, a cache flush, then a cold pass and a warm pass.

| File | Contents |
| --- | --- |
| `run_metrics.csv` | One row for the run: latency percentiles, throughput, KV, prefix-cache hit rate, speculative acceptance. |
| `tasks.csv` | One row per task: reward, peak prompt tokens, duration, exception type. |
| `requests.csv` | One row per LLM request: queue, prefill, TTFT, decode, TPOT, end-to-end. `cached_tokens` is empty. vLLM's span does not carry it. |
| `timeseries.csv` | One row per second. Prefix hits and preemptions are cumulative counters. |
| `server_metrics.jsonl` | Raw Prometheus samples, including `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total`. |
| `traces/timeline.perfetto.json` | Per-request queue, prefill, and decode spans for a Perfetto viewer. |
| `traces/requests_raw.jsonl` | The `llm_request` spans as exported. |
| `torch_profile/{cold,warm}/` | Gzipped PyTorch profiler traces. TraceLens reads the `rank0*.pt.trace.json.gz` files. |
| `harbor/<trial>/` | Harbor trial: `result.json`, `agent/trajectory.json`, verifier logs. |
| `/data/cse291-results/all_runs.csv` | One appended row per run, same columns as `run_metrics.csv`. |

A task passes when the verifier writes a reward of 1. `ContextLengthExceededError` means the agent hit 20,000 tokens with summarization disabled. That failure is a result, not a broken run.

## Reading the agent traces

`trajectory.json` follows ATIF 1.8. Each agent step has one timestamp. `tool_calls[].arguments.duration` is the wait the model requested for `bash_command`, not a measured duration. The measured tool gap is the time from the end of one matched LLM request (`arrival_ts + e2e_s` in `requests.csv`) to the arrival of the next request in the same trial.

vLLM does not record per-request cache hits. A prefix recompute is inferred from `server_metrics.jsonl`: over the next request's prefill, `prefix_cache_hits_total / prefix_cache_queries_total` below 0.5 means more than half of that prompt was rebuilt. At concurrency 4, trial windows overlap, so `requests.csv` leaves `trial` blank when a request falls in more than one window. Join those requests to steps on `(prompt_tokens, completion_tokens)` inside the trial's `started_at`/`finished_at` window. Discard a prefill whose 1-second counter bucket also contains another prefill.

On the recorded 30-task runs, concurrency 1 kept the prefix across tool waits (run hit rate 0.914, no preemptions, peak KV usage 0.52). Concurrency 4 filled the KV cache (hit rate 0.817, 3 preemptions, peak KV usage 1.0) and rebuilt some later prompts of 10k–19k tokens after waits of about one second. Speculative acceptance stayed at about 0.68 in both runs. Output throughput went from 27 tok/s to 60 tok/s, and wall time went from 2 h 05 m to 55 m. Passes went from 4/30 to 2/30.

## SGLang

`run_baseline.sh sglang <job>` targets port 30000 and reads request records from the server's `sglang-request-metrics` directory instead of the OTEL collector. Start it with `bench/serve_sglang.sh` and the same knobs. The 30-task comparison above was not repeated on SGLang.
