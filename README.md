# CSE291_LLM_Agent_Sys
Course project CSE291A: Systems for LLMs and AI Agents

Speculative decoding vs. admission under KV-memory pressure for multi-step, tool-using agent workloads.

- [docs/PLAN.md](docs/PLAN.md): research question, first measurements, decisions, milestones
- [docs/CONCEPTS.md](docs/CONCEPTS.md): serving concepts the project depends on
- [docs/MAC_TRACK.md](docs/MAC_TRACK.md): MacBook benchmarking (vllm-metal), commands and first results
- [docs/SETUP_NVIDIA.md](docs/SETUP_NVIDIA.md): RTX 5070 / vLLM bring-up checklist

## Setup (harness environment, any OS)

```bash
uv sync                      # Python 3.12 venv from uv.lock (includes mlx-lm on macOS)
```

The inference engine (vLLM) lives in its own environment; see SETUP_NVIDIA.md.

## Data

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('DJLougen/hermes-agent-traces-filtered', repo_type='dataset', local_dir='data/raw/hermes', allow_patterns=['data/*.jsonl','README.md'])
snapshot_download('trace-commons/agent-traces', repo_type='dataset', local_dir='data/raw/trace-commons', allow_patterns=['data/*.parquet','README.md'])"
```

`data/`, `models/`, and `runs/` are gitignored.

## Tools

| Script | What it answers |
|---|---|
| `uv run tools/fit_check.py TARGET[:DRAFTER] ...` | Weights, KV per token, KV capacity, concurrent sequences per context length |
| `uv run python -I tools/inspect_traces.py {hermes,trace-commons} PATH` | Steps per task, context growth, prefix sharing, tool-gap timing |
| `uv run python -I tools/mac_spec_smoke.py data/raw/hermes/data/train.jsonl` | Single-stream speculative decoding on Apple Silicon (MLX) |
| `uv run python -I tools/quick_load.py data/raw/hermes/data/train.jsonl --port 8100` | Throughput and latency at 1/4/8 requests in flight against any OpenAI-compatible server |

Run scripts that read downloaded data with `python -I` (isolated mode) so nothing in the data
directory can shadow a Python module.
