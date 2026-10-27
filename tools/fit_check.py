"""Memory fit check for a target model, optionally paired with a drafter.

Answers: after loading weights, how many KV-cache tokens fit, and how many
concurrent sequences of a given context length is that?

    KV bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes/elem

Weights are the summed size of the repo's .safetensors files, which already
reflects quantization (AWQ, FP8, ...). Assumes standard full attention; models
with sliding-window or linear-attention layers need a different formula.

Usage:
    uv run tools/fit_check.py Qwen/Qwen3-4B-Instruct-2507 \
        Qwen/Qwen3-4B-Instruct-2507:Qwen/Qwen3-0.6B --platform rtx5070-laptop-12g
"""

import argparse
import json
import math

from huggingface_hub import HfApi, hf_hub_download

GiB = 1024**3

# Total accelerator memory visible to the runtime, in GiB.
PLATFORMS = {
    "rtx5070-laptop-12g": 11.9,  # 12 GB card as reported by nvidia-smi; verify on the machine
    "rtx5070-laptop-8g": 7.9,
    "m5pro-24g": 17.76,  # Metal max_recommended_working_set_size on the M5 Pro
}

api = HfApi()


def load_config(repo: str) -> dict:
    with open(hf_hub_download(repo, "config.json")) as f:
        cfg = json.load(f)
    # Speculators-format EAGLE heads nest the decoder-layer config.
    return cfg.get("transformer_layer_config", cfg)


def kv_bytes_per_token(cfg: dict, kv_elem_bytes: int) -> int:
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    return 2 * cfg["num_hidden_layers"] * kv_heads * head_dim * kv_elem_bytes


def weight_bytes(repo: str) -> int:
    info = api.model_info(repo, files_metadata=True)
    return sum(s.size for s in info.siblings if s.rfilename.endswith(".safetensors"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pairs", nargs="+", help="TARGET or TARGET:DRAFTER (Hugging Face repo ids)")
    p.add_argument("--platform", choices=PLATFORMS, default="rtx5070-laptop-12g")
    p.add_argument("--mem-util", type=float, default=0.90, help="fraction of memory the runtime may claim (vLLM gpu_memory_utilization)")
    p.add_argument("--overhead-gib", type=float, default=1.0, help="activations, CUDA graphs, sampler workspace")
    p.add_argument("--kv-bytes", type=int, default=2, help="2 = fp16/bf16 KV cache, 1 = fp8 KV cache")
    p.add_argument("--spec-tokens", type=int, default=5, help="lookahead slots reserved per sequence when speculating")
    p.add_argument("--block-size", type=int, default=16, help="KV block size in tokens (vLLM default 16)")
    p.add_argument("--contexts", type=int, nargs="+", default=[512, 4096, 16384])
    args = p.parse_args()

    budget = PLATFORMS[args.platform] * GiB * args.mem_util
    print(f"platform={args.platform}  budget={budget / GiB:.2f} GiB (util {args.mem_util})  "
          f"overhead={args.overhead_gib} GiB  kv_bytes={args.kv_bytes}  spec_tokens={args.spec_tokens}\n")
    ctx_hdr = "".join(f"{'seqs@' + str(c):>11}" for c in args.contexts)
    print(f"{'config':<62}{'weights':>9}{'KV/tok':>9}{'KV pool':>9}{'KV tokens':>11}{ctx_hdr}")
    print(f"{'':<62}{'GiB':>9}{'KiB':>9}{'GiB':>9}{'':>11}")

    for pair in args.pairs:
        repos = pair.split(":")
        w = sum(weight_bytes(r) for r in repos)
        kv_tok = sum(kv_bytes_per_token(load_config(r), args.kv_bytes) for r in repos)
        pool = budget - w - args.overhead_gib * GiB
        cap_tokens = max(0, int(pool // kv_tok))
        lookahead = args.spec_tokens if len(repos) > 1 else 0

        seqs = []
        for c in args.contexts:
            blocks_per_seq = math.ceil((c + lookahead) / args.block_size)
            seqs.append(cap_tokens // (blocks_per_seq * args.block_size))
        label = " + ".join(r.split("/")[-1] for r in repos)
        cols = "".join(f"{s:>11}" for s in seqs)
        print(f"{label:<62}{w / GiB:>9.2f}{kv_tok / 1024:>9.0f}{pool / GiB:>9.2f}{cap_tokens:>11,}{cols}")


if __name__ == "__main__":
    main()
