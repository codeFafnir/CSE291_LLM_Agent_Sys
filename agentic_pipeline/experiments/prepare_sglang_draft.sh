#!/usr/bin/env bash
# thoughtworks/Qwen3-14B-Eagle3 is a SpecForge checkpoint but its config says LlamaForCausalLM,
# so SGLang loads it as a plain Llama. Make a local copy whose config names LlamaForCausalLMEagle3.
set -euo pipefail

SRC="$(ls -d /data/hf/hub/models--thoughtworks--Qwen3-14B-Eagle3/snapshots/*/ | head -n 1)"
DST=/data/models/Qwen3-14B-Eagle3

mkdir -p "${DST}"
ln -sf "${SRC}model.safetensors" "${DST}/model.safetensors"
python3 - "${SRC}config.json" "${DST}/config.json" <<'EOF'
import json
import sys

config = json.load(open(sys.argv[1]))
config["architectures"] = ["LlamaForCausalLMEagle3"]
json.dump(config, open(sys.argv[2], "w"), indent=2)
EOF
echo "draft ready at ${DST}"
