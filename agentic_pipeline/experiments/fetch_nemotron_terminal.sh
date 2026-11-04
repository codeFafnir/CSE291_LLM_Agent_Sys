#!/usr/bin/env bash
# Download nvidia/Nemotron-Terminal-Synthetic-Tasks once into /data and extract it. Re-runs reuse /data.
#   bash experiments/fetch_nemotron_terminal.sh
set -euo pipefail

REPO=nvidia/Nemotron-Terminal-Synthetic-Tasks
ROOT=/data/datasets/nemotron-terminal
RAW="${ROOT}/raw"
EXTRACTED="${ROOT}/extracted"
DONE="${EXTRACTED}/.done"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME=/data/hf HUGGINGFACE_HUB_CACHE=/data/hf/hub TMPDIR=/data/tmp

if [[ -f "${DONE}" ]]; then
  echo "dataset already at ${EXTRACTED}; skipping download"
  exit 0
fi

mkdir -p "${RAW}" "${EXTRACTED}"
hf download "${REPO}" --repo-type dataset --local-dir "${RAW}"
while IFS= read -r -d '' archive; do
  echo "extracting ${archive}"
  tar -xzf "${archive}" -C "${EXTRACTED}"
done < <(find "${RAW}/skill_based" -name '*.tar.gz' -print0 | sort -z)
touch "${DONE}"
echo "dataset ready at ${EXTRACTED}"
