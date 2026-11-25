#!/usr/bin/env bash
# Prep SWE-bench Verified django images on the box, outside any experiment window.
#
#   COUNT=30 SEED=291 DATASET_JSONL=/data/swebench/verified.jsonl OUT_DIR=/data/swebench experiments/prep_images.sh
#
# Required env: COUNT SEED DATASET_JSONL OUT_DIR. Optional env: PREP_PY DOCKER override the box defaults below.
# Writes ${OUT_DIR}/instances_verified.jsonl and ${OUT_DIR}/images_ready.jsonl, pulls each distinct image
# (linux/amd64), then probes python3 in the FIRST image only and logs "probe_ok <image>" to ${OUT_DIR}/probe.log.
# Exit codes: 2 bad env, 6 python3 probe failed (a swe-rex prebuild is required; it is not done here).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREP_PY="${PREP_PY:-/data/envs/sweagent/bin/python}"
DOCKER="${DOCKER:-docker}"

for name in COUNT SEED DATASET_JSONL OUT_DIR; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

INSTANCES_OUT="${OUT_DIR}/instances_verified.jsonl"
READY_OUT="${OUT_DIR}/images_ready.jsonl"
mkdir -p "${OUT_DIR}"

"${PREP_PY}" "${REPO_ROOT}/swesmith/prep_images.py" \
  --dataset-jsonl "${DATASET_JSONL}" --count "${COUNT}" --seed "${SEED}" \
  --out-instances "${INSTANCES_OUT}" --out-ready "${READY_OUT}"

IMAGES="$("${PREP_PY}" -c '
import json, sys
seen = []
for line in open(sys.argv[1]):
    image = json.loads(line)["image"]
    if image not in seen:
        seen.append(image)
print("\n".join(seen))
' "${READY_OUT}")"

while IFS= read -r image; do
  echo "pulling ${image}"
  "${DOCKER}" pull --platform linux/amd64 "${image}"
done <<< "${IMAGES}"

FIRST_IMAGE="$(head -n 1 <<< "${IMAGES}")"
if ! "${DOCKER}" run --rm --entrypoint python3 "${FIRST_IMAGE}" --version; then
  echo "python3 probe failed in ${FIRST_IMAGE}: a swe-rex prebuild is required" >&2
  exit 6
fi
echo "probe_ok ${FIRST_IMAGE}" >> "${OUT_DIR}/probe.log"
