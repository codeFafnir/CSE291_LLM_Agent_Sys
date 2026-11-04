#!/usr/bin/env bash
# Build the SPEED-Bench easy+medium parquet. Does not download cais/hle.
# Needs DATA_ROOT for the prepare venv. Output: non_agentic_pipeline/speedbench/qualitative/test.parquet.
# Uses its own venv so the vLLM env is not touched. Rerunning is a no-op once .done exists.
# Set HF_HOME and TMPDIR off the home directory first; Code Contests is a large download.
set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then echo "missing required env var: DATA_ROOT" >&2; exit 2; fi

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${DATA_ROOT}/envs/specdec"
OUT_DIR="${PIPELINE_ROOT}/speedbench"
PARQUET="${OUT_DIR}/qualitative/test.parquet"

if [[ -f "${OUT_DIR}/.done" ]]; then
  echo "already prepared: ${PARQUET}"
  exit 0
fi

mkdir -p "${DATA_ROOT}/envs" "${OUT_DIR}"
[[ -x "${VENV_DIR}/bin/python" ]] || python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -e "${PIPELINE_ROOT}[prepare]"

"${VENV_DIR}/bin/python" "${PIPELINE_ROOT}/speedbench/prepare_easy_medium.py" --out "${PARQUET}"

[[ -f "${PARQUET}" ]] || { echo "prepare_easy_medium.py did not write ${PARQUET}" >&2; exit 3; }
touch "${OUT_DIR}/.done"
echo "wrote ${PARQUET}"
