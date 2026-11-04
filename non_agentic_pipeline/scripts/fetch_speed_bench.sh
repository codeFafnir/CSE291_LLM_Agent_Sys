#!/usr/bin/env bash
# Download and resolve the SPEED-Bench qualitative split with NVIDIA's prepare_data.py.
# Needs DATA_ROOT. Output: ${DATA_ROOT}/speed-bench/speed/qualitative/test.parquet.
# Uses its own venv so the vLLM env is not touched. Rerunning is a no-op once .done exists.
set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then echo "missing required env var: DATA_ROOT" >&2; exit 2; fi

MODEL_OPTIMIZER_SHA=ba053c3f4cac1f22e0b4cd23ef99f3c954e1d01f
REPO_DIR="${DATA_ROOT}/Model-Optimizer"
VENV_DIR="${DATA_ROOT}/envs/specdec"
OUT_DIR="${DATA_ROOT}/speed-bench"
PARQUET="${OUT_DIR}/speed/qualitative/test.parquet"

if [[ -f "${OUT_DIR}/.done" ]]; then
  echo "already prepared: ${PARQUET}"
  exit 0
fi

mkdir -p "${DATA_ROOT}/envs" "${OUT_DIR}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/NVIDIA/Model-Optimizer.git "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch origin "${MODEL_OPTIMIZER_SHA}"
git -C "${REPO_DIR}" checkout --detach "${MODEL_OPTIMIZER_SHA}"

[[ -x "${VENV_DIR}/bin/python" ]] || python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -r "${REPO_DIR}/examples/specdec_bench/requirements.txt"

# prepare_data.py imports the specdec_bench package from its own directory.
cd "${REPO_DIR}/examples/specdec_bench"
"${VENV_DIR}/bin/python" prepare_data.py --dataset speed --config qualitative --output_dir "${OUT_DIR}"

[[ -f "${PARQUET}" ]] || { echo "prepare_data.py did not write ${PARQUET}" >&2; exit 3; }
touch "${OUT_DIR}/.done"
echo "wrote ${PARQUET}"
