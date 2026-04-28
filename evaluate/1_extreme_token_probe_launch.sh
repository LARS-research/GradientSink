#!/bin/bash
set -euo pipefail

# =========================================================
# 1. User config
# =========================================================
FREE_MEMORY_THRESHOLD=50000
NUM_GPUS_TO_USE=1
OMP_THREADS=4

SCRIPT="evaluate/1_extreme_token_probe.py"
CONDA_ENV_PATH="path/to/your/env"

# checkpoint format from train.py:
#   /.../runs/<run_name>/checkpoints/step_*.pt
CKPT_GLOB="train/runs/vscale_1B/checkpoints/step_*.pt"

OUT_DIR="evaluate/outputs/extreme_token_probe"
TAG=""

TOKENIZER_PATH="path/to/your/GPT2-tokenizer"
DATASET_PATH="path/to/your/datasets--allenai--c4/snapshots"
DATA_NAME="en"
SPLIT="validation"

# Probe config
DTYPE="bf16"                # bf16 / fp32
EVAL_SEQ_LEN=2048
GLOBAL_BATCH=512
LOCAL_BATCH=8
SEED=42
ADD_BOS="false"
ADD_EOS="true"
OVERWRITE_PROBE_CACHE="false"

# Attention metrics
K=16
EPS_LIST="0.3"
EPS_T=64

# Optional rank switch
COMPUTE_RANK="false"        # true / false
RANK_MATS="all"             # qkv / qkvo / all

# Plot control
DO_PLOTS="true"
ONLY_PLOT="false"

# =========================================================
# 2. Detect free GPUs
# =========================================================
echo "--- 1. Detecting idle GPUs (free memory > ${FREE_MEMORY_THRESHOLD} MB) ---"

FREE_GPUS_CSV=$(
    "${CONDA_ENV_PATH}/bin/python" - <<PY
import GPUtil as GPU
threshold = ${FREE_MEMORY_THRESHOLD}
need = ${NUM_GPUS_TO_USE}
gpus = GPU.getGPUs()
free_gpus = [g.id for g in gpus if g.memoryFree > threshold]
selected = free_gpus[:need]
print(",".join(map(str, selected)))
PY
)

if [ -z "${FREE_GPUS_CSV}" ]; then
    echo "❌ Error: no GPU satisfies free memory threshold ${FREE_MEMORY_THRESHOLD} MB."
    exit 1
fi

IFS=',' read -r -a FREE_GPUS_ARRAY <<< "${FREE_GPUS_CSV}"
NUM_FOUND_GPUS=${#FREE_GPUS_ARRAY[@]}

if [ "${NUM_FOUND_GPUS}" -lt "${NUM_GPUS_TO_USE}" ]; then
    echo "❌ Error: only found ${NUM_FOUND_GPUS} idle GPUs, but need ${NUM_GPUS_TO_USE}."
    echo "Found GPU IDs: ${FREE_GPUS_CSV}"
    exit 1
fi

echo "✅ Selected GPUs: ${FREE_GPUS_CSV}"

export CUDA_VISIBLE_DEVICES="${FREE_GPUS_CSV}"
export OMP_NUM_THREADS="${OMP_THREADS}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# =========================================================
# 3. Build args
# =========================================================
ARGS=(
    --ckpt_glob "${CKPT_GLOB}"
    --out_dir "${OUT_DIR}"
    --tokenizer_path "${TOKENIZER_PATH}"
    --dataset_path "${DATASET_PATH}"
    --data_name "${DATA_NAME}"
    --split "${SPLIT}"
    --dtype "${DTYPE}"
    --eval_seq_len "${EVAL_SEQ_LEN}"
    --global_batch "${GLOBAL_BATCH}"
    --local_batch "${LOCAL_BATCH}"
    --seed "${SEED}"
    --add_bos "${ADD_BOS}"
    --add_eos "${ADD_EOS}"
    --overwrite_probe_cache "${OVERWRITE_PROBE_CACHE}"
    --K "${K}"
    --eps_list "${EPS_LIST}"
    --eps_T "${EPS_T}"
    --compute_rank "${COMPUTE_RANK}"
    --rank_mats "${RANK_MATS}"
    --do_plots "${DO_PLOTS}"
    --only_plot "${ONLY_PLOT}"
)

if [ -n "${TAG}" ]; then
    ARGS+=( --tag "${TAG}" )
fi

# =========================================================
# 4. Launch
# =========================================================
echo "--- 2. Launching probe with ${NUM_GPUS_TO_USE} processes ---"
"${CONDA_ENV_PATH}/bin/torchrun" \
    --nproc_per_node="${NUM_GPUS_TO_USE}" \
    --master_port=29999 \
    "${SCRIPT}" \
    "${ARGS[@]}"

unset CUDA_VISIBLE_DEVICES

echo "Probe finished."
