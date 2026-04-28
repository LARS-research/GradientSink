#!/bin/bash

# --- config ---
FREE_MEMORY_THRESHOLD=50000
NUM_GPUS_TO_USE=4
SCRIPT="evaluate/0_ckpt_grads_check.py"
CONDA_ENV_PATH="path/to/your/env"


# ------------------------------------
# Change batch and context length
CHECKPOINT_PATHS="train/runs/vscale_1B/checkpoints/step_*.pt"

OUTPUT_DIR="evaluate/ckpt_grads"


TOKENIZER_PATH="path/to/your/GPT2-tokenizer"
DATASET="path/to/your/datasets--allenai--c4/snapshots"
DATA_NAME="en"
SPLIT="validation"


# --- Choose GPU ---
echo "--- 1. Detecting idle GPUs (free memory > ${FREE_MEMORY_THRESHOLD} MB) ---"

FREE_GPUS_CSV=$(
    ${CONDA_ENV_PATH}/bin/python -c "
import GPUtil as GPU
# Get GPU list, sorted by memory usage ascending
gpus = GPU.getGPUs()
# Filter for GPUs with memory Free > threshold AND not currently busy
free_gpus = [g.id for g in gpus if g.memoryFree > ${FREE_MEMORY_THRESHOLD}]
# Only select the required number of GPUs
selected_gpus = free_gpus[:${NUM_GPUS_TO_USE}]
print(','.join(map(str, selected_gpus)))
"
)

IFS=',' read -r -a FREE_GPUS_ARRAY < <(echo "${FREE_GPUS_CSV}")
NUM_FOUND_GPUS=${#FREE_GPUS_ARRAY[@]}

if [ "${NUM_FOUND_GPUS}" -lt "${NUM_GPUS_TO_USE}" ]; then
    echo "❌ Error: only found ${NUM_FOUND_GPUS} idle GPUs, but need ${NUM_GPUS_TO_USE}."
    echo "Found GPU IDs: ${FREE_GPUS_CSV}"
    exit 1
fi

echo "✅ Selected GPUs: ${FREE_GPUS_CSV}"


# Restrict visible GPUs; inside torchrun they become cuda:0,1,2,...
export CUDA_VISIBLE_DEVICES="${FREE_GPUS_CSV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export OMP_NUM_THREADS=4


echo "--- 3. Start DDP analysis with ${NUM_FOUND_GPUS} GPUs ---"


${CONDA_ENV_PATH}/bin/torchrun \
    --nproc_per_node="${NUM_FOUND_GPUS}" \
    --master_port=29998 \
    "${SCRIPT}" \
    --ckpt_glob "${CHECKPOINT_PATHS}" \
    --out_dir "${OUTPUT_DIR}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET}" \
    --data_name "${DATA_NAME}" \
    --split "${SPLIT}" \
    --local_batch 8 \
    --seq_len 2048


unset CUDA_VISIBLE_DEVICES

echo "Finish"
