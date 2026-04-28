#!/bin/bash
set -euo pipefail

# =========================================================
# 1. User config
# =========================================================
NUM_GPUS_TO_USE=1

# SCRIPT="evaluate/niah_lm_eval.py"
SCRIPT="evaluate/niah_lm_eval_vllm.py"

CONDA_ENV_PATH="path/to/your/env"
PYTHON="${CONDA_ENV_PATH}/bin/python"

MODEL_PATHS="evaluate/models/base/1B_baseline,evaluate/models/bnb/1B_baseline_bnb,evaluate/models/smoothquant/1B_baseline_smoothquant,evaluate/models/gptq/1B_baseline_gptq,evaluate/models/awq/1B_baseline_awq,evaluate/models/base/1B_vscale,evaluate/models/bnb/1B_vscale_bnb,evaluate/models/smoothquant/1B_vscale_smoothquant,evaluate/models/gptq/1B_vscale_gptq,evaluate/models/awq/1B_vscale_awq"

ORIGIN_MAX_LENGTH=2048
OUTPUT_PATH="evaluate/outputs/lm_eval_outputs/"


TASKS="niah_single_2,niah_single_3,niah_multikey_1"
SEEDS="0,1,2,3,4"



# =========================================================
# 2. Detect free GPUs
# =========================================================
FREE_MEMORY_THRESHOLD=75000
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# =========================================================
# 3. Build args
# =========================================================
ARGS=(
    --model-paths "${MODEL_PATHS}"
    --origin-max-length "${ORIGIN_MAX_LENGTH}"
    --output-path "${OUTPUT_PATH}"
    --tasks "${TASKS}"
    --seeds "${SEEDS}"
    --device "cuda:0"
)

#   max_lengths | yarn_factor | batch_size
LENGTH_CONFIGS=(
    "1024,2048|1|32"
    "1024,2048,4096,8192|4|32"
)

# =========================================================
# 4. Launch
# =========================================================
echo "--- 2. Launching with ${NUM_GPUS_TO_USE} processes ---"
for CONFIG in "${LENGTH_CONFIGS[@]}"; do
    IFS='|' read -r MAX_LENGTHS YARN_FACTOR BATCH_SIZE <<< "${CONFIG}"

    "${PYTHON}" "${SCRIPT}" \
        "${ARGS[@]}" \
        --max-lengths "${MAX_LENGTHS}" \
        --yarn-factor "${YARN_FACTOR}" \
        --batch-size "${BATCH_SIZE}"
    echo

    sleep 5
done

echo "Finished."
