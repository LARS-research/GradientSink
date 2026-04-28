#!/bin/bash
set -euo pipefail

# =========================================================
# 1. User config
# =========================================================

# ---- GPU selection ----
FREE_MEMORY_THRESHOLD=75000     # MB
NUM_GPUS_TO_USE=4

# ---- Paths ----
TRAIN_SCRIPT="train/train.py"
CONDA_ENV_PATH="path/to/your/env"

TOKENIZER_PATH="path/to/your/GPT2-tokenizer"
DATASET_PATH="path/to/your/datasets--allenai--c4/snapshots"
OUT_DIR="train/runs"

# ---- Resume ----
# Leave empty to train from scratch, or set to a step_xxxxx.pt checkpoint to resume.
RESUME_CKPT=""

# ---- Run identity ----
RUN_NAME="$(date +%Y%m%d_%H%M%S)"

# ---- Model selection ----
MODEL_TYPE="vscale"             # llama or vscale
V_SCALE_LEARNABLE="true"        # only meaningful when MODEL_TYPE=vscale

# ---- Training hyperparameters ----
SEED=42
GLOBAL_BATCH_SIZE=512
LOCAL_BATCH_SIZE=128
MAX_ITERS=25000
WARMUP_ITERS=1000
MAX_LR=1e-3
MIN_LR=1e-4
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# ---- Model config ----
VOCAB_SIZE=50257
HIDDEN_SIZE=2048
INTERMEDIATE_SIZE=5504
NUM_HIDDEN_LAYERS=18
NUM_ATTENTION_HEADS=16
NUM_KEY_VALUE_HEADS=8
HEAD_DIM=128
MAX_POSITION_EMBEDDINGS=2048
RMS_NORM_EPS=1e-5
BOS_TOKEN_ID=50256
EOS_TOKEN_ID=50256

ATTENTION_BIAS="false"
MLP_BIAS="false"
TIE_WORD_EMBEDDINGS="false"

# ---- Runtime / logging ----
AMP_MODE="auto"                 # auto / bf16 / fp32
ATTN_IMPL="flash_attention_2"
COMPILE="true"                  # true / false
COMPILE_MODE="default"
LOG_INTERVAL=1
VALID_INTERVAL=200
VALID_MAX_BATCHES=32
SAVE_INTERVAL=1000
FULL_SAVE_INTERVAL=5000
NUM_WORKERS=0
OMP_THREADS=4

# ---- Data options ----
DATA_NAME="en"
TRAIN_SPLIT="train"
VALID_SPLIT="validation"
SHUFFLE_BUFFER_SIZE=100000
TOKENIZE_BATCH_SIZE=1024
ADD_BOS="false"
ADD_EOS="true"

# ---- Optimizer grouping ----
# weight decay on 2D parameters only
NO_DECAY_KEYWORDS="bias,norm,rmsnorm,layernorm,vscale,v_scale,theta,lambda"

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

# Restrict visible GPUs; inside torchrun they become cuda:0,1,2,...
export CUDA_VISIBLE_DEVICES="${FREE_GPUS_CSV}"
export OMP_NUM_THREADS="${OMP_THREADS}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# =========================================================
# 3. Build training args
# =========================================================
TRAIN_ARGS=(
    --run_name "${RUN_NAME}"
    --out_dir "${OUT_DIR}"

    --amp "${AMP_MODE}"
    --attn_implementation "${ATTN_IMPL}"
    --compile "${COMPILE}"
    --compile_mode "${COMPILE_MODE}"

    --model_type "${MODEL_TYPE}"
    --v_scale_learnable "${V_SCALE_LEARNABLE}"

    --vocab_size "${VOCAB_SIZE}"
    --hidden_size "${HIDDEN_SIZE}"
    --intermediate_size "${INTERMEDIATE_SIZE}"
    --num_hidden_layers "${NUM_HIDDEN_LAYERS}"
    --num_attention_heads "${NUM_ATTENTION_HEADS}"
    --num_key_value_heads "${NUM_KEY_VALUE_HEADS}"
    --head_dim "${HEAD_DIM}"
    --max_position_embeddings "${MAX_POSITION_EMBEDDINGS}"
    --rms_norm_eps "${RMS_NORM_EPS}"
    --bos_token_id "${BOS_TOKEN_ID}"
    --eos_token_id "${EOS_TOKEN_ID}"
    --attention_bias "${ATTENTION_BIAS}"
    --mlp_bias "${MLP_BIAS}"
    --tie_word_embeddings "${TIE_WORD_EMBEDDINGS}"

    --tokenizer_path "${TOKENIZER_PATH}"
    --train_data_path "${DATASET_PATH}"
    --valid_data_path "${DATASET_PATH}"
    --data_name "${DATA_NAME}"
    --train_split "${TRAIN_SPLIT}"
    --valid_split "${VALID_SPLIT}"
    --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
    --tokenize_batch_size "${TOKENIZE_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --add_bos "${ADD_BOS}"
    --add_eos "${ADD_EOS}"

    --seed "${SEED}"
    --global_batch_size "${GLOBAL_BATCH_SIZE}"
    --local_batch_size "${LOCAL_BATCH_SIZE}"
    --max_iters "${MAX_ITERS}"
    --warmup_iters "${WARMUP_ITERS}"
    --max_lr "${MAX_LR}"
    --min_lr "${MIN_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_clip "${GRAD_CLIP}"

    --log_interval "${LOG_INTERVAL}"
    --valid_interval "${VALID_INTERVAL}"
    --valid_max_batches "${VALID_MAX_BATCHES}"
    --save_interval "${SAVE_INTERVAL}"
    --full_save_interval "${FULL_SAVE_INTERVAL}"

    --no_decay_keywords "${NO_DECAY_KEYWORDS}"
)

if [ -n "${RESUME_CKPT}" ]; then
    TRAIN_ARGS+=( --resume "${RESUME_CKPT}" )
fi

# =========================================================
# 4. Launch
# =========================================================
echo "--- 2. Launching DDP training with ${NUM_GPUS_TO_USE} processes ---"
echo "Run name: ${RUN_NAME}"
echo "Model type: ${MODEL_TYPE}"
if [ -n "${RESUME_CKPT}" ]; then
    echo "Resume from: ${RESUME_CKPT}"
else
    echo "Resume from: <none>"
fi

"${CONDA_ENV_PATH}/bin/torchrun" \
    --nproc_per_node="${NUM_GPUS_TO_USE}" \
    --master_port=29999 \
    "${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}"

unset CUDA_VISIBLE_DEVICES

echo "Training finished."
