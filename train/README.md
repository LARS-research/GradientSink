# Training Guide

This folder trains the baseline Llama model or the V-scale model. For reproducibility, use the launcher:

```bash
bash train/launch_train.sh
```

## What To Edit

Open `train/launch_train.sh` and edit the first config block.

Required paths:

```bash
CONDA_ENV_PATH="path/to/your/env"
TOKENIZER_PATH="path/to/your/GPT2-tokenizer"
DATASET_PATH="path/to/your/datasets--allenai--c4/snapshots"
OUT_DIR="train/runs"
```

## Choose The Model

For V-scale, keep:

```bash
MODEL_TYPE="vscale"
V_SCALE_LEARNABLE="true"
```

For the baseline, use:

```bash
MODEL_TYPE="llama"
V_SCALE_LEARNABLE="true"
```

`V_SCALE_LEARNABLE` is ignored by the baseline model.

## Choose GPUs

The launcher automatically picks idle GPUs using GPUtil:

```bash
FREE_MEMORY_THRESHOLD=75000
NUM_GPUS_TO_USE=4
```

Lower `FREE_MEMORY_THRESHOLD` if your GPUs have less free memory. Change `NUM_GPUS_TO_USE` to match the number of GPUs you want.

## Choose Model Size

The default launcher is set up for a 1B-style V-scale run:

```bash
HIDDEN_SIZE=2048
INTERMEDIATE_SIZE=5504
NUM_HIDDEN_LAYERS=18
NUM_ATTENTION_HEADS=16
NUM_KEY_VALUE_HEADS=8
HEAD_DIM=128
MAX_POSITION_EMBEDDINGS=2048
```

Important: `HIDDEN_SIZE` must equal `NUM_ATTENTION_HEADS * HEAD_DIM`.

## Choose Training Length

The main knobs are:

```bash
GLOBAL_BATCH_SIZE=512
LOCAL_BATCH_SIZE=128
MAX_ITERS=25000
WARMUP_ITERS=1000
MAX_LR=1e-3
MIN_LR=1e-4
SAVE_INTERVAL=1000
FULL_SAVE_INTERVAL=5000
```

Important: `GLOBAL_BATCH_SIZE` must be divisible by `LOCAL_BATCH_SIZE * NUM_GPUS_TO_USE`.

## Run

From the repository root:

```bash
chmod +x train/launch_train.sh
bash train/launch_train.sh
```

Expected output:

```text
train/runs/vscale_1B/
  config.json
  train_log.jsonl
  tb/
  checkpoints/
    step_01000.pt
    step_02000.pt
    ...
```

## Resume

Set `RESUME_CKPT` in `train/launch_train.sh`:

```bash
RESUME_CKPT="train/runs/vscale_1B/checkpoints/step_05000.pt"
```

Then run the launcher again:

```bash
bash train/launch_train.sh
```

Exact resume requires a checkpoint that includes optimizer state. Full optimizer checkpoints are controlled by `FULL_SAVE_INTERVAL`.

## Notes

- Use `ATTN_IMPL="flash_attention_2"` for real training when FlashAttention is installed.
