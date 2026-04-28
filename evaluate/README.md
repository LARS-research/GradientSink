# Evaluation Guide

This folder analyzes trained checkpoints and runs downstream evaluations. The recommended path is to use the launch scripts instead of typing long `torchrun` commands by hand.

With the default training launcher, checkpoints are expected at:

```text
train/runs/vscale_1B/checkpoints/step_*.pt
```

## Main Launchers

```text
0_ckpt_grads_check_launch.sh      gradient statistics for checkpoints
1_extreme_token_probe_launch.sh   extreme-token / sink metrics and plots
niah_lm_eval_launch.sh            NIAH long-context evaluation
```

Run all launchers from the repository root.

## 1. Extreme-Token Probe

Use this first for checkpoint-level sink/extreme-token analysis.

Open `evaluate/1_extreme_token_probe_launch.sh` and edit:

```bash
CONDA_ENV_PATH="path/to/your/env"
CKPT_GLOB="train/runs/vscale_1B/checkpoints/step_*.pt"
OUT_DIR="evaluate/outputs/extreme_token_probe"
TAG="vscale_1B"
TOKENIZER_PATH="path/to/your/GPT2-tokenizer"
DATASET_PATH="path/to/your/datasets--allenai--c4/snapshots"
NUM_GPUS_TO_USE=1
```

Then run:

```bash
chmod +x evaluate/1_extreme_token_probe_launch.sh
bash evaluate/1_extreme_token_probe_launch.sh
```

Expected output:

```text
evaluate/outputs/extreme_token_probe/vscale_1B/
  probe_cache/
  ckpts/
  plots/
```

To only regenerate plots, set this in the launcher:

```bash
ONLY_PLOT="true"
DO_PLOTS="true"
```

## 2. Gradient Probe

Open `evaluate/0_ckpt_grads_check_launch.sh` and edit:

```bash
CONDA_ENV_PATH="path/to/your/env"
CHECKPOINT_PATHS="train/runs/vscale_1B/checkpoints/step_*.pt"
OUTPUT_DIR="evaluate/ckpt_grads"
TOKENIZER_PATH="path/to/your/GPT2-tokenizer"
DATASET="path/to/your/datasets--allenai--c4/snapshots"
NUM_GPUS_TO_USE=4
```

Then run:

```bash
chmod +x evaluate/0_ckpt_grads_check_launch.sh
bash evaluate/0_ckpt_grads_check_launch.sh
```

Expected output:

```text
evaluate/ckpt_grads/<timestamp>/
  probe_cache/
  ckpts/
  summary.csv
```

## 3. NIAH Evaluation

First convert checkpoints to Hugging Face model folders if needed. Then open `evaluate/niah_lm_eval_launch.sh` and edit:

```bash
CONDA_ENV_PATH="path/to/your/env"
MODEL_PATHS="evaluate/models/base/vscale_1B"
OUTPUT_PATH="evaluate/outputs/lm_eval_outputs"
DEVICE="cuda:0"
TASKS="niah_single_2,niah_single_3,niah_multikey_1"
MAX_LENGTHS="4096"
SEEDS="0,1,2,3,4"
```

Then run:

```bash
chmod +x evaluate/niah_lm_eval_launch.sh
bash evaluate/niah_lm_eval_launch.sh
```

## Other Scripts

These scripts do not have launchers yet. Edit their top-level constants before running them:

```text
trans_and_quant.py          convert a train checkpoint to HF format, then BNB NF4
quantization.py             AWQ/GPTQ/SmoothQuant/W8A8 via llm-compressor
lmeval_batch_test.py        lm-eval with HF backend
lmeval_batch_test_vllm.py   lm-eval with vLLM backend
see_vscale_theta.py         plot V-scale theta / effective C over checkpoints
niah_summerize.py           aggregate NIAH JSON outputs
```

## Common Fixes

- If GPU selection fails, lower `FREE_MEMORY_THRESHOLD` or reduce `NUM_GPUS_TO_USE`.
- If CUDA OOM occurs, reduce `LOCAL_BATCH`, `GLOBAL_BATCH`, or sequence length in the launcher.
- If V-scale HF model loading fails, use `trust_remote_code=True`.
