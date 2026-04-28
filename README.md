# Gradient Sink / V-scale Experiment Code

This repository contains the training and evaluation code for the V-scale experiments in the submission. It is meant to be used as a reproducibility package: train Llama-style baseline/V-scale models, inspect checkpoints, convert checkpoints to Hugging Face format, quantize models, and run downstream evaluations.

## Start Here

If you are new to this codebase, follow this order:

1. Prepare a GPU machine and install the environment.
2. Choose a tokenizer and a text dataset.
3. Edit and run the training launcher.
4. Run the baseline and V-scale training jobs.
5. Run checkpoint probes on the saved checkpoints.
6. Convert checkpoints to Hugging Face model folders.
7. Run lm-eval, NIAH, and quantization experiments.

The launcher workflow for steps 3 and 4 is in `train/README.md`. The launcher workflow for steps 5 to 7 is in `evaluate/README.md`.

## What Is Included

```text
.
  train/
    train.py                 # main training script
    modeling_vscale.py       # V-scale Llama implementation
    launch_train.sh          # training launcher with placeholders
  evaluate/
    0_ckpt_grads_check.py    # gradient probe for training checkpoints
    1_extreme_token_probe.py # extreme-token / sink probe
    trans_and_quant.py       # checkpoint -> HF model -> BNB quantized model
    quantization.py          # AWQ/GPTQ/SmoothQuant/W8A8 recipes
    lmeval_*.py              # lm-eval runners
    niah_*.py                # NIAH runners
  environment.yml            # conda environment snapshot
  requirements.txt           # pip package snapshot
```

Generated files such as checkpoints, TensorBoard logs, plots, converted models, and quantized models are not included.

## What You Need To Provide

You need these external resources:

- A CUDA GPU machine. Training and checkpoint-gradient probes require CUDA.
- A tokenizer. For the default settings, GPT-2 style tokenization is expected.
- A dataset with a `text` column. C4 is the intended default.
- Checkpoints produced by `train/train.py` if you want to run checkpoint probes.
- Hugging Face-format model directories if you want to run lm-eval, NIAH, or quantization scripts directly.

Example tokenizer choices:

```text
gpt2
/path/to/local/GPT2-tokenizer
```

Example dataset choices:

```text
allenai/c4
/path/to/local/c4_snapshot
```

When using C4, keep `--data_name en`.

## Install

Option A: create the full conda environment from the snapshot.

```bash
conda env create -f environment.yml
conda activate LLM
```

Option B: create a fresh environment and install the pip snapshot.

```bash
conda create -n gradientsink python=3.12 -y
conda activate gradientsink
pip install -r requirements.txt
```

CUDA-related packages are machine-specific. If installation fails around `torch`, `flash-attn`, or `vllm`, install the matching build for your CUDA driver first, then retry the remaining packages.

## Workflow

Start with the training launcher:

```bash
bash train/launch_train.sh
```

Then run the evaluation launchers:

```bash
bash evaluate/1_extreme_token_probe_launch.sh
bash evaluate/0_ckpt_grads_check_launch.sh
```

Edit the config block at the top of each launcher before running it. The two subfolder READMEs explain exactly which variables to change.
