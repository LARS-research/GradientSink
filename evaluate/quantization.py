import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import gc
import torch
import random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs

NUM_CALIBRATION_SAMPLES=512 # 256 or 512
MAX_SEQUENCE_LENGTH=2048    # Change this

model_name = "03B_vscale"
model_path = "evaluate/models/base/" + model_name


method = "smoothquant" # awq / gptq / smoothquant / w8a8 / w8a8_act / w8a8_stress / w8a8_down_only
output_dir = f"evaluate/models/{method}/{model_name}_{method}" 
tokenizer = AutoTokenizer.from_pretrained(model_path)


random.seed(42)
torch.manual_seed(42)

dataset = "path/to/your/datasets--allenai--c4/snapshots"
dataset_name = "en"
split = "validation"


# --- data ---
def preprocess_calibration_data(dataset, tokenizer, num_samples, max_length):
    samples = []
    count = 0
    for entry in dataset:
        if count >= num_samples:
            break
        tokenized = tokenizer(
            entry["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        # only keep samples long enough
        if len(tokenized["input_ids"]) > max_length // 2:
            samples.append({
                "input_ids": tokenized["input_ids"],
            })
            count += 1
    return Dataset.from_list(samples)

raw_data = load_dataset(dataset, dataset_name, split=split, streaming=True)
calib_list = preprocess_calibration_data(raw_data, tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH)


if method == "awq":
    # --- AWQ ---
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": False,     # ASYM
                        "strategy": "group",
                        "group_size": 128,
                    },
                }
            },
        ),
    ]
elif method == "gptq":
    # --- GPTQ ---
    recipe = [
        GPTQModifier(
            ignore=["lm_head"],
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": False,
                        "strategy": "group",
                        "group_size": 128,
                    }
                }
            },
        ),
    ]
elif method == "smoothquant":
    recipe = [
        SmoothQuantModifier(
            smoothing_strength=0.8,
            # mappings=[
            #     [
            #         ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            #         "re:.*input_layernorm",
            #     ],
            #     [
            #         ["re:.*gate_proj", "re:.*up_proj"],
            #         "re:.*post_attention_layernorm",
            #     ],
            # ],
        ),
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=["lm_head"],
        ),
    ]
elif method == "w8a8":
    int8_weight_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    }

    int8_act_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    }

    recipe = [
        QuantizationModifier(
            config_groups={
                "linear_w8a8": {
                    "targets": ["Linear"],
                    "weights": int8_weight_args,
                    "input_activations": int8_act_args,
                }
            },
            ignore=["lm_head"],
        )
    ]
elif method == "w8a8_act":
    weight_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    }

    activation_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }

    recipe = [
        QuantizationModifier(
            config_groups={
                "linear_w8a8": {
                    "targets": ["Linear"],
                    "weights": weight_args,
                    "input_activations": activation_args,
                }
            },
            ignore=["lm_head"],
        )
    ]
elif method == "w8a8_stress":
    int8_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }
    recipe = [
        QuantizationModifier(
            config_groups={
                "linear_w8a8": {
                    "targets": ["Linear"],
                    "weights": int8_args,
                    "input_activations": int8_args,
                }
            },
            ignore=["lm_head"],
        )
    ]
elif method == "w8a8_down_only":
    weight_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    }

    activation_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }

    recipe = [
        QuantizationModifier(
            config_groups={
                "down_proj_a8_static": {
                    "targets": [r"re:.*mlp\.down_proj$"],
                    "weights": weight_args,
                    "input_activations": activation_args,
                }
            },
            ignore=[],
        )
    ]
else:
    raise ValueError("unknown quantization method")

print(f"{method.upper()} quantization start ...")

gc.collect()
torch.cuda.empty_cache()

try:
    oneshot(
        model=model_path,
        dataset=calib_list,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        trust_remote_code_model=True,
        precision="auto",
        batch_size=1,
    )
    print(f"✅ Success! model saved to {output_dir}")
except Exception as e:
    print(f"❌ Failed, error: {e}")
