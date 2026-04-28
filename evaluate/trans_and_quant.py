"""
Transform the models to HF form and quant them by bnb
"""
import torch
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from training_tools import load_model_from_checkpoint
from modeling_vscale import VscaleForCausalLM, VscaleModel

# register for AutoModel
VscaleForCausalLM.register_for_auto_class("AutoModelForCausalLM")
VscaleModel.register_for_auto_class("AutoModel")

tokenizer_path = "path/to/your/GPT2-tokenizer"

ckpt_path = "train/runs/baseline_1B/checkpoints/step_25000.pt"
name = "1B_baseline"

model_path = "evaluate/models/base/" + name
quant_path = "evaluate/models/bnb/" + name + "_bnb"



def safe_convert_to_mixed_bf16(model: torch.nn.Module):
    """
    Mixed precision.
    Keep VRadialScale.theta FP32
    """
    theta_backup = {}

    for name, module in model.named_modules():
        if type(module).__name__ == "VRadialScale" and hasattr(module, "theta") and module.theta is not None:
            param_name = name + ".theta"
            theta_backup[param_name] = module.theta.detach().clone().float().cpu()
            print(f"🛡️  detected [{param_name}], backup FP32")

    # to BF16
    print("⏳ transform to BF16...")
    model.to(torch.bfloat16)

    for name, param in model.named_parameters():
        if name in theta_backup:
            param.data = theta_backup[name].to(device=param.device, dtype=torch.float32)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=torch.float32)
            print(f"✅ recover [{name}] to {param.dtype}")

    bad = []
    for name, param in model.named_parameters():
        if name.endswith("v_radial.theta") and param.dtype != torch.float32:
            bad.append((name, param.dtype))

    if bad:
        raise RuntimeError(f"{bad}")

    return model


# ======================== main ========================

print("Loading checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model, config, _ = load_model_from_checkpoint(ckpt_path)
model.eval()

os.makedirs(model_path, exist_ok=True)
os.makedirs(quant_path, exist_ok=True)

print("\n--- transform HF weights ---")
model = safe_convert_to_mixed_bf16(model)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

config = AutoConfig.from_pretrained(model_path)
try:
    if config.auto_map["AutoModelForCausalLM"] == "modeling_vscale.VscaleForCausalLM":
        config.auto_map["AutoModel"] = "modeling_vscale.VscaleModel"
        # copy .py
        shutil.copyfile(os.path.join(model_path, "modeling_vscale.py"),
                        os.path.join(quant_path, "modeling_vscale.py"))
        config.save_pretrained(model_path)
except:
    pass

print(f"--- BF16 model and Tokenizer are saved to {model_path} ---\n")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Start BNB NF4 transform...")
q_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

q_model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"✅ BNB model saved to {quant_path}")
