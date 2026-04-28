import os
import re
import csv
import time
import inspect
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from transformers import LlamaConfig, LlamaForCausalLM

try:
    from torch.serialization import safe_globals
except Exception:
    safe_globals = None


# -------------------------
# Checkpoint loader
# -------------------------
def load_checkpoint_obj(checkpoint_path: str) -> Dict:
    if safe_globals is not None:
        with safe_globals([LlamaConfig]):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    print(f"Loaded checkpoint with keys {ckpt.keys()}")
    # ['model', 'optimizer', 'scaler', 'train_state', 'args', 'rng_state']
    if not isinstance(ckpt, dict):
        # raw state_dict
        ckpt = {"model": ckpt}
    return ckpt


def fix_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    new_state_dict = OrderedDict()
    prefix_to_remove = "_orig_mod."
    for k, v in state_dict.items():
        if k.startswith(prefix_to_remove):
            nk = k[len(prefix_to_remove):]
        else:
            nk = k
        new_state_dict[nk] = v
    return new_state_dict


def load_model_from_checkpoint(checkpoint_path: str, attn_implement:str="eager", ntk_factor=1) -> Tuple[LlamaForCausalLM, LlamaConfig, int]:
    ckpt = load_checkpoint_obj(checkpoint_path)
    state_dict = ckpt.get("model", ckpt)
    state_dict = fix_state_dict_keys(state_dict)

    config = ckpt.get("args", None)

    signature = inspect.signature(LlamaConfig.__init__)
    valid_keys = set(signature.parameters.keys())
    raw_args = dict(config) 
    filtered_args = {k: v for k, v in raw_args.items() if k in valid_keys and k != 'self'}
    config = LlamaConfig(**filtered_args)
    config.model_type = raw_args["model_type"]  # Must do this

    # NTK-aware extend
    if ntk_factor != 1:
        config.rope_scaling = {
            "type": "dynamic",      # or "linear"
            "factor": ntk_factor    # for example 2048 -> 4096, the factor=2.0
        }
    
    config._attn_implementation = attn_implement

    step = int(ckpt.get("train_state", {}).get("step", 0))
    modeltype = config.model_type
    if modeltype.lower() == "vscale":
        from modeling_vscale import VscaleForCausalLM
        config.v_scale_learnable = raw_args.get("v_scale_learnable", False)
        model = VscaleForCausalLM(config)
        print(f"model_type={config.model_type}, initialize VscaleForCausalLM")
    else:
        model = LlamaForCausalLM(config)
        print(f"model_type={config.model_type}, initialize LlamaForCausalLM")
    model.load_state_dict(state_dict, strict=True)

    return model, config, step
