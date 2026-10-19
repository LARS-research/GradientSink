#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified Extreme-Token probe.

What it keeps:
- activation magnitude metrics
- attention / sink metrics
- optional stable-rank metrics

What it removes:
- gradient-related metrics
- anisotropy
- dark / spectral / entropy metrics
- other auxiliary analyses

Checkpoint loading is aligned with the NEW train.py save format:
    payload = {
        "model": state_dict,
        "optimizer": ...,
        "scaler": ...,
        "train_state": {"step": ..., ...},
        "args": vars(args),
    }

Torchrun / DDP behavior:
- ranks split checkpoints by index
- all ranks share one cached probe batch tensor for comparability
- each checkpoint writes its own metrics.pt + meta.json
- rank0 optionally plots
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

try:
    from modeling_vscale_mode import VscaleForCausalLM
except Exception:
    VscaleForCausalLM = None

from draw_helper import plot_ckpt_lines_colorbar, plot_ckpt_group_lines

# =============================
# distributed helpers
# =============================

def init_distributed() -> Tuple[int, int, int, bool, bool, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    is_main = rank == 0

    if torch.cuda.is_available():
        if is_ddp:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        if is_ddp:
            dist.init_process_group(backend="gloo")

    return world_size, rank, local_rank, is_ddp, is_main, device


def barrier_if_needed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.barrier()


def destroy_distributed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# =============================
# basic utilities
# =============================

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {x}")


def parse_float_list(s: str) -> List[float]:
    if not s.strip():
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def get_autocast_dtype(dtype_name: str) -> Optional[torch.dtype]:
    dtype_name = dtype_name.lower()
    if dtype_name == "fp32":
        return None
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_name}")


def make_autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type != "cuda" or amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    raw = model.module if isinstance(model, DDP) else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def infer_step_from_path(path: str) -> int:
    m = re.search(r"step_(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return -1


def _get_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    raw = _unwrap_model(model)
    if hasattr(raw, "model") and hasattr(raw.model, "layers"):
        return list(raw.model.layers)
    if hasattr(raw, "layers"):
        return list(raw.layers)
    raise AttributeError("Cannot find transformer blocks.")


def _get_attn_module(block: torch.nn.Module) -> torch.nn.Module:
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(block, name):
            return getattr(block, name)
    raise AttributeError("Cannot find attention module in block.")


def _get_mlp_module(block: torch.nn.Module) -> torch.nn.Module:
    for name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(block, name):
            return getattr(block, name)
    raise AttributeError("Cannot find MLP module in block.")


# =============================
# tokenization / probe
# =============================

def tokenize_and_group(examples: Dict[str, List], tokenizer, block_size: int, add_bos: bool = False, add_eos: bool = True):
    texts = examples["text"]
    tokenized = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)

    all_tokens: List[int] = []
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    for ids in tokenized["input_ids"]:
        if add_bos and bos_id is not None:
            all_tokens.append(bos_id)
        all_tokens.extend(ids)
        if add_eos and eos_id is not None:
            all_tokens.append(eos_id)

    total_len = (len(all_tokens) // block_size) * block_size
    if total_len == 0:
        return {"input_ids": [], "labels": []}

    all_tokens = all_tokens[:total_len]
    blocks = [all_tokens[i : i + block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": blocks, "labels": [b.copy() for b in blocks]}


def collate_batch(examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([x["input_ids"] for x in examples], dtype=torch.long)
    labels = torch.tensor([x["labels"] for x in examples], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


def probe_cache_path(base_dir: str, T: int, local_b: int, accum: int, seed: int) -> str:
    return os.path.join(base_dir, f"probe.T{T}.localB{local_b}.acc{accum}.seed{seed}.pt")


@torch.no_grad()
def build_dataset_probe_shared(
    tokenizer_path: str,
    dataset_path: str,
    data_name: str,
    split: str,
    block_size: int,
    local_batch_size: int,
    accumulation_steps: int,
    seed: int,
    add_bos: bool,
    add_eos: bool,
    cache_path: str,
    overwrite_cache: bool,
    rank: int,
    is_ddp: bool,
):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if rank == 0 and overwrite_cache and os.path.exists(cache_path):
        os.remove(cache_path)
    barrier_if_needed(is_ddp)

    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    if rank != 0:
        barrier_if_needed(is_ddp)
        return torch.load(cache_path, map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.model_max_length = int(1e8)

    raw = load_dataset(dataset_path, data_name, split=split, streaming=True)
    raw = raw.shuffle(seed=seed, buffer_size=100_000)

    dataset = raw.map(
        tokenize_and_group,
        batched=True,
        batch_size=1024,
        remove_columns=raw.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "block_size": block_size,
            "add_bos": add_bos,
            "add_eos": add_eos,
        },
    )

    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    batches = []
    it = iter(loader)
    for _ in range(accumulation_steps):
        batch = next(it)
        batches.append(batch["input_ids"].contiguous())

    probe = torch.stack(batches, dim=0)  # [ACC, B, T]
    torch.save(probe, cache_path)
    barrier_if_needed(is_ddp)
    return probe


# =============================
# model / checkpoint loading
# =============================

def get_arg(saved_args: Dict, key: str, default):
    return saved_args[key] if key in saved_args else default


def build_model_from_saved_args(saved_args: Dict) -> torch.nn.Module:
    config = LlamaConfig(
        vocab_size=get_arg(saved_args, "vocab_size", 50257),
        hidden_size=get_arg(saved_args, "hidden_size", 1024),
        intermediate_size=get_arg(saved_args, "intermediate_size", 2816),
        num_hidden_layers=get_arg(saved_args, "num_hidden_layers", 16),
        num_attention_heads=get_arg(saved_args, "num_attention_heads", 8),
        num_key_value_heads=get_arg(saved_args, "num_key_value_heads", 4),
        max_position_embeddings=get_arg(saved_args, "max_position_embeddings", 2048),
        rms_norm_eps=get_arg(saved_args, "rms_norm_eps", 1e-5),
        bos_token_id=get_arg(saved_args, "bos_token_id", 50256),
        eos_token_id=get_arg(saved_args, "eos_token_id", 50256),
        head_dim=get_arg(saved_args, "head_dim", 128),
        attention_bias=get_arg(saved_args, "attention_bias", False),
        mlp_bias=get_arg(saved_args, "mlp_bias", False),
        tie_word_embeddings=get_arg(saved_args, "tie_word_embeddings", False),
        use_cache=False,
    )

    model_type = get_arg(saved_args, "model_type", "llama")
    if model_type in ["llama", "baseline"]:
        model = LlamaForCausalLM(config)
    elif model_type == "vscale":
        if VscaleForCausalLM is None:
            raise ImportError("modeling_vscale.VscaleForCausalLM is unavailable, but checkpoint requires vscale.")
        config.model_type = "vscale"
        config.v_scale_learnable = get_arg(saved_args, "v_scale_learnable", True)
        config.v_scale_mode = get_arg(saved_args, "v_scale_mode", "full")
        model = VscaleForCausalLM(config)
    else:
        raise ValueError(f"Unknown model_type in checkpoint args: {model_type}")

    model.config._attn_implementation = "eager"
    model.config.use_cache = False
    model.eval()
    return model


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """
    Strip wrapper prefixes introduced by compile / DDP style wrappers.

    Typical cases:
      _orig_mod.model.layers.0...
      module.model.layers.0...
      _orig_mod.module.model.layers.0...
    """
    if not state_dict:
        return state_dict

    def strip_prefix_once(k: str) -> str:
        for p in ("_orig_mod.", "module."):
            if k.startswith(p):
                return k[len(p):]
        return k

    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        # repeatedly strip, in case of nested wrappers
        while True:
            stripped = strip_prefix_once(nk)
            if stripped == nk:
                break
            nk = stripped
        new_sd[nk] = v
    return new_sd


@torch.no_grad()
def load_model_from_new_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, int, Dict]:
    payload = torch.load(ckpt_path, map_location="cpu")

    # new train.py format
    if isinstance(payload, dict) and "model" in payload:
        raw_state_dict = payload["model"]
        train_state = payload.get("train_state", {})
        saved_args = payload.get("args", {})
        step = int(train_state.get("step", infer_step_from_path(ckpt_path)))
    else:
        # backward compatibility for older checkpoints
        raw_state_dict = payload.get("model_state", payload)
        saved_args = payload.get("args", {})
        step = int(payload.get("iteration", infer_step_from_path(ckpt_path)))

    state_dict = _normalize_state_dict_keys(raw_state_dict)

    saved_args = payload.get("args", {})
    model = build_model_from_saved_args(saved_args)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # stricter sanity check: allow no unexpected keys after normalization
    if unexpected:
        raise RuntimeError(
            f"Unexpected key(s) after normalization when loading {ckpt_path}:\n"
            + "\n".join(unexpected[:50])
        )

    # missing keys are usually also a bug here; keep this strict unless you really want partial load
    if missing:
        raise RuntimeError(
            f"Missing key(s) when loading {ckpt_path}:\n"
            + "\n".join(missing[:50])
        )

    model.to(device)
    model.eval()

    train_state = payload.get("train_state", {}) or {}
    step = int(train_state.get("step", infer_step_from_path(ckpt_path)))
    return model, step, saved_args


# =============================
# metrics
# =============================

@torch.no_grad()
def compute_sink_metrics_prefix(attns: List[torch.Tensor], K: int, eps_list: List[float], eps_T: int) -> Dict[str, torch.Tensor]:
    """
    attns: list length L, each [B,H,T,T]
    outputs:
      sink_mass_layer_k [K,L]
      sink_win_layer_k  [K,L]
      sink_eps_layer_*_k [K,L]
    """
    L = len(attns)
    B, H, T, T2 = attns[0].shape
    assert T == T2
    K_eff = min(K, T)

    mass_layer = torch.zeros(K_eff, L, device=attns[0].device, dtype=torch.float32)
    win_layer = torch.zeros(K_eff, L, device=attns[0].device, dtype=torch.float32)
    sink_eps_layer = {eps: torch.zeros(K_eff, L, device=attns[0].device, dtype=torch.float32) for eps in eps_list}

    for l, A in enumerate(attns):
        winner = A.argmax(dim=-1)  # [B,H,T]
        for k in range(K_eff):
            col_vis = A[:, :, k:, k]  # [B,H,T-k]
            if col_vis.numel() == 0:
                continue
            mass_layer[k, l] = col_vis.mean(dtype=torch.float32)
            win_layer[k, l] = (winner[:, :, k:] == k).to(torch.float32).mean(dtype=torch.float32)

            end_t = min(k + eps_T, T)
            if end_t > k:
                alpha_h = A[:, :, k:end_t, k].mean(dim=2, dtype=torch.float32)  # [B,H]
                for eps in eps_list:
                    sink_eps_layer[eps][k, l] = (alpha_h > eps).to(torch.float32).mean(dtype=torch.float32)

    out = {
        "sink_mass_layer_k": mass_layer,
        "sink_win_layer_k": win_layer,
    }
    for eps in eps_list:
        out[f"sink_eps_layer_{str(eps).replace('.', 'p')}_k"] = sink_eps_layer[eps]
    return out


@torch.no_grad()
def matrix_stable_rank(weight: Optional[torch.Tensor], device: torch.device) -> float:
    if weight is None or weight.ndim < 2:
        return float("nan")
    try:
        W = weight.detach().to(device=device, dtype=torch.float32)
        s = torch.linalg.svdvals(W)
        fro2 = torch.sum(s * s)
        sigma_max_sq = s[0] * s[0]
        sr = fro2 / (sigma_max_sq + 1e-12)
        return float(sr.item())
    except Exception:
        return float("nan")


class ActivationRecorder:
    def __init__(self, model: torch.nn.Module, seq_len: int):
        self.model = model
        self.seq_len = int(seq_len)
        self.blocks = _get_transformer_blocks(model)
        self.L = len(self.blocks)
        self.device = next(_unwrap_model(model).parameters()).device

        def zeros_lt():
            return torch.zeros((self.L, self.seq_len), device=self.device, dtype=torch.float32)

        self.x_in_sum = zeros_lt()
        self.attn_sum = zeros_lt()
        self.x_half_sum = zeros_lt()
        self.mlp_sum = zeros_lt()
        self.x_out_sum = zeros_lt()
        self.v_sum = zeros_lt()

        self.x_in_cnt = torch.zeros((self.L,), device=self.device, dtype=torch.float32)
        self.attn_cnt = torch.zeros((self.L,), device=self.device, dtype=torch.float32)
        self.x_half_cnt = torch.zeros((self.L,), device=self.device, dtype=torch.float32)
        self.mlp_cnt = torch.zeros((self.L,), device=self.device, dtype=torch.float32)
        self.x_out_cnt = torch.zeros((self.L,), device=self.device, dtype=torch.float32)
        self.v_cnt = torch.zeros((self.L,), device=self.device, dtype=torch.float32)

        raw = _unwrap_model(model)
        config = getattr(raw, "config", None)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 1)))
        self.num_q_heads = int(getattr(config, "num_attention_heads", 1))
        self.hidden_size = int(getattr(config, "hidden_size", 0))
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // max(self.num_q_heads, 1)))

        self._tmp_x_in: Dict[int, torch.Tensor] = {}
        self._tmp_x_half: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._installed = False

    def reset(self):
        for t in [self.x_in_sum, self.attn_sum, self.x_half_sum, self.mlp_sum, self.x_out_sum, self.v_sum]:
            t.zero_()
        for c in [self.x_in_cnt, self.attn_cnt, self.x_half_cnt, self.mlp_cnt, self.x_out_cnt, self.v_cnt]:
            c.zero_()
        self._tmp_x_in = {}
        self._tmp_x_half = {}

    def close(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self._installed = False
        self._tmp_x_in = {}
        self._tmp_x_half = {}

    def install(self):
        if self._installed:
            return self

        for layer_idx, block in enumerate(self.blocks):
            attn = _get_attn_module(block)
            mlp = _get_mlp_module(block)

            def make_block_pre_hook(li: int):
                def pre_hook(module, inputs):
                    if not inputs:
                        return
                    X = inputs[0]
                    if not (torch.is_tensor(X) and X.dim() == 3):
                        return
                    B, T, _ = X.shape
                    T_eff = min(T, self.seq_len)
                    X = X[:, :T_eff, :].float()
                    self.x_in_sum[li, :T_eff] += X.norm(dim=-1).sum(dim=0, dtype=torch.float32)
                    self.x_in_cnt[li] += float(B)
                    self._tmp_x_in[li] = X.detach()
                return pre_hook

            def make_attn_hook(li: int):
                def hook(module, inputs, output):
                    Y = output[0] if isinstance(output, (tuple, list)) else output
                    if not (torch.is_tensor(Y) and Y.dim() == 3):
                        return
                    B, T, _ = Y.shape
                    T_eff = min(T, self.seq_len)
                    Y = Y[:, :T_eff, :].float()
                    self.attn_sum[li, :T_eff] += Y.norm(dim=-1).sum(dim=0, dtype=torch.float32)
                    self.attn_cnt[li] += float(B)

                    X = self._tmp_x_in.get(li, None)
                    if X is not None:
                        H = (X[:, :T_eff, :] + Y).float()
                        self.x_half_sum[li, :T_eff] += H.norm(dim=-1).sum(dim=0, dtype=torch.float32)
                        self.x_half_cnt[li] += float(B)
                        self._tmp_x_half[li] = H.detach()
                        del self._tmp_x_in[li]
                return hook

            def make_vproj_hook(li: int):
                def vhook(module, inputs, output):
                    V = output
                    if not (torch.is_tensor(V) and V.dim() == 3):
                        return
                    B, T, C = V.shape
                    T_eff = min(T, self.seq_len)
                    V = V[:, :T_eff, :].float()
                    Hkv = int(self.num_kv_heads)
                    Dhd = int(self.head_dim)
                    if Hkv * Dhd != C:
                        Hkv = 1
                        Dhd = C
                    Vh = V.view(B, T_eff, Hkv, Dhd).transpose(1, 2)  # [B,H,T,D]
                    vn = Vh.norm(dim=-1)  # [B,H,T]
                    self.v_sum[li, :T_eff] += vn.sum(dim=(0, 1), dtype=torch.float32)
                    self.v_cnt[li] += float(B * Hkv)
                return vhook

            def make_mlp_hook(li: int):
                def hook(module, inputs, output):
                    Z = output[0] if isinstance(output, (tuple, list)) else output
                    if not (torch.is_tensor(Z) and Z.dim() == 3):
                        return
                    B, T, _ = Z.shape
                    T_eff = min(T, self.seq_len)
                    Z = Z[:, :T_eff, :].float()
                    self.mlp_sum[li, :T_eff] += Z.norm(dim=-1).sum(dim=0, dtype=torch.float32)
                    self.mlp_cnt[li] += float(B)

                    X = self._tmp_x_half.get(li, None)
                    if X is not None:
                        O = (X[:, :T_eff, :] + Z).float()
                        self.x_out_sum[li, :T_eff] += O.norm(dim=-1).sum(dim=0, dtype=torch.float32)
                        self.x_out_cnt[li] += float(B)
                        del self._tmp_x_half[li]
                return hook

            self.handles.append(block.register_forward_pre_hook(make_block_pre_hook(layer_idx), with_kwargs=False))
            self.handles.append(attn.register_forward_hook(make_attn_hook(layer_idx)))
            if hasattr(attn, "v_proj") and torch.is_tensor(attn.v_proj.weight):
                self.handles.append(attn.v_proj.register_forward_hook(make_vproj_hook(layer_idx)))
            self.handles.append(mlp.register_forward_hook(make_mlp_hook(layer_idx)))

        self._installed = True
        return self

    @torch.no_grad()
    def finalize_cpu(self) -> Dict[str, torch.Tensor]:
        def mean(sum_lt: torch.Tensor, cnt_l: torch.Tensor) -> torch.Tensor:
            return (sum_lt / cnt_l.clamp_min(1.0).unsqueeze(1)).detach().cpu().float()

        return {
            "x_in_l2_mean_lt": mean(self.x_in_sum, self.x_in_cnt),
            "attn_l2_mean_lt": mean(self.attn_sum, self.attn_cnt),
            "x_half_l2_mean_lt": mean(self.x_half_sum, self.x_half_cnt),
            "mlp_l2_mean_lt": mean(self.mlp_sum, self.mlp_cnt),
            "x_out_l2_mean_lt": mean(self.x_out_sum, self.x_out_cnt),
            "v_l2_mean_lt": mean(self.v_sum, self.v_cnt),
        }


@torch.no_grad()
def collect_rank_metrics(model: torch.nn.Module, device: torch.device, rank_mats: str) -> Dict[str, torch.Tensor]:
    blocks = _get_transformer_blocks(model)
    L = len(blocks)

    keys = []
    if rank_mats == "qkv":
        keys = ["wq", "wk", "wv"]
    elif rank_mats == "qkvo":
        keys = ["wq", "wk", "wv", "wo"]
    else:
        keys = ["wq", "wk", "wv", "wo", "mlp_up", "mlp_gate", "mlp_down"]

    out = {f"sr_{k}_l": torch.full((L,), float("nan"), dtype=torch.float32) for k in keys}

    for l, block in enumerate(blocks):
        attn = _get_attn_module(block)
        mlp = _get_mlp_module(block)

        if "wq" in keys and hasattr(attn, "q_proj"):
            out["sr_wq_l"][l] = matrix_stable_rank(attn.q_proj.weight, device)
        if "wk" in keys and hasattr(attn, "k_proj"):
            out["sr_wk_l"][l] = matrix_stable_rank(attn.k_proj.weight, device)
        if "wv" in keys and hasattr(attn, "v_proj"):
            out["sr_wv_l"][l] = matrix_stable_rank(attn.v_proj.weight, device)
        if "wo" in keys and hasattr(attn, "o_proj"):
            out["sr_wo_l"][l] = matrix_stable_rank(attn.o_proj.weight, device)
        if "mlp_up" in keys and hasattr(mlp, "up_proj"):
            out["sr_mlp_up_l"][l] = matrix_stable_rank(mlp.up_proj.weight, device)
        if "mlp_gate" in keys and hasattr(mlp, "gate_proj"):
            out["sr_mlp_gate_l"][l] = matrix_stable_rank(mlp.gate_proj.weight, device)
        if "mlp_down" in keys and hasattr(mlp, "down_proj"):
            out["sr_mlp_down_l"][l] = matrix_stable_rank(mlp.down_proj.weight, device)

    return out


# =============================
# analysis per checkpoint
# =============================

@torch.no_grad()
def analyze_one_checkpoint(
    ckpt_path: str,
    probe: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    accumulation_steps: int,
    K: int,
    eps_list: List[float],
    eps_T: int,
    compute_rank: bool,
    rank_mats: str,
) -> Tuple[int, Dict[str, torch.Tensor], Dict]:
    model, step, saved_args = load_model_from_new_checkpoint(ckpt_path, device)
    model.config._attn_implementation = "eager"
    model.eval()

    recorder = ActivationRecorder(model=model, seq_len=probe.shape[-1]).install()
    recorder.reset()

    rank_metrics = collect_rank_metrics(model, device, rank_mats) if compute_rank else {}

    sink_sum: Dict[str, torch.Tensor] = {}

    for acc_i in range(accumulation_steps):
        input_ids = probe[acc_i].to(device, non_blocking=True)
        with make_autocast_context(device, amp_dtype):
            out = model(input_ids=input_ids, output_attentions=True, use_cache=False)

        attns = out.attentions
        if attns is None:
            raise RuntimeError("attentions is None. Ensure eager attention + output_attentions=True.")

        sink_metrics = compute_sink_metrics_prefix(attns, K=K, eps_list=eps_list, eps_T=eps_T)
        for name, value in sink_metrics.items():
            sink_sum[name] = value if name not in sink_sum else sink_sum[name] + value

        del out, attns, input_ids

    sink_avg = {k: (v / float(accumulation_steps)).detach().cpu().float() for k, v in sink_sum.items()}
    act_metrics = recorder.finalize_cpu()
    recorder.close()

    metrics: Dict[str, torch.Tensor] = {}
    metrics.update(sink_avg)
    metrics.update(act_metrics)
    metrics.update({k: v.detach().cpu().float() for k, v in rank_metrics.items()})

    meta = {
        "ckpt_path": ckpt_path,
        "ckpt_name": os.path.basename(ckpt_path),
        "step": int(step),
        "seq_len": int(probe.shape[-1]),
        "K": int(K),
        "accumulation_steps": int(accumulation_steps),
        "dtype": "fp32" if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
        "compute_rank": bool(compute_rank),
        "rank_mats": rank_mats,
        "model_type": get_arg(saved_args, "model_type", "llama"),
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return step, metrics, meta


# =============================
# plotting
# =============================

def get_sorted_checkpoint_results(run_dir: str) -> Tuple[List[str], List[int]]:
    ckpts_base = os.path.join(run_dir, "ckpts")
    if not os.path.isdir(ckpts_base):
        return [], []
    results = []
    for d in os.listdir(ckpts_base):
        full = os.path.join(ckpts_base, d)
        if not (d.startswith("ckpt_") and os.path.isdir(full)):
            continue
        m = re.search(r"ckpt_(\d+)", d)
        if m and os.path.exists(os.path.join(full, "metrics.pt")):
            results.append((int(m.group(1)), full))
    results.sort(key=lambda x: x[0])
    if not results:
        return [], []
    steps, dirs = zip(*results)
    return list(dirs), list(steps)


def _group_reduce_token_axis(mat_lt: torch.Tensor, group: str, token_max: int, rest_head: int) -> torch.Tensor:
    group = group.lower()
    L, T = mat_lt.shape
    Tm = min(int(token_max), int(T))
    if group.startswith("tok"):
        t = max(0, min(int(group[3:]), Tm - 1))
        return mat_lt[:, t]
    if Tm <= 1:
        return mat_lt[:, 0]
    rest = mat_lt[:, 1:Tm]
    if group == "mean_rest":
        return rest.mean(dim=1)
    if group == "max_rest":
        return rest[:, : min(rest.shape[1], int(rest_head))].amax(dim=1)
    raise ValueError(f"Unknown group: {group}")


def _group_reduce_key_axis(mat_kL: torch.Tensor, group: str, key_max: int, rest_head: int) -> torch.Tensor:
    group = group.lower()
    K, L = mat_kL.shape
    Km = min(int(key_max), int(K))
    if group.startswith("tok"):
        k = max(0, min(int(group[3:]), Km - 1))
        return mat_kL[k, :]
    if Km <= 1:
        return mat_kL[0, :]
    rest = mat_kL[1:Km, :]
    if group == "mean_rest":
        return rest.mean(dim=0)
    if group == "max_rest":
        return rest[: min(rest.shape[0], int(rest_head)), :].amax(dim=0)
    raise ValueError(f"Unknown group: {group}")


def do_plots(out_dir: str, ckpt_dirs_sorted: List[str], steps_sorted: List[int], K: int):
    if not ckpt_dirs_sorted:
        return
    
    groups = ("tok0", "mean_rest", "max_rest")
    group_style = {
        "tok0":      dict(dh=0.00, ds=0.0, dv=0.0, ls="-",  lw=1.70, alpha=1.00),
        "mean_rest": dict(dh=-0.18, ds=0.0, dv=0.0, ls="--", lw=1.20, alpha=0.86),
        "max_rest":  dict(dh=+0.18, ds=0.0, dv=0.0, ls=":",  lw=1.20, alpha=0.86),
    }
    group_label_map = {
        "tok0": "token 0",
        "mean_rest": "mean early rest",
        "max_rest": "max early rest",
    }

    metrics_all = [torch.load(os.path.join(d, "metrics.pt"), map_location="cpu") for d in ckpt_dirs_sorted]
    ref = metrics_all[-1]
    if "x_in_l2_mean_lt" not in ref:
        return

    L, T = ref["x_in_l2_mean_lt"].shape
    x = list(range(L))
    eps_keys = sorted([k for k in ref.keys() if k.startswith("sink_eps_layer_") and k.endswith("_k")])

    # tok0-only plots
    plot_ckpt_lines_colorbar(out_dir, x, [m["sink_mass_layer_k"][0].tolist() for m in metrics_all], steps_sorted,
                             "AS: sink mass (token 0)", "layer", "mass", "AS_sink_mass_tok0.png")
    plot_ckpt_lines_colorbar(out_dir, x, [m["sink_win_layer_k"][0].tolist() for m in metrics_all], steps_sorted,
                             "AS: sink win rate (token 0)", "layer", "win rate", "AS_sink_win_tok0.png")
    for key in eps_keys:
        plot_ckpt_lines_colorbar(out_dir, x, [m[key][0].tolist() for m in metrics_all], steps_sorted,
                                 f"AS: {key} (token 0)", "layer", "fraction", f"{key}_tok0.png")

    act_keys = [
        ("x_in_l2_mean_lt", "ACT_x_in_tok0.png", "||x_in||"),
        ("attn_l2_mean_lt", "ACT_attn_tok0.png", "||attn_out||"),
        ("x_half_l2_mean_lt", "ACT_x_half_tok0.png", "||x_half||"),
        ("mlp_l2_mean_lt", "ACT_mlp_tok0.png", "||mlp_out||"),
        ("x_out_l2_mean_lt", "ACT_x_out_tok0.png", "||x_out||"),
        ("v_l2_mean_lt", "ACT_v_tok0.png", "||v||"),
    ]
    for key, fname, ylabel in act_keys:
        if key in ref:
            plot_ckpt_lines_colorbar(out_dir, x, [m[key][:, 0].tolist() for m in metrics_all], steps_sorted,
                                     f"Activation: {ylabel} at token 0", "layer", ylabel, fname)

    # compact compare plots
    compare_groups = ("tok0", "mean_rest", "max_rest")
    rest_head = 15
    compare_dir = os.path.join(os.path.dirname(out_dir.rstrip("/")), "plots_compare")
    os.makedirs(compare_dir, exist_ok=True)

    as_mass_groups = {g: [] for g in compare_groups}
    as_win_groups = {g: [] for g in compare_groups}
    for m in metrics_all:
        for g in compare_groups:
            as_mass_groups[g].append(_group_reduce_key_axis(m["sink_mass_layer_k"], g, K, rest_head).tolist())
            as_win_groups[g].append(_group_reduce_key_axis(m["sink_win_layer_k"], g, K, rest_head).tolist())
    plot_ckpt_group_lines(compare_dir, x, as_mass_groups, steps_sorted, "AS mass: tok0 / mean_rest / max_rest", "mass", "COMPARE_AS_mass.png", cmap_name="Blues", group_style=group_style)
    plot_ckpt_group_lines(compare_dir, x, as_win_groups, steps_sorted, "AS win: tok0 / mean_rest / max_rest", "win rate", "COMPARE_AS_win.png", cmap_name="Blues", group_style=group_style)

    for key, fname, ylabel in [
        ("x_in_l2_mean_lt", "COMPARE_ACT_x_in.png", "||x_in||"),
        ("x_out_l2_mean_lt", "COMPARE_ACT_x_out.png", "||x_out||"),
        ("mlp_l2_mean_lt", "COMPARE_ACT_mlp.png", "||mlp_out||"),
        ("v_l2_mean_lt", "COMPARE_ACT_v.png", "||v||"),
    ]:
        if key not in ref:
            continue
        group_lines = {g: [] for g in compare_groups}
        for m in metrics_all:
            for g in compare_groups:
                group_lines[g].append(_group_reduce_token_axis(m[key], g, K, rest_head).tolist())
        plot_ckpt_group_lines(compare_dir, x, group_lines, steps_sorted, f"{ylabel}: tok0 / mean_rest / max_rest", ylabel, fname, cmap_name="Blues", group_style=group_style)

    # optional rank plots
    debug_dir = os.path.join(os.path.dirname(out_dir.rstrip("/")), "plots_rank")
    os.makedirs(debug_dir, exist_ok=True)
    for key in sorted([k for k in ref.keys() if k.startswith("sr_") and k.endswith("_l")]):
        plot_ckpt_lines_colorbar(debug_dir, x, [m[key].tolist() for m in metrics_all], steps_sorted,
                                 f"Stable rank: {key}", "layer", "stable rank", f"{key}.png", cmap_name="Oranges")


# =============================
# main
# =============================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_glob", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="")

    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--data_name", type=str, default="en")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--add_bos", type=str2bool, default=False)
    p.add_argument("--add_eos", type=str2bool, default=True)
    p.add_argument("--overwrite_probe_cache", type=str2bool, default=False)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--eval_seq_len", type=int, default=2048)
    p.add_argument("--global_batch", type=int, default=128)
    p.add_argument("--local_batch", type=int, default=8)

    p.add_argument("--K", type=int, default=16)
    p.add_argument("--eps_list", type=str, default="0.3")
    p.add_argument("--eps_T", type=int, default=64)

    p.add_argument("--compute_rank", type=str2bool, default=False)
    p.add_argument("--rank_mats", type=str, default="all", choices=["qkv", "qkvo", "all"])

    p.add_argument("--do_plots", type=str2bool, default=True)
    p.add_argument("--only_plot", type=str2bool, default=False)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    world_size, rank, local_rank, is_ddp, is_main, device = init_distributed()
    amp_dtype = get_autocast_dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.tag:
        tag = args.tag
    else:
        tag = time.strftime("%Y%m%d_%H%M%S") if is_main else ""
    if is_ddp:
        obj = [tag]
        dist.broadcast_object_list(obj, src=0)
        tag = obj[0]

    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    if args.only_plot:
        if is_main:
            ckpt_dirs_sorted, steps_sorted = get_sorted_checkpoint_results(run_dir)
            do_plots(os.path.join(run_dir, "plots"), ckpt_dirs_sorted, steps_sorted, args.K)
            print(f"[rank0] plots saved under {os.path.join(run_dir, 'plots')}")
        destroy_distributed(is_ddp)
        return

    if args.global_batch % args.local_batch != 0:
        raise ValueError("global_batch must be divisible by local_batch")
    accum = args.global_batch // args.local_batch
    eps_list = parse_float_list(args.eps_list)

    probe_dir = os.path.join(run_dir, "probe_cache")
    os.makedirs(probe_dir, exist_ok=True)
    cache_path = probe_cache_path(probe_dir, args.eval_seq_len, args.local_batch, accum, args.seed)
    probe = build_dataset_probe_shared(
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path,
        data_name=args.data_name,
        split=args.split,
        block_size=args.eval_seq_len,
        local_batch_size=args.local_batch,
        accumulation_steps=accum,
        seed=args.seed,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        cache_path=cache_path,
        overwrite_cache=args.overwrite_probe_cache,
        rank=rank,
        is_ddp=is_ddp,
    )
    print(f"[rank {rank}] probe dataset built")

    barrier_if_needed(is_ddp)

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpt_glob}")

    def sort_key(p: str) -> int:
        return infer_step_from_path(p)

    ckpts = sorted(ckpts, key=sort_key)
    ckpts_local = ckpts[rank::world_size]

    if is_main:
        print(f"[rank0] total checkpoints: {len(ckpts)} | world_size={world_size}")
        print(f"[rank0] run_dir={run_dir}")
    print(f"[rank {rank}] evaluating {len(ckpts_local)} checkpoints on {device}")

    for ckpt_path in ckpts_local:
        step, metrics, meta = analyze_one_checkpoint(
            ckpt_path=ckpt_path,
            probe=probe,
            device=device,
            amp_dtype=amp_dtype,
            accumulation_steps=accum,
            K=args.K,
            eps_list=eps_list,
            eps_T=args.eps_T,
            compute_rank=args.compute_rank,
            rank_mats=args.rank_mats,
        )

        ckpt_dir = os.path.join(run_dir, "ckpts", f"ckpt_{step:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(metrics, os.path.join(ckpt_dir, "metrics.pt"))
        with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[rank {rank}] step={step} -> saved {ckpt_dir}")

    barrier_if_needed(is_ddp)

    if is_main and args.do_plots:
        ckpt_dirs_sorted, steps_sorted = get_sorted_checkpoint_results(run_dir)
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        do_plots(plots_dir, ckpt_dirs_sorted, steps_sorted, args.K)
        print(f"[rank0] plots saved under {plots_dir}")

    destroy_distributed(is_ddp)


if __name__ == "__main__":
    main()
