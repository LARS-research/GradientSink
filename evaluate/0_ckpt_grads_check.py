#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compact gradient probe script for Llama/V-scale checkpoints.

This version is designed for downstream plotting of:
1. gradient violin plots
2. Q/K/V gradient line plots
3. activation-vs-gradient-ratio scatter plots

Key design choices:
- Match training semantics as closely as possible:
  * streaming dataset + fixed probe microbatches
  * per-rank sharding
  * AMP autocast
  * loss averaged over accumulation steps
  * global-batch mean gradient after cross-rank reduction
- Keep exact vector accumulation IN MEMORY during the run where needed,
  so branch-difference norms are computed exactly.
- SAVE ONLY token-wise / layer-wise norm summaries. No full [L,T,D] or
  [L,T,C] vectors are written to disk.

Output directory structure:
  out_dir/<timestamp>/
    probe_cache/
    ckpts/ckpt_<step>/
      collects.pt
      meta.json
    summary.csv

Saved payload format:
  {"collects": ..., "meta": ...}

Compatibility note:
- Legacy field names such as D_q_sum / D_x_in_sum are retained, but they now
  store norm summaries, not full vectors.
- Explicit *_norm aliases are also saved for readability.
"""

import argparse
import contextlib
import csv
import glob
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training_tools import load_model_from_checkpoint


# =============================
# distributed helpers
# =============================

def setup_distributed() -> Tuple[int, int, int, bool, bool, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    is_main = rank == 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe script.")

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    return world_size, rank, local_rank, is_ddp, is_main, device


def barrier_if_needed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def infer_step_from_path(path: str) -> int:
    base = os.path.basename(path)
    matches = re.findall(r"(\d+)", base)
    return int(matches[-1]) if matches else -1


def get_autocast_dtype(amp_mode: str) -> Optional[torch.dtype]:
    amp_mode = amp_mode.lower()
    if amp_mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_mode == "fp32":
        return None
    if amp_mode == "bf16":
        return torch.bfloat16
    if amp_mode == "fp16":
        return torch.float16
    raise ValueError(f"Unknown amp mode: {amp_mode}")


def make_autocast_context(amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def get_accum_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp32":
        return torch.float32
    if name == "fp64":
        return torch.float64
    raise ValueError(f"Unknown accum dtype: {name}")


# =============================
# tokenization / probe
# =============================

def tokenize_and_group(
    examples: Dict[str, List],
    tokenizer,
    block_size: int,
    add_bos: bool = False,
    add_eos: bool = True,
):
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


@torch.no_grad()
def build_probe_microbatches_for_rank(
    tokenizer_path: str,
    dataset_path: str,
    data_name: str,
    split: str,
    block_size: int,
    local_batch_size: int,
    accumulation_steps: int,
    seed: int,
    world_size: int,
    rank: int,
    cache_path: str,
    add_bos: bool = False,
    add_eos: bool = True,
) -> torch.Tensor:
    """Returns [ACCUM_STEPS, LOCAL_BATCH, T] input_ids."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.model_max_length = int(1e8)

    raw = load_dataset(dataset_path, data_name, split=split, streaming=True)
    raw = raw.shuffle(seed=seed, buffer_size=100000)
    raw = raw.shard(num_shards=world_size, index=rank)

    remove_columns = raw.column_names
    ds = raw.map(
        tokenize_and_group,
        batched=True,
        batch_size=1024,
        remove_columns=remove_columns,
        fn_kwargs={
            "tokenizer": tok,
            "block_size": block_size,
            "add_bos": add_bos,
            "add_eos": add_eos,
        },
    )

    loader = DataLoader(
        ds,
        batch_size=local_batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )

    batches = []
    it = iter(loader)
    for _ in range(accumulation_steps):
        batch = next(it)
        batches.append(batch["input_ids"].contiguous())

    probe = torch.stack(batches, dim=0)
    torch.save(probe, cache_path)
    return probe


# =============================
# Hook-based collector
# =============================

class GradCollectorCompact:
    """
    Accumulate exact global-batch-mean gradient vectors in memory, but save only
    norm summaries.

    In-memory exact vectors:
      - q / k / v projected gradients
      - optional v_postscale gradients
      - x_in / attn_norm_out / x_half / mlp_norm_out / x_out gradients

    Saved summaries only:
      - token-wise / layer-wise norm tensors, plus exact branch-difference norms
      - forward L2 summaries
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        accum_dtype: torch.dtype = torch.float32,
        collect_v_postscale: bool = True,
    ):
        self.model = model
        self.device = device
        self.accum_dtype = accum_dtype
        self.collect_v_postscale = collect_v_postscale

        raw = self._unwrap_model(model)
        cfg = raw.config

        self.num_layers = int(cfg.num_hidden_layers)
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(cfg, "num_key_value_heads", self.num_q_heads))
        self.hidden_size = int(cfg.hidden_size)
        self.head_dim = int(getattr(cfg, "head_dim", self.hidden_size // self.num_q_heads))
        self.q_proj_dim = self.num_q_heads * self.head_dim
        self.kv_proj_dim = self.num_kv_heads * self.head_dim

        self.seq_len: Optional[int] = None
        self.buf: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self.has_v_radial_layers: List[bool] = [False for _ in range(self.num_layers)]
        self._register_hooks()

    def _unwrap_model(self, m: torch.nn.Module) -> torch.nn.Module:
        if hasattr(m, "module"):
            m = m.module
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        return m

    def _alloc_once(self, T: int):
        if self.seq_len is not None:
            if int(T) != int(self.seq_len):
                raise RuntimeError(f"Seq len changed: was {self.seq_len}, got {T}")
            return

        self.seq_len = int(T)
        L = self.num_layers
        D = self.hidden_size
        T = self.seq_len

        def z(shape):
            return torch.zeros(shape, device=self.device, dtype=self.accum_dtype)

        # Exact vectors kept only in memory.
        self.buf["D_q_vec"] = z((L, T, self.q_proj_dim))
        self.buf["D_k_vec"] = z((L, T, self.kv_proj_dim))
        self.buf["D_v_vec"] = z((L, T, self.kv_proj_dim))
        if any(self.has_v_radial_layers):
            self.buf["D_v_postscale_vec"] = z((L, T, self.kv_proj_dim))

        self.buf["D_x_in_vec"] = z((L, T, D))
        self.buf["D_attn_norm_out_vec"] = z((L, T, D))
        self.buf["D_x_half_vec"] = z((L, T, D))
        self.buf["D_mlp_norm_out_vec"] = z((L, T, D))
        self.buf["D_x_out_vec"] = z((T, D))

        # Forward norms.
        self.buf["x_in_l2"] = z((L, T))
        self.buf["x_half_l2"] = z((L, T))
        self.buf["x_out_l2"] = z((T,))
        self.buf["attn_out_l2"] = z((L, T))
        self.buf["mlp_out_l2"] = z((L, T))
        if any(self.has_v_radial_layers):
            self.buf["v_postscale_l2"] = z((L, T))

    def _register_hooks(self):
        raw = self._unwrap_model(self.model)

        for layer_idx in range(self.num_layers):
            layer = raw.model.layers[layer_idx]
            attn = layer.self_attn
            mlp = layer.mlp

            if self.collect_v_postscale and hasattr(attn, "v_radial"):
                self.has_v_radial_layers[layer_idx] = True

            self._hooks.append(
                layer.register_forward_pre_hook(self._make_hidden_pre_hook(layer_idx, "x_in"), with_kwargs=False)
            )

            if hasattr(layer, "input_layernorm"):
                self._hooks.append(
                    layer.input_layernorm.register_forward_hook(self._make_hidden_out_hook(layer_idx, "attn_norm_out"))
                )

            self._hooks.append(attn.q_proj.register_forward_hook(self._make_proj_hook(layer_idx, "q")))
            self._hooks.append(attn.k_proj.register_forward_hook(self._make_proj_hook(layer_idx, "k")))
            self._hooks.append(attn.v_proj.register_forward_hook(self._make_proj_hook(layer_idx, "v")))

            if self.collect_v_postscale and hasattr(attn, "v_radial"):
                self._hooks.append(attn.v_radial.register_forward_hook(self._make_v_postscale_hook(layer_idx)))

            self._hooks.append(attn.register_forward_hook(self._make_hidden_out_hook(layer_idx, "attn_out")))

            if hasattr(layer, "post_attention_layernorm"):
                self._hooks.append(
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        self._make_hidden_pre_hook(layer_idx, "x_half"), with_kwargs=False
                    )
                )
                self._hooks.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        self._make_hidden_out_hook(layer_idx, "mlp_norm_out")
                    )
                )

            self._hooks.append(mlp.register_forward_hook(self._make_hidden_out_hook(layer_idx, "mlp_out")))

            if layer_idx == self.num_layers - 1:
                self._hooks.append(layer.register_forward_hook(self._make_last_layer_output_hook()))

    def _accum_hidden_grad(self, vec_key: str, layer_idx: int, grad: torch.Tensor):
        g = grad.detach().to(self.accum_dtype).sum(dim=0)
        if vec_key == "D_x_out_vec":
            self.buf[vec_key] += g
        else:
            self.buf[vec_key][layer_idx] += g

    def _accum_proj_grad(self, vec_key: str, layer_idx: int, grad: torch.Tensor):
        g = grad.detach().to(self.accum_dtype).sum(dim=0)
        self.buf[vec_key][layer_idx] += g

    def _accum_v_postscale_grad(self, layer_idx: int, grad: torch.Tensor):
        g = grad.detach().to(self.accum_dtype).transpose(1, 2).reshape(grad.shape[0], grad.shape[2], -1).sum(dim=0)
        self.buf["D_v_postscale_vec"][layer_idx] += g

    def _accum_hidden_fwd_l2(self, key: str, layer_idx: int, x: torch.Tensor):
        l2 = torch.linalg.vector_norm(x.detach().to(self.accum_dtype), ord=2, dim=-1).mean(dim=0)
        if key == "x_out":
            self.buf["x_out_l2"] += l2
        else:
            self.buf[f"{key}_l2"][layer_idx] += l2

    def _accum_v_postscale_fwd_l2(self, layer_idx: int, x: torch.Tensor):
        l2 = torch.linalg.vector_norm(
            x.detach().to(self.accum_dtype).transpose(1, 2).reshape(x.shape[0], x.shape[2], -1),
            ord=2,
            dim=-1,
        ).mean(dim=0)
        self.buf["v_postscale_l2"][layer_idx] += l2

    def _make_hidden_pre_hook(self, layer_idx: int, key: str):
        def pre_hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None or (not torch.is_tensor(x)) or x.dim() != 3:
                return

            _, T, _ = x.shape
            self._alloc_once(T)

            if key in ("x_in", "x_half"):
                self._accum_hidden_fwd_l2(key, layer_idx, x)

            if x.requires_grad:
                if key == "x_in":
                    x.register_hook(lambda grad, li=layer_idx: self._accum_hidden_grad("D_x_in_vec", li, grad))
                elif key == "x_half":
                    x.register_hook(lambda grad, li=layer_idx: self._accum_hidden_grad("D_x_half_vec", li, grad))

        return pre_hook

    def _make_hidden_out_hook(self, layer_idx: int, key: str):
        def fwd_hook(module, inputs, output):
            y = output[0] if isinstance(output, (tuple, list)) else output
            if y is None or (not torch.is_tensor(y)):
                return None

            if y.dim() != 3:
                return None

            _, T, _ = y.shape
            self._alloc_once(T)

            if key == "attn_out":
                self._accum_hidden_fwd_l2("attn_out", layer_idx, y)
            elif key == "mlp_out":
                self._accum_hidden_fwd_l2("mlp_out", layer_idx, y)
            elif key == "attn_norm_out":
                if y.requires_grad:
                    y.register_hook(lambda grad, li=layer_idx: self._accum_hidden_grad("D_attn_norm_out_vec", li, grad))
            elif key == "mlp_norm_out":
                if y.requires_grad:
                    y.register_hook(lambda grad, li=layer_idx: self._accum_hidden_grad("D_mlp_norm_out_vec", li, grad))
            return None

        return fwd_hook

    def _make_last_layer_output_hook(self):
        def fwd_hook(module, inputs, output):
            y = output[0] if isinstance(output, (tuple, list)) else output
            if y is None or (not torch.is_tensor(y)) or y.dim() != 3:
                return None

            _, T, _ = y.shape
            self._alloc_once(T)
            self._accum_hidden_fwd_l2("x_out", self.num_layers - 1, y)
            if y.requires_grad:
                y.register_hook(lambda grad: self._accum_hidden_grad("D_x_out_vec", self.num_layers - 1, grad))
            return None

        return fwd_hook

    def _make_proj_hook(self, layer_idx: int, key: str):
        def fwd_hook(module, inputs, output):
            y = output
            if y is None or (not torch.is_tensor(y)) or y.dim() != 3:
                return None
            _, T, _ = y.shape
            self._alloc_once(T)
            if y.requires_grad:
                if key == "q":
                    y.register_hook(lambda grad, li=layer_idx: self._accum_proj_grad("D_q_vec", li, grad))
                elif key == "k":
                    y.register_hook(lambda grad, li=layer_idx: self._accum_proj_grad("D_k_vec", li, grad))
                elif key == "v":
                    y.register_hook(lambda grad, li=layer_idx: self._accum_proj_grad("D_v_vec", li, grad))
            return None

        return fwd_hook

    def _make_v_postscale_hook(self, layer_idx: int):
        def fwd_hook(module, inputs, output):
            y = output
            if y is None or (not torch.is_tensor(y)) or y.dim() != 4:
                return None
            _, _, T, _ = y.shape
            self._alloc_once(T)
            self._accum_v_postscale_fwd_l2(layer_idx, y)
            if y.requires_grad:
                y.register_hook(lambda grad, li=layer_idx: self._accum_v_postscale_grad(li, grad))
            return None

        return fwd_hook

    def allreduce_(self, is_ddp: bool) -> None:
        if self.seq_len is None or not is_ddp:
            return
        for tensor in self.buf.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def finalize(self, world_size: int, accumulation_steps: int) -> Dict[str, torch.Tensor]:
        if self.seq_len is None:
            raise RuntimeError("No data collected; did the backward pass run?")

        # Normalize exact gradient vectors to global-batch mean gradients.
        q_vec = self.buf["D_q_vec"] / float(world_size)
        k_vec = self.buf["D_k_vec"] / float(world_size)
        v_vec = self.buf["D_v_vec"] / float(world_size)
        x_in_vec = self.buf["D_x_in_vec"] / float(world_size)
        attn_norm_out_vec = self.buf["D_attn_norm_out_vec"] / float(world_size)
        x_half_vec = self.buf["D_x_half_vec"] / float(world_size)
        mlp_norm_out_vec = self.buf["D_mlp_norm_out_vec"] / float(world_size)
        x_out_vec = self.buf["D_x_out_vec"] / float(world_size)
        v_postscale_vec = None
        if "D_v_postscale_vec" in self.buf:
            v_postscale_vec = self.buf["D_v_postscale_vec"] / float(world_size)

        # Forward means.
        x_in_l2 = self.buf["x_in_l2"] / float(world_size * accumulation_steps)
        x_half_l2 = self.buf["x_half_l2"] / float(world_size * accumulation_steps)
        x_out_l2 = self.buf["x_out_l2"] / float(world_size * accumulation_steps)
        attn_out_l2 = self.buf["attn_out_l2"] / float(world_size * accumulation_steps)
        mlp_out_l2 = self.buf["mlp_out_l2"] / float(world_size * accumulation_steps)
        v_postscale_l2 = None
        if "v_postscale_l2" in self.buf:
            v_postscale_l2 = self.buf["v_postscale_l2"] / float(world_size * accumulation_steps)

        # Exact branch-difference vectors, but only their norms are kept.
        attn_branch_vec = x_in_vec - x_half_vec
        x_out_like_vec = torch.zeros_like(x_in_vec)
        x_out_like_vec[:-1] = x_in_vec[1:]
        x_out_like_vec[-1] = x_out_vec
        mlp_branch_vec = x_half_vec - x_out_like_vec

        # Save summaries only.
        out: Dict[str, torch.Tensor] = {}

        # Legacy names retained, but values are norms now.
        out["D_q_sum"] = torch.linalg.vector_norm(q_vec, ord=2, dim=-1)
        out["D_k_sum"] = torch.linalg.vector_norm(k_vec, ord=2, dim=-1)
        out["D_v_sum"] = torch.linalg.vector_norm(v_vec, ord=2, dim=-1)
        if v_postscale_vec is not None:
            out["D_v_postscale_sum"] = torch.linalg.vector_norm(v_postscale_vec, ord=2, dim=-1)

        out["D_x_in_sum"] = torch.linalg.vector_norm(x_in_vec, ord=2, dim=-1)
        out["D_attn_norm_out_sum"] = torch.linalg.vector_norm(attn_norm_out_vec, ord=2, dim=-1)
        out["D_x_half_sum"] = torch.linalg.vector_norm(x_half_vec, ord=2, dim=-1)
        out["D_mlp_norm_out_sum"] = torch.linalg.vector_norm(mlp_norm_out_vec, ord=2, dim=-1)
        out["D_x_out_sum"] = torch.linalg.vector_norm(x_out_vec, ord=2, dim=-1)
        out["D_attn_branch_sum"] = torch.linalg.vector_norm(attn_branch_vec, ord=2, dim=-1)
        out["D_x_out_like_sum"] = torch.linalg.vector_norm(x_out_like_vec, ord=2, dim=-1)
        out["D_mlp_branch_sum"] = torch.linalg.vector_norm(mlp_branch_vec, ord=2, dim=-1)

        # Explicit aliases.
        out["D_q_norm"] = out["D_q_sum"]
        out["D_k_norm"] = out["D_k_sum"]
        out["D_v_norm"] = out["D_v_sum"]
        if v_postscale_vec is not None:
            out["D_v_postscale_norm"] = out["D_v_postscale_sum"]
        out["D_x_in_norm"] = out["D_x_in_sum"]
        out["D_attn_norm_out_norm"] = out["D_attn_norm_out_sum"]
        out["D_x_half_norm"] = out["D_x_half_sum"]
        out["D_mlp_norm_out_norm"] = out["D_mlp_norm_out_sum"]
        out["D_x_out_norm"] = out["D_x_out_sum"]
        out["D_attn_branch_norm"] = out["D_attn_branch_sum"]
        out["D_x_out_like_norm"] = out["D_x_out_like_sum"]
        out["D_mlp_branch_norm"] = out["D_mlp_branch_sum"]

        # Forward summaries.
        out["x_in_l2"] = x_in_l2
        out["x_half_l2"] = x_half_l2
        out["x_out_l2"] = x_out_l2
        out["attn_out_l2"] = attn_out_l2
        out["mlp_out_l2"] = mlp_out_l2
        if v_postscale_l2 is not None:
            out["v_postscale_l2"] = v_postscale_l2

        return out

    def close(self):
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks = []


# =============================
# Main analysis per checkpoint
# =============================

def analyze_one_checkpoint(
    ckpt_path: str,
    probe: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    is_ddp: bool,
    is_main: bool,
    world_size: int,
    accumulation_steps: int,
    accum_dtype: torch.dtype,
) -> Tuple[int, Dict[str, torch.Tensor], Dict]:
    model, config, step = load_model_from_checkpoint(ckpt_path)
    if step < 0:
        step = infer_step_from_path(ckpt_path)

    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"
        model.config.use_cache = False

    model.to(device)

    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    for param in model.parameters():
        param.requires_grad_(False)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.train()
    model.zero_grad(set_to_none=True)

    collector = GradCollectorCompact(
        model=model,
        device=device,
        accum_dtype=accum_dtype,
        collect_v_postscale=True,
    )

    for acc_idx in range(accumulation_steps):
        input_ids = probe[acc_idx].to(device, non_blocking=True)
        labels = input_ids

        with make_autocast_context(amp_dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / accumulation_steps

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at checkpoint {ckpt_path}: {float(loss.item())}")

        if is_main and acc_idx == 0:
            print(f"[Step {step}] loss (microbatch {acc_idx + 1}/{accumulation_steps}) = {loss.item():.6f}")

        loss.backward()

    barrier_if_needed(is_ddp)
    collector.allreduce_(is_ddp)
    collects = collector.finalize(world_size=world_size, accumulation_steps=accumulation_steps)

    meta = {
        "checkpoint_path": ckpt_path,
        "step": int(step),
        "seq_len": int(probe.shape[2]),
        "local_batch": int(probe.shape[1]),
        "accumulation_steps": int(accumulation_steps),
        "world_size": int(world_size),
        "global_batch": int(probe.shape[1] * accumulation_steps * world_size),
        "amp": str(amp_dtype),
        "accum_dtype": str(accum_dtype),
        "num_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "num_q_heads": int(config.num_attention_heads),
        "num_kv_heads": int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "collector_variant": "compact_save_norm_only",
        "saved_tensor_semantics": "tokenwise_norm_only",
        "forward_l2_stat": "mean_over_global_batch",
        "grad_vector_stat": "global_batch_mean_gradient_before_norm",
        "has_v_radial_layers": collector.has_v_radial_layers,
        "attn_implementation_for_probe": getattr(model.config, "_attn_implementation", None),
    }

    collector.close()
    del model
    torch.cuda.empty_cache()
    return step, collects, meta


# =============================
# Saving
# =============================

def save_checkpoint_outputs(run_dir: str, step: int, meta: Dict, collects: Dict[str, torch.Tensor]) -> str:
    ckpt_dir = os.path.join(run_dir, "ckpts", f"ckpt_{step:05d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    meta_path = os.path.join(ckpt_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    collects_path = os.path.join(ckpt_dir, "collects.pt")
    collects_cpu = {k: v.detach().cpu() for k, v in collects.items()}
    torch.save({"collects": collects_cpu, "meta": meta}, collects_path)
    return collects_path


def append_summary_csv(run_dir: str, row: Dict):
    path = os.path.join(run_dir, "summary.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =============================
# CLI
# =============================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_glob", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--accum_dtype", type=str, default="fp32", choices=["fp32", "fp64"])

    ap.add_argument("--tokenizer_path", type=str, default="GPT2-tokenizer")
    ap.add_argument("--dataset_path", type=str, default="allenai/c4")
    ap.add_argument("--data_name", type=str, default="en")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--add_bos", type=str2bool, default=False)
    ap.add_argument("--add_eos", type=str2bool, default=True)

    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--global_batch", type=int, default=512)
    ap.add_argument("--local_batch", type=int, default=32)
    return ap


# =============================
# Main
# =============================

def main():
    args = build_argparser().parse_args()

    world_size, rank, local_rank, is_ddp, is_main, device = setup_distributed()
    del local_rank
    seed_everything(args.seed, rank)

    if args.global_batch % (args.local_batch * world_size) != 0:
        raise ValueError("global_batch must be divisible by local_batch * WORLD_SIZE")

    accumulation_steps = args.global_batch // (args.local_batch * world_size)
    amp_dtype = get_autocast_dtype(args.amp_dtype)
    accum_dtype = get_accum_dtype(args.accum_dtype)

    tag = time.strftime("%Y%m%d_%H%M%S") if is_main else ""
    if is_ddp:
        obj = [tag]
        dist.broadcast_object_list(obj, src=0)
        tag = obj[0]

    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    probe_cache_dir = os.path.join(run_dir, "probe_cache")
    os.makedirs(probe_cache_dir, exist_ok=True)
    probe_cache_path = os.path.join(
        probe_cache_dir,
        f"probe.rank{rank}.T{args.seq_len}.localB{args.local_batch}.acc{accumulation_steps}.seed{args.seed}.pt",
    )

    barrier_if_needed(is_ddp)
    probe = build_probe_microbatches_for_rank(
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path,
        data_name=args.data_name,
        split=args.split,
        block_size=args.seq_len,
        local_batch_size=args.local_batch,
        accumulation_steps=accumulation_steps,
        seed=args.seed,
        world_size=world_size,
        rank=rank,
        cache_path=probe_cache_path,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )
    barrier_if_needed(is_ddp)

    if is_main:
        print(f"Probe cache ready: {os.path.basename(probe_cache_path)}")
        print(
            f"world_size={world_size} local_batch={args.local_batch} global_batch={args.global_batch} "
            f"accumulation_steps={accumulation_steps} amp={args.amp_dtype} accum_dtype={args.accum_dtype}"
        )

    ckpts = sorted(glob.glob(args.ckpt_glob), key=infer_step_from_path)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpt_glob}")

    if is_main:
        print(f"Found {len(ckpts)} checkpoints.")

    try:
        for ckpt_path in ckpts:
            t0 = time.time()
            step, collects, meta = analyze_one_checkpoint(
                ckpt_path=ckpt_path,
                probe=probe,
                device=device,
                amp_dtype=amp_dtype,
                is_ddp=is_ddp,
                is_main=is_main,
                world_size=world_size,
                accumulation_steps=accumulation_steps,
                accum_dtype=accum_dtype,
            )
            barrier_if_needed(is_ddp)

            if is_main:
                collects_path = save_checkpoint_outputs(run_dir, step, meta, collects)
                row = {
                    "step": step,
                    "elapsed_sec": round(time.time() - t0, 3),
                    "collects_path": collects_path,
                }
                append_summary_csv(run_dir, row)
                print(f"[rank0] step={step} saved. time={row['elapsed_sec']}s")

            barrier_if_needed(is_ddp)
    finally:
        cleanup_distributed(is_ddp)


if __name__ == "__main__":
    main()
