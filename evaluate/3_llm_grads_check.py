import os
import json
import math
import time
import random
import argparse
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================== Configuration (edit here) ====================
DEVICE = "cuda"

MODEL_NAME_OR_PATH = "THUDM/glm-4-9b-hf"
OUT_DIR = "evaluate/outputs/llm_grads_check/" + datetime.now().strftime("%Y%m%d%H%M%S")

# Data source (HuggingFace dataset)
HF_DATASET = "path/to/your/datasets--allenai--c4/snapshots"
HF_DATASET_CONFIG = "en"
SPLIT = "train"
TEXT_KEY = "text"

# Tokenization & filtering
SEQ_LEN = 2048
BATCH_SIZE = 1
MAX_SAMPLES = 512
MAX_RAW_TEXTS = 5000
ADD_SPECIAL_TOKENS = True
NUM_WORKERS = 0
REBUILD_CACHE = False

# Runtime
SEED = 42
ACCUM_DTYPE = "fp32"          # "fp32" or "fp64"
ATTN_IMPLEMENTATION = "flash_attention_2" # e.g., "flash_attention_2"
# ====================================================================


# -------------------------
# Basic utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


# -------------------------
# Text loading
# -------------------------
def load_texts(
    hf_dataset: Optional[str],
    hf_dataset_config: Optional[str],
    split: str,
    text_key: str,
    max_raw_texts: Optional[int],
) -> List[str]:
    print(f"loading text from {hf_dataset}...")
    ds = load_dataset(hf_dataset, hf_dataset_config, split=split, streaming=True)
    if text_key not in ds.column_names:
        raise KeyError(f"text_key='{text_key}' not found in dataset columns: {ds.column_names}")
    texts = []
    for row in ds:
        text = row[text_key]
        if isinstance(text, str) and text.strip():
            texts.append(text)
            if max_raw_texts is not None and len(texts) >= max_raw_texts:
                break

    if max_raw_texts is not None:
        texts = texts[:max_raw_texts]
    return texts


# -------------------------
# Tokenized probe dataset
# -------------------------
class TokenizedProbeDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        assert input_ids.ndim == 2 and attention_mask.ndim == 2
        assert input_ids.shape == attention_mask.shape
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


@torch.no_grad()
def build_probe_dataset(
    tokenizer,
    hf_dataset: Optional[str],
    hf_dataset_config: Optional[str],
    split: str,
    text_key: str,
    seq_len: int,
    max_samples: int,
    add_special_tokens: bool,
    cache_path: Optional[str] = None,
    max_raw_texts: Optional[int] = None,
) -> TokenizedProbeDataset:
    # if cache_path and os.path.exists(cache_path):
        # obj = torch.load(cache_path, map_location="cpu")
        # return TokenizedProbeDataset(obj["input_ids"], obj["attention_mask"])

    texts = load_texts(
        hf_dataset=hf_dataset,
        hf_dataset_config=hf_dataset_config,
        split=split,
        text_key=text_key,
        max_raw_texts=max_raw_texts,
    )
    if not texts:
        raise ValueError("No usable texts found.")

    # === chunk concat ===
    eos_id = tokenizer.eos_token_id
    all_ids: List[int] = []

    for text in texts:
        ids = tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
        if not ids:
            continue
        all_ids.extend(ids)
        if eos_id is not None:
            all_ids.append(eos_id)
        if len(all_ids) >= seq_len * max_samples:
            break

    num_tokens = len(all_ids)
    num_full = num_tokens // seq_len
    num_keep = min(num_full, max_samples)
    if num_keep <= 0:
        raise ValueError(
            f"Not enough tokens to form one full chunk: have {num_tokens}, need at least seq_len={seq_len}."
        )

    kept = all_ids[: num_keep * seq_len]
    input_ids = torch.tensor(kept, dtype=torch.long).view(num_keep, seq_len)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, cache_path)

    return TokenizedProbeDataset(input_ids, attention_mask)


# -------------------------
# Collector
# -------------------------
class PositionStatsCollector:
    """
    Save only compact position-level statistics.

    Activation stats:
      mean over samples of ||x||_2

    Gradient stats:
      mean over batches of || sum_b grad_b ||_2

    All outputs are [L, T]. No feature axis is retained.
    """

    HIDDEN_ACT_NAMES = (
        "x_in",
        "attn_norm_out",
        "attn_out",
        "x_half",
        "mlp_norm_out",
        "mlp_out",
        "x_out",
    )
    HIDDEN_GRAD_NAMES = HIDDEN_ACT_NAMES
    QKV_GRAD_NAMES = ("q", "k", "v")
    FEEDBACK_GRAD_NAMES = ("attn_fb", "mlp_fb")

    def __init__(
        self,
        model,
        device: torch.device,
        accum_dtype: torch.dtype = torch.float64,
        collect_hidden_acts: Tuple[str, ...] = HIDDEN_ACT_NAMES,
        collect_hidden_grads: Tuple[str, ...] = HIDDEN_GRAD_NAMES,
        collect_qkv_grads: Tuple[str, ...] = QKV_GRAD_NAMES,
        collect_feedback_grads: Tuple[str, ...] = FEEDBACK_GRAD_NAMES,
    ):
        self.model = model
        self.device = device
        self.accum_dtype = accum_dtype
        self.collect_hidden_acts = set(collect_hidden_acts)
        self.collect_hidden_grads = set(collect_hidden_grads)
        self.collect_qkv_grads = set(collect_qkv_grads)
        self.collect_feedback_grads = set(collect_feedback_grads)

        raw = self._unwrap_model(model)
        if not hasattr(raw, "model") or not hasattr(raw.model, "layers"):
            raise TypeError("This script currently supports Llama/Qwen-like models with model.layers.")

        cfg = raw.config
        self.num_layers = int(cfg.num_hidden_layers)
        self.hidden_size = int(cfg.hidden_size)
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(cfg, "num_key_value_heads", self.num_q_heads))

        self.seq_len: Optional[int] = None
        self.buf: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, float] = {}
        self._hooks = []
        self._batch_valid_mask: Optional[torch.Tensor] = None  # [B,T]
        self._tmp_grads: Dict[Tuple[int, str], torch.Tensor] = {}
        self._register_hooks()

    def _unwrap_model(self, m):
        if hasattr(m, "module"):
            m = m.module
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        return m

    def set_batch_attention_mask(self, attention_mask: torch.Tensor) -> None:
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be [B,T], got {tuple(attention_mask.shape)}")
        self._batch_valid_mask = attention_mask.detach().to(device=self.device, dtype=torch.bool)
        self._alloc_once(int(attention_mask.shape[1]))

    def clear_batch_cache(self) -> None:
        self._batch_valid_mask = None
        self._tmp_grads.clear()

    def _alloc_once(self, T: int):
        if self.seq_len is not None:
            if int(T) != int(self.seq_len):
                raise RuntimeError(f"Sequence length changed: {self.seq_len} -> {T}")
            return

        self.seq_len = int(T)
        L = self.num_layers

        def z_layer():
            return torch.zeros((L, T), dtype=self.accum_dtype, device=self.device)

        for name in self.collect_hidden_acts:
            self.buf[f"act_{name}_sum"] = z_layer()
            self.buf[f"act_{name}_valid_count"] = z_layer()
            self.counts[f"act_{name}_sum"] = 1.0

        for name in self.collect_hidden_grads:
            self.buf[f"grad_{name}_sum"] = z_layer()
            self.counts[f"grad_{name}_sum"] = 0.0

        for name in self.collect_qkv_grads:
            self.buf[f"grad_{name}_sum"] = z_layer()
            self.counts[f"grad_{name}_sum"] = 0.0

        for name in self.collect_feedback_grads:
            self.buf[f"grad_{name}_sum"] = z_layer()
            self.counts[f"grad_{name}_sum"] = 0.0

    @staticmethod
    def _extract_tensor(output):
        if isinstance(output, (tuple, list)):
            return output[0] if output else None
        return output

    def _require_mask(self) -> torch.Tensor:
        if self._batch_valid_mask is None:
            raise RuntimeError("Batch attention_mask not set in collector.")
        return self._batch_valid_mask

    def _accum_act_hidden(self, name: str, layer_idx: int, x: torch.Tensor):
        # x: [B,T,D]; count only valid positions
        x = x.detach().to(self.accum_dtype)
        mask = self._require_mask().to(dtype=self.accum_dtype)  # [B,T]
        l2 = torch.linalg.vector_norm(x, ord=2, dim=-1)  # [B,T]
        weighted = l2 * mask
        self.buf[f"act_{name}_sum"][layer_idx] += weighted.sum(dim=0)
        self.buf[f"act_{name}_valid_count"][layer_idx] += mask.sum(dim=0)

    def _accum_grad_hidden(self, name: str, layer_idx: int, grad: torch.Tensor):
        # grad: [B,T,D]; zero out invalid positions before batch aggregation
        g = grad.detach().to(self.accum_dtype)
        mask = self._require_mask().to(dtype=self.accum_dtype).unsqueeze(-1)  # [B,T,1]
        g_sum = (g * mask).sum(dim=0)  # [T,D]
        l2 = torch.linalg.vector_norm(g_sum, ord=2, dim=-1)  # [T]
        self.buf[f"grad_{name}_sum"][layer_idx] += l2
        self.counts[f"grad_{name}_sum"] += 1.0

    def _accum_grad_qkv(self, which: str, layer_idx: int, grad: torch.Tensor):
        # grad: [B,T,H*D]
        g = grad.detach().to(self.accum_dtype)
        mask = self._require_mask().to(dtype=self.accum_dtype).unsqueeze(-1)  # [B,T,1]
        B, T, C = g.shape
        H = self.num_q_heads if which == "q" else self.num_kv_heads
        if C % H != 0:
            raise RuntimeError(f"[{which}] last dim={C} is not divisible by H={H}")
        D = C // H
        g = g.view(B, T, H, D)
        mask = mask.unsqueeze(-1)  # [B,T,1,1]
        g_sum = (g * mask).sum(dim=0)  # [T,H,D]
        l2_head = torch.linalg.vector_norm(g_sum, ord=2, dim=-1)  # [T,H]
        l2_pos = l2_head.mean(dim=-1)  # [T]
        self.buf[f"grad_{which}_sum"][layer_idx] += l2_pos
        self.counts[f"grad_{which}_sum"] += 1.0

    def _maybe_accum_feedback(self, layer_idx: int, which: str):
        if which == "attn_fb":
            needed = [(layer_idx, "x_in"), (layer_idx, "x_half")]
        elif which == "mlp_fb":
            needed = [(layer_idx, "x_half"), (layer_idx, "x_out")]
        else:
            raise ValueError(which)

        if not all(k in self._tmp_grads for k in needed):
            return

        gx0 = self._tmp_grads[needed[0]]
        gx1 = self._tmp_grads[needed[1]]
        diff = gx0 - gx1  # [B,T,D], still vector-level
        mask = self._require_mask().to(dtype=self.accum_dtype).unsqueeze(-1)  # [B,T,1]
        diff_sum = (diff * mask).sum(dim=0)  # [T,D]
        l2 = torch.linalg.vector_norm(diff_sum, ord=2, dim=-1)  # [T]
        self.buf[f"grad_{which}_sum"][layer_idx] += l2
        self.counts[f"grad_{which}_sum"] += 1.0

        # free memory as soon as the downstream feedback is materialized
        if which == "attn_fb":
            del self._tmp_grads[(layer_idx, "x_in")]
            if (layer_idx, "x_half") in self._tmp_grads:
                del self._tmp_grads[(layer_idx, "x_half")]
        elif which == "mlp_fb":
            if (layer_idx, "x_out") in self._tmp_grads:
                del self._tmp_grads[(layer_idx, "x_out")]

    def _stash_grad_for_feedback(self, layer_idx: int, name: str, grad: torch.Tensor):
        if not (("attn_fb" in self.collect_feedback_grads and name in {"x_in", "x_half"}) or
                ("mlp_fb" in self.collect_feedback_grads and name in {"x_half", "x_out"})):
            return
        self._tmp_grads[(layer_idx, name)] = grad.detach().to(self.accum_dtype)
        if name in {"x_in", "x_half"} and "attn_fb" in self.collect_feedback_grads:
            self._maybe_accum_feedback(layer_idx, "attn_fb")
        if name in {"x_half", "x_out"} and "mlp_fb" in self.collect_feedback_grads:
            self._maybe_accum_feedback(layer_idx, "mlp_fb")

    def _make_layer_input_hook(self, layer_idx: int):
        def pre_hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None or (not torch.is_tensor(x)) or x.ndim != 3:
                return
            _, T, _ = x.shape
            self._alloc_once(T)

            if "x_in" in self.collect_hidden_acts:
                self._accum_act_hidden("x_in", layer_idx, x)

            if "x_in" in self.collect_hidden_grads or "attn_fb" in self.collect_feedback_grads:
                def _hook(grad):
                    if grad is None:
                        return None
                    if "x_in" in self.collect_hidden_grads:
                        self._accum_grad_hidden("x_in", layer_idx, grad)
                    self._stash_grad_for_feedback(layer_idx, "x_in", grad)
                    return None
                x.register_hook(_hook)
        return pre_hook

    def _make_hidden_out_hook(self, layer_idx: int, name: str):
        def fwd_hook(module, inputs, output):
            y = self._extract_tensor(output)
            if y is None or (not torch.is_tensor(y)) or y.ndim != 3:
                return
            _, T, _ = y.shape
            self._alloc_once(T)

            if name in self.collect_hidden_acts:
                self._accum_act_hidden(name, layer_idx, y)

            if name in self.collect_hidden_grads or name in {"x_half", "x_out"}:
                def _hook(grad):
                    if grad is None:
                        return None
                    if name in self.collect_hidden_grads:
                        self._accum_grad_hidden(name, layer_idx, grad)
                    self._stash_grad_for_feedback(layer_idx, name, grad)
                    return None
                y.register_hook(_hook)
        return fwd_hook

    def _make_qkv_proj_hook(self, layer_idx: int, which: str):
        def fwd_hook(module, inputs, output):
            y = output
            if y is None or (not torch.is_tensor(y)) or y.ndim != 3:
                return
            _, T, _ = y.shape
            self._alloc_once(T)
            if which in self.collect_qkv_grads:
                y.register_hook(lambda grad: self._accum_grad_qkv(which, layer_idx, grad) if grad is not None else None)
        return fwd_hook

    def _register_hooks(self):
        raw = self._unwrap_model(self.model)
        layers = raw.model.layers

        for l, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_pre_hook(self._make_layer_input_hook(l), with_kwargs=False))

            if hasattr(layer, "input_layernorm"):
                self._hooks.append(layer.input_layernorm.register_forward_hook(self._make_hidden_out_hook(l, "attn_norm_out")))

            attn = layer.self_attn
            self._hooks.append(attn.q_proj.register_forward_hook(self._make_qkv_proj_hook(l, "q")))
            self._hooks.append(attn.k_proj.register_forward_hook(self._make_qkv_proj_hook(l, "k")))
            self._hooks.append(attn.v_proj.register_forward_hook(self._make_qkv_proj_hook(l, "v")))
            self._hooks.append(attn.register_forward_hook(self._make_hidden_out_hook(l, "attn_out")))

            if hasattr(layer, "post_attention_layernorm"):
                self._hooks.append(
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        self._make_layernorm_input_hook(l, "x_half"), with_kwargs=False
                    )
                )
                self._hooks.append(layer.post_attention_layernorm.register_forward_hook(self._make_hidden_out_hook(l, "mlp_norm_out")))

            self._hooks.append(layer.mlp.register_forward_hook(self._make_hidden_out_hook(l, "mlp_out")))
            self._hooks.append(layer.register_forward_hook(self._make_hidden_out_hook(l, "x_out")))

    def _make_layernorm_input_hook(self, layer_idx: int, name: str):
        def pre_hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None or (not torch.is_tensor(x)) or x.ndim != 3:
                return
            _, T, _ = x.shape
            self._alloc_once(T)

            if name in self.collect_hidden_acts:
                self._accum_act_hidden(name, layer_idx, x)

            if name in self.collect_hidden_grads or name in {"x_half"}:
                def _hook(grad):
                    if grad is None:
                        return None
                    if name in self.collect_hidden_grads:
                        self._accum_grad_hidden(name, layer_idx, grad)
                    self._stash_grad_for_feedback(layer_idx, name, grad)
                    return None
                x.register_hook(_hook)
        return pre_hook

    def finalize(self) -> Dict[str, torch.Tensor]:
        if self.seq_len is None:
            raise RuntimeError("Collector saw no tensors. Did the model run successfully?")
        if self._tmp_grads:
            # Should be empty after each batch. If not, something in hook pairing is broken.
            raise RuntimeError(f"Temporary feedback gradient cache not empty at finalize: {list(self._tmp_grads.keys())[:8]}")

        out: Dict[str, torch.Tensor] = {}
        for key, value in self.buf.items():
            if key.endswith("_valid_count"):
                out[key] = value.detach().cpu()
                continue
            if key.startswith("act_"):
                valid_key = key.replace("_sum", "_valid_count")
                denom = self.buf[valid_key].clamp_min(1.0)
                out[key.replace("_sum", "_mean")] = (value / denom).detach().cpu()
            else:
                denom = self.counts.get(key, 0.0)
                if denom <= 0:
                    raise RuntimeError(f"Zero count for buffer: {key}")
                out[key.replace("_sum", "_mean")] = (value / denom).detach().cpu()
        return out

    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self.clear_batch_cache()


# -------------------------
# Model helpers
# -------------------------
def enable_input_require_grads_safely(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    emb = model.get_input_embeddings()

    def _set_require_grads(module, inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)

    emb.register_forward_hook(_set_require_grads)


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    attn_implementation: Optional[str],
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    model_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    model.config.use_cache = False
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    enable_input_require_grads_safely(model)
    return model, tokenizer


# -------------------------
# Main run
# -------------------------
def run_one_pass(
    model,
    dataloader,
    device: torch.device,
    collector: PositionStatsCollector,
    use_autocast: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    losses: List[float] = []

    pbar = tqdm(dataloader, desc="Running Gradient Check", unit="batch", leave=True)
    for batch_idx, batch in enumerate(pbar):
    # for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        collector.set_batch_attention_mask(attention_mask)
        model.zero_grad(set_to_none=True)

        if use_autocast:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        collector.clear_batch_cache()

    stats = collector.finalize()
    stats["loss_mean"] = torch.tensor(sum(losses) / max(len(losses), 1), dtype=torch.float32)
    stats["loss_all"] = torch.tensor(losses, dtype=torch.float32)
    return stats


# -------------------------
# Save
# -------------------------
def save_outputs(
    out_dir: str,
    stats: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(stats, os.path.join(out_dir, "stats.pt"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# -------------------------
# main
# -------------------------
def main():
    set_seed(SEED)

    device = torch.device(DEVICE)
    
    model_dtype = torch.bfloat16
    accum_dtype = torch.float64 if ACCUM_DTYPE == "fp64" else torch.float32
    use_autocast = bool(device.type == "cuda" and model_dtype in (torch.bfloat16, torch.float16))

    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUT_DIR, "probe_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_tag = (
        f"T{SEQ_LEN}.N{MAX_SAMPLES}."
        f"special{int(ADD_SPECIAL_TOKENS)}.seed{SEED}.pt"
    )
    probe_cache_path = os.path.join(cache_dir, cache_tag)
    if REBUILD_CACHE and os.path.exists(probe_cache_path):
        os.remove(probe_cache_path)

    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=MODEL_NAME_OR_PATH,
        dtype=model_dtype,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        device=device,
    )

    print(f"[OK] Model loaded")

    probe_ds = build_probe_dataset(
        tokenizer=tokenizer,
        hf_dataset=HF_DATASET,
        hf_dataset_config=HF_DATASET_CONFIG,
        split=SPLIT,
        text_key=TEXT_KEY,
        seq_len=SEQ_LEN,
        max_samples=MAX_SAMPLES,
        add_special_tokens=ADD_SPECIAL_TOKENS,
        cache_path=probe_cache_path,
        max_raw_texts=MAX_RAW_TEXTS,
    )

    print(f"[OK] probe built")

    loader = DataLoader(
        probe_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    collector = PositionStatsCollector(
        model=model,
        device=device,
        accum_dtype=accum_dtype,
    )

    print(f"[OK] ready to run...")
    stats = run_one_pass(
        model=model,
        dataloader=loader,
        device=device,
        collector=collector,
        use_autocast=use_autocast,
        amp_dtype=model_dtype,
    )
    collector.close()

    special_tokens = {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }

    example_ids = probe_ds.input_ids[0, : min(32, probe_ds.input_ids.shape[1])].tolist()
    meta = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "device": str(device),
        "accum_dtype": ACCUM_DTYPE,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "hf_dataset": HF_DATASET,
        "hf_dataset_config": HF_DATASET_CONFIG,
        "split": SPLIT,
        "text_key": TEXT_KEY,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "num_batches": math.ceil(len(probe_ds) / BATCH_SIZE),
        "num_samples": len(probe_ds),
        "add_special_tokens": ADD_SPECIAL_TOKENS,
        "seed": SEED,
        "special_tokens": special_tokens,
        "example_input_ids_first32": example_ids,
        "elapsed_sec": round(time.time() - t0, 3),
        "stat_keys": sorted(list(stats.keys())),
    }

    save_outputs(OUT_DIR, stats, meta)

    print(json.dumps({
        "saved_dir": OUT_DIR,
        "num_samples": len(probe_ds),
        "num_batches": math.ceil(len(probe_ds) / BATCH_SIZE),
        "loss_mean": float(stats["loss_mean"].item()),
        "elapsed_sec": meta["elapsed_sec"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
