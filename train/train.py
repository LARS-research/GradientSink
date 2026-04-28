import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import datetime

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

# Optional custom model.
try:
    from modeling_vscale import VscaleForCausalLM
except Exception:
    VscaleForCausalLM = None


# -----------------------------
# Utilities
# -----------------------------

def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def setup_distributed() -> Tuple[int, int, int, bool, bool, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    is_main = rank == 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    return world_size, rank, local_rank, is_ddp, is_main, device


def cleanup_distributed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    # Per-rank seeding is the common LLM training convention.
    s = seed + rank
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def barrier_if_needed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device, is_ddp: bool) -> float:
    if not is_ddp:
        return float(value)
    x = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return float(x.item())


def all_reduce_sum_count(sum_value: float, count_value: int, device: torch.device, is_ddp: bool) -> Tuple[float, int]:
    if not is_ddp:
        return float(sum_value), int(count_value)
    x = torch.tensor([sum_value, float(count_value)], dtype=torch.float64, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return float(x[0].item()), int(round(x[1].item()))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


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


def grad_scaler_for_amp(amp_dtype: Optional[torch.dtype]) -> torch.cuda.amp.GradScaler:
    # GradScaler is only needed for fp16.
    enabled = amp_dtype == torch.float16
    return torch.amp.GradScaler(enabled=enabled)


def cosine_lr(step: int, warmup_iters: int, max_iters: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_iters:
        return max_lr * float(step + 1) / max(1, warmup_iters)
    if step >= max_iters:
        return min_lr
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def get_fused_adamw_supported() -> bool:
    try:
        return "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    except Exception:
        return False


def get_no_decay_keywords(args) -> List[str]:
    keywords = [x.strip().lower() for x in args.no_decay_keywords.split(",") if x.strip()]
    return keywords


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    no_decay_keywords = get_no_decay_keywords(args)

    named_params = list(unwrap_model(model).named_parameters())
    for name, param in named_params:
        if not param.requires_grad:
            continue
        lname = name.lower()
        force_no_decay = any(key in lname for key in no_decay_keywords)
        if force_no_decay or param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    fused_ok = args.fused_adamw and torch.cuda.is_available() and get_fused_adamw_supported()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=fused_ok,
    )
    return optimizer


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.zeros(1, device=device, dtype=torch.float32)
    for g in grads:
        total += g.float().pow(2).sum()
    return float(total.sqrt().item())


# -----------------------------
# Data pipeline
# -----------------------------

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


def build_train_dataset(args, tokenizer, world_size: int, rank: int, skip_sequences_per_rank: int = 0):
    raw = load_dataset(args.train_data_path, args.data_name, split=args.train_split, streaming=True)
    raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    raw = raw.shard(num_shards=world_size, index=rank)

    remove_columns = raw.column_names
    dataset = raw.map(
        tokenize_and_group,
        batched=True,
        batch_size=args.tokenize_batch_size,
        remove_columns=remove_columns,
        fn_kwargs={
            "tokenizer": tokenizer,
            "block_size": args.max_position_embeddings,
            "add_bos": args.add_bos,
            "add_eos": args.add_eos,
        },
    )

    if skip_sequences_per_rank > 0:
        dataset = dataset.skip(skip_sequences_per_rank)

    return dataset


def build_valid_dataset(args, tokenizer, world_size: int, rank: int):
    raw = load_dataset(args.valid_data_path, args.data_name, split=args.valid_split, streaming=True)
    raw = raw.shard(num_shards=world_size, index=rank)

    remove_columns = raw.column_names
    dataset = raw.map(
        tokenize_and_group,
        batched=True,
        batch_size=args.tokenize_batch_size,
        remove_columns=remove_columns,
        fn_kwargs={
            "tokenizer": tokenizer,
            "block_size": args.max_position_embeddings,
            "add_bos": args.add_bos,
            "add_eos": args.add_eos,
        },
    )
    return dataset


# -----------------------------
# Model + checkpoint
# -----------------------------

@dataclass
class CheckpointState:
    step: int
    consumed_sequences_per_rank: int


def build_model(args) -> torch.nn.Module:
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.rms_norm_eps,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        head_dim=args.head_dim,
        attention_bias=args.attention_bias,
        mlp_bias=args.mlp_bias,
        tie_word_embeddings=args.tie_word_embeddings,
        use_cache=False,
    )

    if args.hidden_size != args.num_attention_heads * args.head_dim:
        raise ValueError(
            f"Inconsistent config: hidden_size={args.hidden_size} != num_attention_heads({args.num_attention_heads}) * head_dim({args.head_dim})."
        )

    if args.model_type == "llama":
        model = LlamaForCausalLM(config)
    elif args.model_type == "vscale":
        if VscaleForCausalLM is None:
            raise ImportError("modeling_vscale.VscaleForCausalLM is not available, but --model_type=vscale was requested.")
        config.model_type = "vscale"
        config.v_scale_learnable = args.v_scale_learnable
        model = VscaleForCausalLM(config)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.config._attn_implementation = args.attn_implementation
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


def save_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    state: CheckpointState,
    savefull: bool,
    args,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    payload = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if savefull else None,
        "scaler": scaler.state_dict(),
        "train_state": asdict(state),
        "args": vars(args),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    map_location: str = "cpu",
) -> CheckpointState:
    payload = torch.load(ckpt_path, map_location=map_location)
    unwrap_model(model).load_state_dict(payload["model"], strict=True)

    if "optimizer" not in payload:
        raise ValueError("Checkpoint does not contain optimizer state; exact resume is not possible.")
    optimizer.load_state_dict(payload["optimizer"])

    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])

    train_state = payload.get("train_state", None)
    if train_state is None:
        raise ValueError("Checkpoint missing train_state; cannot recover step/consumed data.")

    return CheckpointState(
        step=int(train_state["step"]),
        consumed_sequences_per_rank=int(train_state["consumed_sequences_per_rank"]),
    )


# -----------------------------
# Eval + logging
# -----------------------------

def evaluate(model: torch.nn.Module, valid_loader: DataLoader, device: torch.device, amp_dtype: Optional[torch.dtype], is_ddp: bool, max_batches: int) -> float:
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with make_autocast_context(amp_dtype):
                loss = model(input_ids=input_ids, labels=labels).loss
            loss_sum += float(loss.item())
            batch_count += 1

    total_loss_sum, total_batch_count = all_reduce_sum_count(loss_sum, batch_count, device, is_ddp)
    model.train()
    return total_loss_sum / max(1, total_batch_count)


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Train
# -----------------------------

def train(args) -> None:
    world_size, rank, local_rank, is_ddp, is_main, device = setup_distributed()
    seed_everything(args.seed, rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = int(1e8)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args)
    model.to(device)  # keep parameters in fp32; AMP handles low-precision forward

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = build_optimizer(model, args)
    amp_dtype = get_autocast_dtype(args.amp)
    scaler = grad_scaler_for_amp(amp_dtype)

    grad_accum_steps = args.global_batch_size // (args.local_batch_size * world_size)
    if args.global_batch_size % (args.local_batch_size * world_size) != 0:
        raise ValueError("global_batch_size must be divisible by local_batch_size * world_size.")
    local_sequences_per_step = args.local_batch_size * grad_accum_steps
    global_sequences_per_step = args.global_batch_size
    tokens_per_step_global = global_sequences_per_step * args.max_position_embeddings

    run_name = args.run_name or f"{args.model_type}_{args.hidden_size}d_{args.num_hidden_layers}L_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "tb"
    text_log_path = out_dir / "train_log.jsonl"
    cfg_path = out_dir / "config.json"

    writer = None
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        writer = SummaryWriter(log_dir=str(log_dir))

    start_step = 0
    consumed_sequences_per_rank = 0

    if args.resume:
        resume_state = load_checkpoint(args.resume, model, optimizer, scaler, map_location="cpu")
        start_step = resume_state.step
        consumed_sequences_per_rank = resume_state.consumed_sequences_per_rank
        if is_main:
            print(f"[Resume] step={start_step}, consumed_sequences_per_rank={consumed_sequences_per_rank}")

    barrier_if_needed(is_ddp)

    train_dataset = build_train_dataset(
        args=args,
        tokenizer=tokenizer,
        world_size=world_size,
        rank=rank,
        skip_sequences_per_rank=consumed_sequences_per_rank,
    )
    valid_dataset = build_valid_dataset(args=args, tokenizer=tokenizer, world_size=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.local_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.local_batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )

    if is_main:
        n_params = sum(p.numel() for p in unwrap_model(model).parameters())
        print(f"[Init] run_name={run_name}")
        print(f"[Init] device={device}, world_size={world_size}, amp={args.amp}, grad_accum_steps={grad_accum_steps}")
        print(f"[Init] params={n_params:,}, local_batch_size={args.local_batch_size}, global_batch_size={args.global_batch_size}")
        print(f"[Init] local_sequences_per_step={local_sequences_per_step}, tokens_per_step_global={tokens_per_step_global:,}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    running_loss = 0.0
    micro_step = 0
    step = start_step
    oom_count = 0

    train_iter = iter(train_loader)
    while step < args.max_iters:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        micro_step += 1
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        lr = cosine_lr(step, args.warmup_iters, args.max_iters, args.max_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            sync_context = model.no_sync() if (is_ddp and micro_step < grad_accum_steps) else contextlib.nullcontext()
            with sync_context:
                with make_autocast_context(amp_dtype):
                    loss = model(input_ids=input_ids, labels=labels).loss
                    loss = loss / grad_accum_steps

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss detected: {float(loss.item())}")

                scaler.scale(loss).backward()
                running_loss += float(loss.item())

        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if is_main:
                print(f"[OOM] micro_step skipped. oom_count={oom_count}")
            micro_step = 0
            continue

        if micro_step < grad_accum_steps:
            continue

        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        micro_step = 0
        consumed_sequences_per_rank += local_sequences_per_step

        display_loss = all_reduce_mean(running_loss, device, is_ddp)
        running_loss = 0.0

        if is_main and step % args.log_interval == 0:
            elapsed = time.time() - t0
            tokens_seen_global = step * tokens_per_step_global
            record = {
                "step": step,
                "train_loss": display_loss,
                "lr": lr,
                "grad_norm": float(gnorm),
                "tokens_seen_global": int(tokens_seen_global),
                "elapsed_sec": elapsed,
            }
            append_jsonl(text_log_path, record)
            if writer is not None:
                writer.add_scalar("train/loss", display_loss, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", float(gnorm), step)
                writer.add_scalar("train/tokens_seen_global", tokens_seen_global, step)
            
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            approx_total_str = str(datetime.timedelta(seconds=int((elapsed / step) * args.max_iters)))
            print(
                f"[Training] <{elapsed_str}/{approx_total_str}> "
                f"step={step}/{args.max_iters} loss={display_loss:.4f} lr={lr:.3e} "
                f"gnorm={float(gnorm):.4f} tokens={tokens_seen_global:,}"
            )

        if step % args.valid_interval == 0:
            val_loss = evaluate(
                model=model,
                valid_loader=valid_loader,
                device=device,
                amp_dtype=amp_dtype,
                is_ddp=is_ddp,
                max_batches=args.valid_max_batches,
            )
            if is_main:
                ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                append_jsonl(
                    text_log_path,
                    {
                        "step": step,
                        "valid_loss": val_loss,
                        "perplexity": ppl,
                    },
                )
                if writer is not None:
                    writer.add_scalar("valid/loss", val_loss, step)
                    writer.add_scalar("valid/ppl", ppl, step)
                print(f"[Valid] step={step} val_loss={val_loss:.4f} ppl={ppl:.2f}")

        if is_main and step % args.save_interval == 0:
            state = CheckpointState(
                step=step,
                consumed_sequences_per_rank=consumed_sequences_per_rank,
            )
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            savefull = step % args.full_save_interval == 0
            save_checkpoint(ckpt_path, model, optimizer, scaler, state, savefull, args)
            print(f"[Checkpoint] saved to {ckpt_path}, save optimizer={savefull}")

    cleanup_distributed(is_ddp)


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Core run setup
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--out_dir", type=str, default="./runs")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--compile", type=str2bool, default=False)
    p.add_argument("--compile_mode", type=str, default="default")
    p.add_argument("--amp", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    # Model
    p.add_argument("--model_type", type=str, default="llama", choices=["llama", "vscale"])
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--intermediate_size", type=int, default=2816)
    p.add_argument("--num_hidden_layers", type=int, default=16)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--num_key_value_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--max_position_embeddings", type=int, default=2048)
    p.add_argument("--rms_norm_eps", type=float, default=1e-5)
    p.add_argument("--bos_token_id", type=int, default=50256)
    p.add_argument("--eos_token_id", type=int, default=50256)
    p.add_argument("--attention_bias", type=str2bool, default=False)
    p.add_argument("--mlp_bias", type=str2bool, default=False)
    p.add_argument("--tie_word_embeddings", type=str2bool, default=False)
    # Vscale
    p.add_argument("--v_scale_learnable", type=str2bool, default=True)

    # Data
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--train_data_path", type=str, required=True)
    p.add_argument("--valid_data_path", type=str, required=True)
    p.add_argument("--data_name", type=str, default="en")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--valid_split", type=str, default="validation")
    p.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    p.add_argument("--tokenize_batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--add_bos", type=str2bool, default=False)
    p.add_argument("--add_eos", type=str2bool, default=True)

    # Optimizer / schedule
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global_batch_size", type=int, default=512)
    p.add_argument("--local_batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=25_000)
    p.add_argument("--warmup_iters", type=int, default=1_000)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    # AdamW
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--fused_adamw", type=str2bool, default=True)
    p.add_argument(
        "--no_decay_keywords",
        type=str,
        default="bias,norm,rmsnorm,layernorm,vscale,v_scale,theta,lambda",
        help="Comma-separated name keywords that force a parameter into the no-decay group. 1D params are also no-decay by default.",
    )

    # Logging / eval / saving
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--valid_interval", type=int, default=200)
    p.add_argument("--valid_max_batches", type=int, default=32)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--full_save_interval", type=int, default=5000)
    
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
