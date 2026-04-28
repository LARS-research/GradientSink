import argparse
import json
import os
import inspect
from lm_eval import tasks as lm_tasks
from datetime import datetime
from typing import List, Any
from statistics import mean, stdev
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def filter_kwargs_by_signature(fn, kwargs: dict):
    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_varkw:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _get_simple_evaluate():
    import lm_eval
    if hasattr(lm_eval, "simple_evaluate"):
        return lm_eval.simple_evaluate
    from lm_eval import evaluator
    return evaluator.simple_evaluate


def _get_hflm_class():
    try:
        from lm_eval.models.huggingface import HFLM
        return HFLM
    except Exception:
        pass
    try:
        from lm_eval.models.hf import HFLM
        return HFLM
    except Exception:
        pass
    raise ImportError("Cannot import HFLM from lm_eval. Please check your lm-eval installation.")


def _build_hflm(model, tokenizer, batch_size, device, max_length):
    HFLM = _get_hflm_class()
    kwargs = {
        "pretrained": model,
        "tokenizer": tokenizer,
        "batch_size": batch_size,
        "device": device,
        "backend": "causal",
        "trust_remote_code": True,
        "max_length": max_length,
    }
    kwargs = filter_kwargs_by_signature(HFLM, kwargs)
    return HFLM(**kwargs)


def run_niah_eval(
    model,
    tokenizer,
    model_path: str,
    task_name: str,
    max_lengths: List[int],
    batch_size,
    device: str,
    seed: int=42,
):
    from lm_eval import tasks as lm_tasks

    simple_evaluate = _get_simple_evaluate()
    lm = _build_hflm(model=model, tokenizer=tokenizer, 
                     batch_size=batch_size, device=device, 
                     max_length=max(max_lengths))

    task_manager = lm_tasks.TaskManager(
        metadata={
            "max_seq_lengths": max_lengths,
            "pretrained": model_path,
        }
    )

    kwargs = {
        "model": lm,
        "tasks": [task_name],
        "task_manager": task_manager,
        "batch_size": batch_size,
        "device": device,
        "num_fewshot": 0,
        "random_seed": seed,
        "numpy_random_seed": seed,
        "torch_random_seed": seed,
        "fewshot_random_seed": seed,
    }
    kwargs = filter_kwargs_by_signature(simple_evaluate, kwargs)
    return simple_evaluate(**kwargs)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model-paths", type=str, required=True)
    p.add_argument("--output-path", type=str, default="evaluate/outputs/lm_eval_outputs")
    p.add_argument("--origin-max-length", type=int, default=2048)
    p.add_argument("--yarn-factor", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=32)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tasks", type=str, default="niah_single_2,niah_single_3,niah_multikey_1")
    p.add_argument("--max-lengths", type=str, default="2048")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    return p


def main():
    def parse_csv_list(s: str, cast=str) -> List[Any]:
        if s is None or s.strip() == "":
            return []
        return [cast(x.strip()) for x in s.split(",") if x.strip()]
    args = build_argparser().parse_args()
    model_paths = parse_csv_list(args.model_paths, str)
    output_path = os.path.join(args.output_path, datetime.now().strftime("%Y%m%d%H%M%S"))
    yarn_factor = args.yarn_factor
    batch_size = args.batch_size
    device = args.device
    tasks = parse_csv_list(args.tasks, str)
    max_lengths = parse_csv_list(args.max_lengths, int)
    seeds = parse_csv_list(args.seeds, int)

    os.makedirs(output_path, exist_ok=True)


    for model_path in model_paths:
        all_results = {}
        model_name = os.path.basename(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if yarn_factor > 1:
            config.rope_scaling = {
                "rope_type": "yarn",
                "factor": yarn_factor,
                "original_max_position_embeddings": args.origin_max_length,
                # "attention_factor": 1.0,
                # "beta_fast": 32,
                # "beta_slow": 1,
            }

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype="auto",
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

        for task in tasks:
            all_results[task] = {}

            for length in max_lengths:
                metric_key = f"{length},none"
                print(f"\n[Start] evaluating {model_name} on {task} with length={length}")

                seed_results = {}
                acc_list = []

                for seed in seeds:
                    print(f"  - seed = {seed}")
                    result = run_niah_eval(
                        model=model,
                        tokenizer=tokenizer,
                        model_path=model_path,
                        task_name=task,
                        max_lengths=[length],
                        batch_size=batch_size,
                        device=device,
                        seed=seed,
                    )
                    acc = result["results"][task][metric_key]
                    seed_results[str(seed)] = acc
                    acc_list.append(acc)

                    print(f"    acc = {acc:.6f}")

                mean_acc = float(mean(acc_list))
                std_acc = float(stdev(acc_list)) if len(acc_list) > 1 else 0.0

                summary = {
                    "per_seed": seed_results,
                    "mean": mean_acc,
                    "std": std_acc,
                    "n": len(acc_list),
                }

                all_results[task][metric_key] = summary
                print(
                    f"[Done] {model_name} | {task} | {metric_key} | "
                    f"mean = {mean_acc:.6f}, std = {std_acc:.6f}, n = {len(acc_list)}"
                )

        with open(os.path.join(output_path, f"{model_name}.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
