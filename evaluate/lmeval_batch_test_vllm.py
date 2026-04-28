import os
import json
import inspect
from datetime import datetime
from typing import List, Optional
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from lm_eval import tasks as lm_tasks


max_embedding = 2048

model_paths = [
    "evaluate/models/smoothquant/1B_baseline_smoothquant",
    "evaluate/models/gptq/1B_baseline_gptq",
    "evaluate/models/awq/1B_baseline_awq",
    "evaluate/models/smoothquant/1B_vscale_smoothquant",
    "evaluate/models/gptq/1B_vscale_gptq",
    "evaluate/models/awq/1B_vscale_awq",
]
tasks = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "copa",
    "hellaswag",
    "lambada_openai",
    "openbookqa",
    "piqa",
    "sciq",
    "wikitext",
    "winogrande",
]


tensor_parallel_size = 1
gpu_memory_utilization = 0.9
swap_space = 4
batch_size = 64
max_batch_size = 64
trust_remote_code = True

output_paths = (
    "evaluate/outputs/lm_eval_outputs/"
    + datetime.now().strftime("%Y%m%d%H%M%S")
)
os.makedirs(output_paths, exist_ok=True)
output_scv = os.path.join(output_paths, "summary_results.csv")


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



def safe_get(metrics: dict, keys):
    if not metrics:
        return None
    for k in keys:
        # match v0.3 format
        if k in metrics and metrics[k] is not None:
            return metrics[k]
        # match v0.4 format (no filter)
        base_k = k.split(",")[0]
        if base_k in metrics and metrics[base_k] is not None:
            return metrics[base_k]
    return None



def run_lm_eval_vllm(
    model_path: str,
    tasks: List[str],
    batch_size,
    max_batch_size: Optional[int] = None,
    seeds: Optional[dict] = None,
):
    simple_evaluate = _get_simple_evaluate()

    task_manager = lm_tasks.TaskManager(
        metadata={
            "pretrained": model_path,
        }
    )

    model_args = {
        "pretrained": model_path,
        "tokenizer": model_path,
        "dtype": "auto",
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_embedding,
        "swap_space": swap_space,
        # "quantization": "compressed_tensors",
    }

    kwargs = {
        "model": "vllm",
        "model_args": model_args,
        "tasks": tasks,
        "task_manager": task_manager,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "device": "cuda",
        "num_fewshot": 0,
    }

    kwargs = filter_kwargs_by_signature(simple_evaluate, kwargs)
    return simple_evaluate(**kwargs)


def main():
    results_summary = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"[Start] evaluating with vLLM: {model_name}")

        results = run_lm_eval_vllm(
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )

        raw_path = os.path.join(output_paths, f"{model_name}_raw.json")

        safe_results = {
            "results": results.get("results", {}),
            "versions": results.get("versions", {})
        }

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(safe_results, f, ensure_ascii=False, indent=2)

        model_metrics = {"model": model_name}
        task_results = results.get("results", {})

        for task_name in tasks:
            metrics = task_results.get(task_name, {})
            task_lower = task_name.lower()

            if "wiki" in task_lower:
                model_metrics[f"{task_name}_PPL"] = safe_get(metrics, ["word_perplexity,none", "word_perplexity", "perplexity,none", "perplexity"])
            elif "lambada" in task_lower:
                model_metrics[f"{task_name}_PPL"] = safe_get(metrics, ["perplexity,none", "perplexity"])
                model_metrics[f"{task_name}_Acc"] = safe_get(metrics, ["acc,none", "acc"])
            else:
                val_acc_norm = safe_get(metrics, ["acc_norm,none", "acc_norm"])
                val_acc = safe_get(metrics, ["acc,none", "acc"])
                model_metrics[f"{task_name}_Score"] = val_acc_norm if val_acc_norm is not None else val_acc

        results_summary.append(model_metrics)
        

    df = pd.DataFrame(results_summary)
    df.to_csv(output_scv, index=False)

    print("\n" + "=" * 80)
    print("🎉 Evaluation complete!")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"📊 CSV summary saved to {output_paths}")
    print(f"📄 Raw JSON saved to {output_paths}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
