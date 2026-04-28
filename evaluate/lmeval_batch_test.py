import os
import json
import gc
import inspect
import pandas as pd
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- config ----------
model_paths = [
    r"evaluate/models/base/1B_baseline",
    r"evaluate/models/bnb/1B_baseline_bnb",
    r"evaluate/models/base/1B_vscale",
    r"evaluate/models/bnb/1B_vscale_bnb",
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

FEW_SHOT = 0
DEVICE = "cuda"
BATCH_SIZE = 64
OUT_DIR = "evaluate/outputs/lm_eval_outputs/" + datetime.now().strftime("%Y%m%d%H%M%S")
OUT_CSV = os.path.join(OUT_DIR, "summary_results.csv")

EXTRA_MODEL_ARGS = "trust_remote_code=True"


# ---------- utils ----------
def get_simple_evaluate():
    import lm_eval
    if hasattr(lm_eval, "simple_evaluate"):
        return lm_eval.simple_evaluate
    from lm_eval import evaluator
    return evaluator.simple_evaluate


def filter_kwargs_by_signature(fn, kwargs: dict):
    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_varkw:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


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


def cleanup_cuda():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ---------- main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    simple_evaluate = get_simple_evaluate()
    results_summary = []

    for path in model_paths:
        model_name = os.path.basename(os.path.normpath(path))
        print(f"\n>>> Start evaluating: {model_name}\n    path: {path}")

        model_args_parts = [f"pretrained={path},dtype=auto"]
        if EXTRA_MODEL_ARGS:
            model_args_parts.append(EXTRA_MODEL_ARGS.strip().strip(","))
        model_args = ",".join(model_args_parts)

        kwargs = dict(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            num_fewshot=FEW_SHOT,
        )
        kwargs = filter_kwargs_by_signature(simple_evaluate, kwargs)

        try:
            results = simple_evaluate(**kwargs)
        except TypeError as e:
            if str(BATCH_SIZE).lower() == "auto":
                kwargs["batch_size"] = 1
                results = simple_evaluate(**kwargs)
            else:
                raise
        except Exception as e:
            print(f"{e}")
            raise
        
        raw_path = os.path.join(OUT_DIR, f"{model_name}_raw.json")

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
        cleanup_cuda()

    df = pd.DataFrame(results_summary)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 80)
    print("🎉 Evaluation complete!")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"📊 CSV summary saved to {OUT_CSV}")
    print(f"📄 Raw JSON saved to {OUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
