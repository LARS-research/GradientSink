import os
from typing import Dict, List, Tuple, Callable

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# ============================================================
MODEL_TAG = "03B"   # 01B / 03B / 1B
MODEL_TITLE = "0.3B"

BASELINE_RUN_DIR = f"evaluate/outputs/extreme_token_probe/{MODEL_TAG}_baseline"
VSCALE_RUN_DIR   = f"evaluate/outputs/extreme_token_probe/{MODEL_TAG}_vscale"

OUT_DIR = f"evaluate/outputs/extreme_token_probe/{MODEL_TAG}_compare"      # None -> <baseline parent>/compare_final_plots/<MODEL_TAG>
EXT = "pdf"         # pdf / png / svg

EARLY_WINDOW = 16
EPS_KEY = "auto"    # auto / sink_eps_layer_0p3_k / "0.3"

# Colors and styles
COLOR_BASELINE = "#0369cf"
COLOR_VSCALE   = "#fc9630"

MARKER_BASELINE = "o"
MARKER_VSCALE   = "x"

LINESTYLE_TOK0  = "-"
LINESTYLE_REST  = "--"

LW_TOK0 = 2.0
LW_REST = 1.5
MS_BASELINE = 5
MS_VSCALE = 5

ALPHA_TOK0 = 1.0
ALPHA_REST = 0.9

FIGSIZE = (4, 4)


# ============================================================
# checkpoint / metrics read
# ============================================================
def get_sorted_checkpoint_results(run_dir: str) -> Tuple[List[str], List[int]]:
    ckpts_base = os.path.join(run_dir, "ckpts")
    if not os.path.isdir(ckpts_base):
        raise RuntimeError(f"Missing ckpts dir: {ckpts_base}")

    all_dirs = [
        d for d in os.listdir(ckpts_base)
        if d.startswith("ckpt_") and os.path.isdir(os.path.join(ckpts_base, d))
    ]

    results = []
    for d in all_dirs:
        try:
            step_val = int(d.split("_")[1])
        except Exception:
            continue

        full_path = os.path.join(ckpts_base, d)
        pt_path = os.path.join(full_path, "metrics.pt")
        if os.path.exists(pt_path):
            results.append((step_val, full_path))

    results.sort(key=lambda x: x[0])
    if not results:
        raise RuntimeError(f"No ckpt_*/metrics.pt found under: {run_dir}")

    steps_sorted, ckpt_dirs_sorted = zip(*results)
    return list(ckpt_dirs_sorted), list(steps_sorted)


def load_final_metrics(run_dir: str) -> Tuple[Dict, int]:
    ckpt_dirs_sorted, steps_sorted = get_sorted_checkpoint_results(run_dir)
    last_ckpt_dir = ckpt_dirs_sorted[-1]
    last_step = steps_sorted[-1]
    pt_path = os.path.join(last_ckpt_dir, "metrics.pt")
    metrics = torch.load(pt_path, map_location="cpu")
    return metrics, last_step


# ============================================================
# group reduce
# ============================================================
def _group_reduce_token_axis(mat_lt: torch.Tensor, group: str, token_max: int) -> torch.Tensor:
    """
    mat_lt: [L, T]
    return: [L]
    """
    assert mat_lt.ndim == 2, f"Expect [L, T], got shape={tuple(mat_lt.shape)}"
    L, T = mat_lt.shape
    Tmax = min(int(token_max), int(T))
    if Tmax <= 0:
        raise ValueError(f"Invalid Tmax={Tmax} for shape={tuple(mat_lt.shape)}")

    if group == "tok0":
        return mat_lt[:, 0]

    if group == "mean_rest":
        if Tmax <= 1:
            return mat_lt[:, 0]
        return mat_lt[:, 1:Tmax].mean(dim=1)

    raise ValueError(f"Unknown token group: {group}")


def _group_reduce_key_axis(mat_kL: torch.Tensor, group: str, key_max: int) -> torch.Tensor:
    """
    mat_kL: [K, L]
    return: [L]
    """
    assert mat_kL.ndim == 2, f"Expect [K, L], got shape={tuple(mat_kL.shape)}"
    K, L = mat_kL.shape
    Kmax = min(int(key_max), int(K))
    if Kmax <= 0:
        raise ValueError(f"Invalid Kmax={Kmax} for shape={tuple(mat_kL.shape)}")

    if group == "tok0":
        return mat_kL[0, :]

    if group == "mean_rest":
        if Kmax <= 1:
            return mat_kL[0, :]
        return mat_kL[1:Kmax, :].mean(dim=0)

    raise ValueError(f"Unknown key group: {group}")


# ============================================================
# eps key
# ============================================================
def discover_eps_keys(ref_metric: Dict) -> List[str]:
    keys = [k for k in ref_metric.keys() if k.startswith("sink_eps_layer_") and k.endswith("_k")]
    return sorted(keys)


def canonicalize_eps_key(user_eps_key: str, ref_metric: Dict) -> str:
    eps_keys = discover_eps_keys(ref_metric)
    if not eps_keys:
        raise RuntimeError("No sink_eps_layer_*_k found in metrics.pt")

    if user_eps_key == "auto":
        return eps_keys[0]

    if user_eps_key in ref_metric:
        return user_eps_key

    s = str(user_eps_key).strip().replace(".", "p")
    k = f"sink_eps_layer_{s}_k"
    if k in ref_metric:
        return k

    raise KeyError(f"Cannot resolve eps key from: {user_eps_key}. Available: {eps_keys}")


# ============================================================
# metric extract & group reduce
# ============================================================
def get_metric_tensor(metrics: Dict, metric_name: str) -> torch.Tensor:
    if metric_name not in metrics:
        raise KeyError(f"Missing metric key: {metric_name}")
    return metrics[metric_name].float()


def get_sink_eps_tensor(metrics: Dict) -> torch.Tensor:
    eps_key = canonicalize_eps_key(EPS_KEY, metrics)
    return metrics[eps_key].float()


def extract_group_lines(
    metrics: Dict,
    metric_getter: Callable[[Dict], torch.Tensor],
    axis_kind: str,
    early_window: int,
) -> Dict[str, torch.Tensor]:
    """
    return:
        {
            "tok0": [L],
            "mean_rest": [L],
        }
    """
    mat = metric_getter(metrics).float()

    if axis_kind == "token":
        L, T = mat.shape
        token_max = min(int(T), int(early_window))
        return {
            "tok0": _group_reduce_token_axis(mat, "tok0", token_max),
            "mean_rest": _group_reduce_token_axis(mat, "mean_rest", token_max),
        }

    if axis_kind == "key":
        K, L = mat.shape
        key_max = min(int(K), int(early_window))
        return {
            "tok0": _group_reduce_key_axis(mat, "tok0", key_max),
            "mean_rest": _group_reduce_key_axis(mat, "mean_rest", key_max),
        }

    raise ValueError(f"Unknown axis_kind: {axis_kind}")


# ============================================================
# plot helper
# ============================================================
def make_output_dir() -> str:
    if OUT_DIR is not None:
        os.makedirs(OUT_DIR, exist_ok=True)
        return OUT_DIR

    parent = os.path.dirname(BASELINE_RUN_DIR.rstrip("/"))
    out_dir = os.path.join(parent, "compare_final_plots", MODEL_TAG)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def style_axis(ax):
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_alpha(0.65)


def add_compact_legend(ax):
    handles = [
        Line2D(
            [0], [0],
            color=COLOR_BASELINE,
            marker=MARKER_BASELINE,
            linestyle="None",
            markersize=MS_BASELINE + 0.8,
            label="Baseline",
        ),
        Line2D(
            [0], [0],
            color=COLOR_VSCALE,
            marker=MARKER_VSCALE,
            linestyle="None",
            markersize=MS_VSCALE + 0.5,
            markeredgewidth=1.6,
            label="V-scale",
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle=LINESTYLE_TOK0,
            linewidth=LW_TOK0,
            label="token 0",
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle=LINESTYLE_REST,
            linewidth=LW_REST,
            label="mean rest",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="best",
        frameon=True,
        framealpha=0.92,
        fontsize=9,
        ncol=1,
        borderpad=0.35,
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=0.35,
    )


def plot_metric_compare_for_pair(
    out_path: str,
    baseline_metrics: Dict,
    vscale_metrics: Dict,
    metric_getter: Callable[[Dict], torch.Tensor],
    axis_kind: str,
    title: str,
    ylabel: str,
):
    baseline_lines = extract_group_lines(
        metrics=baseline_metrics,
        metric_getter=metric_getter,
        axis_kind=axis_kind,
        early_window=EARLY_WINDOW,
    )
    vscale_lines = extract_group_lines(
        metrics=vscale_metrics,
        metric_getter=metric_getter,
        axis_kind=axis_kind,
        early_window=EARLY_WINDOW,
    )

    Lb = baseline_lines["tok0"].numel()
    Lv = vscale_lines["tok0"].numel()
    if Lb != Lv:
        raise RuntimeError(f"Layer number mismatch: baseline={Lb}, vscale={Lv}")

    x = list(range(Lb))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # baseline
    ax.plot(
        x, baseline_lines["tok0"].tolist(),
        color=COLOR_BASELINE,
        linestyle=LINESTYLE_TOK0,
        linewidth=LW_TOK0,
        marker=MARKER_BASELINE,
        markersize=MS_BASELINE,
        alpha=ALPHA_TOK0,
    )
    ax.plot(
        x, baseline_lines["mean_rest"].tolist(),
        color=COLOR_BASELINE,
        linestyle=LINESTYLE_REST,
        linewidth=LW_REST,
        marker=MARKER_BASELINE,
        markersize=MS_BASELINE,
        alpha=ALPHA_REST,
    )

    # v-scale learn
    ax.plot(
        x, vscale_lines["tok0"].tolist(),
        color=COLOR_VSCALE,
        linestyle=LINESTYLE_TOK0,
        linewidth=LW_TOK0,
        marker=MARKER_VSCALE,
        markersize=MS_VSCALE,
        markeredgewidth=1.4,
        alpha=ALPHA_TOK0,
    )
    ax.plot(
        x, vscale_lines["mean_rest"].tolist(),
        color=COLOR_VSCALE,
        linestyle=LINESTYLE_REST,
        linewidth=LW_REST,
        marker=MARKER_VSCALE,
        markersize=MS_VSCALE,
        markeredgewidth=1.4,
        alpha=ALPHA_REST,
    )

    style_axis(ax)
    ax.set_title(f"{title} ({MODEL_TITLE})", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim([min(x)-0.5, max(x)+0.5])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(ylabel)

    add_compact_legend(ax)

    fig.subplots_adjust(left=0.17, right=1, top=0.94, bottom=0.11)
    # fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=320)
    plt.close(fig)

    print(f"[done] saved: {out_path}")


# ============================================================
# main
# ============================================================
def main():
    out_dir = make_output_dir()

    baseline_metrics, baseline_step = load_final_metrics(BASELINE_RUN_DIR)
    vscale_metrics, vscale_step = load_final_metrics(VSCALE_RUN_DIR)

    print(f"[info] baseline final step = {baseline_step}")
    print(f"[info] vscale   final step = {vscale_step}")

    metric_specs = [
        {
            "filename": f"{MODEL_TAG}_AS_mass_final_compare.{EXT}",
            "title": "Attention Sink Mass",
            "ylabel": "Sink Mass",
            "axis_kind": "key",
            "metric_getter": lambda m: get_metric_tensor(m, "sink_mass_layer_k"),
        },
        {
            "filename": f"{MODEL_TAG}_AS_eps_final_compare.{EXT}",
            "title": r"$\mathrm{Sink}^{\epsilon}$",
            "ylabel": r"$\mathrm{Sink}^{\epsilon}$",
            "axis_kind": "key",
            "metric_getter": get_sink_eps_tensor,
        },
        {
            "filename": f"{MODEL_TAG}_MA_attn_out_final_compare.{EXT}",
            "title": "Attention Output Norm",
            "ylabel": "Activation Norm",
            "axis_kind": "token",
            "metric_getter": lambda m: get_metric_tensor(m, "attn_l2_mean_lt"),
        },
        {
            "filename": f"{MODEL_TAG}_MA_mlp_out_final_compare.{EXT}",
            "title": "MLP Output Norm",
            "ylabel": "Activation Norm",
            "axis_kind": "token",
            "metric_getter": lambda m: get_metric_tensor(m, "mlp_l2_mean_lt"),
        },
        {
            "filename": f"{MODEL_TAG}_MA_x_out_final_compare.{EXT}",
            "title": "Residual-Stream Output Norm",
            "ylabel": "Activation Norm",
            "axis_kind": "token",
            "metric_getter": lambda m: get_metric_tensor(m, "x_out_l2_mean_lt"),
        },
    ]

    for spec in metric_specs:
        plot_metric_compare_for_pair(
            out_path=os.path.join(out_dir, spec["filename"]),
            baseline_metrics=baseline_metrics,
            vscale_metrics=vscale_metrics,
            metric_getter=spec["metric_getter"],
            axis_kind=spec["axis_kind"],
            title=spec["title"],
            ylabel=spec["ylabel"],
        )

    print(f"[done] all figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
