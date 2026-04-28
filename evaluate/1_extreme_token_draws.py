import os
import re
from typing import Dict, List, Tuple

import torch

import matplotlib
matplotlib.use("Agg")

from draw_helper import *

# ============================================================
RUN_DIR = "evaluate/outputs/extreme_token_probe/1B_vscale"
OUT_DIR = None                     # None -> RUN_DIR/paper_plots
EXT = "pdf"                       # pdf / svg / png

EARLY_WINDOW = 16
EPS_KEY = "auto"                  # auto / sink_eps_layer_0p3_k / "0.3"

BASE_CMAP = "Blues"

DRAW_STEPS = [1000, 3000, 6000, 10000, 15000, 20000, 25000]         # [] for all


# ============================================================
def get_sorted_checkpoint_results(run_dir: str):
    ckpts_base = os.path.join(run_dir, "ckpts")
    all_dirs = [
        d for d in os.listdir(ckpts_base)
        if d.startswith("ckpt_") and os.path.isdir(os.path.join(ckpts_base, d))
    ]

    results = []
    for d in all_dirs:
        step_val = int(d.split("_")[1])
        if step_val not in DRAW_STEPS and len(DRAW_STEPS) > 0:
            continue
        full_path = os.path.join(ckpts_base, d)
        pt_path = os.path.join(full_path, "metrics.pt")
        if os.path.exists(pt_path):
            results.append((step_val, full_path))

    results.sort(key=lambda x: x[0])
    if not results:
        return [], []

    steps_sorted, ckpt_dirs_sorted = zip(*results)
    return list(ckpt_dirs_sorted), list(steps_sorted)


def load_metrics_list(run_dir: str):
    ckpt_dirs_sorted, steps_sorted = get_sorted_checkpoint_results(run_dir)
    if not ckpt_dirs_sorted:
        raise RuntimeError(f"No ckpt_*/metrics.pt found under: {run_dir}")

    metrics_all = []
    for ckpt_dir in ckpt_dirs_sorted:
        pt_path = os.path.join(ckpt_dir, "metrics.pt")
        metrics_all.append(torch.load(pt_path, map_location="cpu"))
    return metrics_all, steps_sorted


# ============================================================
def _parse_group_token(group: str):
    g = group.strip().lower()
    if g.startswith("tok"):
        return ("tok", int(g[3:]))
    if g == "mean_rest":
        return ("mean_rest", None)
    if g == "max_rest":
        return ("max_rest", None)
    raise ValueError(f"Unknown token group: {group}")


def _group_reduce_token_axis(mat_lt: torch.Tensor, group: str, token_max: int):
    """
    mat_lt: [L, T]
    return: [L]
    """
    kind, val = _parse_group_token(group)
    L, T = mat_lt.shape
    Tmax = min(int(token_max), int(T))

    if kind == "tok":
        t = int(val)
        t = max(0, min(t, Tmax - 1))
        return mat_lt[:, t]

    if Tmax <= 1:
        return mat_lt[:, 0]

    rest = mat_lt[:, 1:Tmax]

    if kind == "mean_rest":
        return rest.mean(dim=1)

    if kind == "max_rest":
        return rest.amax(dim=1)

    raise RuntimeError("unreachable")


def _group_reduce_key_axis(mat_kL: torch.Tensor, group: str, key_max: int):
    """
    mat_kL: [K, L]
    return: [L]
    """
    kind, val = _parse_group_token(group)
    K, L = mat_kL.shape
    Kmax = min(int(key_max), int(K))

    if kind == "tok":
        k = int(val)
        k = max(0, min(k, Kmax - 1))
        return mat_kL[k, :]

    if Kmax <= 1:
        return mat_kL[0, :]

    rest = mat_kL[1:Kmax, :]

    if kind == "mean_rest":
        return rest.mean(dim=0)

    if kind == "max_rest":
        return rest.amax(dim=0)

    raise RuntimeError("unreachable")


def build_group_lines_token(
    metrics_all: List[Dict],
    metric_key: str,
    groups: Tuple[str, ...],
    token_max: int
):
    group_lines = {g: [] for g in groups}
    for m in metrics_all:
        mat = m[metric_key].float()  # [L, T]
        for g in groups:
            vec = _group_reduce_token_axis(mat, g, token_max=token_max)
            group_lines[g].append(vec.tolist())
    return group_lines


def build_group_lines_key(
    metrics_all: List[Dict],
    metric_key: str,
    groups: Tuple[str, ...],
    key_max: int,
):
    group_lines = {g: [] for g in groups}
    for m in metrics_all:
        mat = m[metric_key].float()  # [K, L]
        for g in groups:
            vec = _group_reduce_key_axis(mat, g, key_max=key_max)
            group_lines[g].append(vec.tolist())
    return group_lines


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
def main():
    run_dir = RUN_DIR
    out_dir = OUT_DIR or os.path.join(run_dir, "paper_plots")
    os.makedirs(out_dir, exist_ok=True)

    metrics_all, steps_sorted = load_metrics_list(run_dir)
    ref = metrics_all[-1]

    if "x_out_l2_mean_lt" not in ref:
        raise RuntimeError("Missing x_out_l2_mean_lt in metrics.pt")

    L, T = ref["x_out_l2_mean_lt"].shape
    layer_x = list(range(L))

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

    key_max = min(int(ref["sink_mass_layer_k"].shape[0]), int(EARLY_WINDOW))
    token_max = min(int(T), int(EARLY_WINDOW))

    eps_key = canonicalize_eps_key(EPS_KEY, ref)

    as_mass_groups = build_group_lines_key(
        metrics_all=metrics_all,
        metric_key="sink_mass_layer_k",
        groups=groups,
        key_max=key_max,
    )

    as_eps_groups = build_group_lines_key(
        metrics_all=metrics_all,
        metric_key=eps_key,
        groups=groups,
        key_max=key_max,
    )

    ma_mlp_groups = None
    if "mlp_l2_mean_lt" in ref:
        ma_mlp_groups = build_group_lines_token(
            metrics_all=metrics_all,
            metric_key="mlp_l2_mean_lt",
            groups=groups,
            token_max=token_max,
        )

    ma_xout_groups = build_group_lines_token(
        metrics_all=metrics_all,
        metric_key="x_out_l2_mean_lt",
        groups=groups,
        token_max=token_max,
    )

    v_groups = None
    if "v_l2_mean_lt" in ref:
        v_groups = build_group_lines_token(
            metrics_all=metrics_all,
            metric_key="v_l2_mean_lt",
            groups=groups,
            token_max=token_max,
        )

    plot_ckpt_group_lines(
        out_path=os.path.join(out_dir, f"AS_mass_compare.{EXT}"),
        x=layer_x,
        group_lines=as_mass_groups,
        steps_sorted=steps_sorted,
        title="Attention Sink Mass",
        xlabel="Layer Index",
        ylabel="Sink Mass",
        cmap_name=BASE_CMAP,
        group_style=group_style,
        group_label_map=group_label_map,
        figsize=FIGSIZE_COMPARE,
    )

    plot_ckpt_group_lines(
        out_path=os.path.join(out_dir, f"AS_eps_compare.{EXT}"),
        x=layer_x,
        group_lines=as_eps_groups,
        steps_sorted=steps_sorted,
        title=r"$\mathrm{Sink}^{\epsilon}$",
        xlabel="Layer Index",
        ylabel=r"$\mathrm{Sink}^{\epsilon}$",
        cmap_name=BASE_CMAP,
        group_style=group_style,
        group_label_map=group_label_map,
        figsize=FIGSIZE_COMPARE,
    )

    if ma_mlp_groups is not None:
        plot_ckpt_group_lines(
            out_path=os.path.join(out_dir, f"MA_mlp_out_compare.{EXT}"),
            x=layer_x,
            group_lines=ma_mlp_groups,
            steps_sorted=steps_sorted,
            title="MLP Output Norm",
            xlabel="Layer Index",
            ylabel="Activation Norm",
            cmap_name=BASE_CMAP,
            group_style=group_style,
            group_label_map=group_label_map,
            figsize=FIGSIZE_COMPARE,
        )

    plot_ckpt_group_lines(
        out_path=os.path.join(out_dir, f"MA_x_out_compare.{EXT}"),
        x=layer_x,
        group_lines=ma_xout_groups,
        steps_sorted=steps_sorted,
        title="Residual-Stream Output Norm",
        xlabel="Layer Index",
        ylabel="Activation Norm",
        cmap_name=BASE_CMAP,
        group_style=group_style,
        group_label_map=group_label_map,
        figsize=FIGSIZE_COMPARE,
    )

    if v_groups is not None:
        plot_ckpt_group_lines(
            out_path=os.path.join(out_dir, f"V_norm_compare.{EXT}"),
            x=layer_x,
            group_lines=v_groups,
            steps_sorted=steps_sorted,
            title="Value Norm (per-head mean)",
            xlabel="Layer Index",
            ylabel="Value Norm (per-head mean)",
            cmap_name=BASE_CMAP,
            group_style=group_style,
            group_label_map=group_label_map,
            figsize=FIGSIZE_COMPARE,
        )

    save_step_colorbar(
        os.path.join(out_dir, f"extreme_steps.{EXT}"),
        0, max(steps_sorted), cmap_name=BASE_CMAP,
        figsize=COLORBAR_SIZE_COMPARE,
        cax_rect=COLORBAR_INNER_COMPARE,
        tick_values=list(range(0, max(steps_sorted)+1, 5000)),
        label="Checkpoint Step"
    )

    print(f"[done] Saved paper plots to: {out_dir}")


if __name__ == "__main__":
    main()
