import os
import numpy as np
import torch
import pandas as pd
import seaborn
import matplotlib.ticker as ticker

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, hsv_to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from typing import List
import colorsys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter

# === paper-figure defaults ===
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

FIGSIZE_SCATTER_GROUP = (4, 4)
COLORBAR_SIZE_SCATTER_GROUP = (0.8, 4)
COLORBAR_INNER_SCATTER_GROUP = (0.1, 0.11, 0.25, 0.83)  # [left, bottom, width, height] in figure fraction

FIGSIZE_LINE = (4, 3)
COLORBAR_SIZE_LINE = (0.8, 3)
COLORBAR_INNER_LINE = (0.1, 0.15, 0.25, 0.77)

FIGSIZE_COMPARE = (4, 4)
COLORBAR_SIZE_COMPARE = (0.8, 4)
COLORBAR_INNER_COMPARE = (0.1, 0.11, 0.25, 0.83)

EXPORT_DPI = 320


def make_step_cmap(cmap_name: str = None, base_color: str = None):
    """
      - use a named matplotlib cmap, e.g. 'Blues'
      - or build white -> base_color, e.g. base_color='#A82064'
    """
    if cmap_name is not None:
        return plt.get_cmap(cmap_name)
    if base_color is not None:
        return LinearSegmentedColormap.from_list(
            f"white_to_{str(base_color).replace('#', '')}",
            [(1, 1, 1), to_rgb(base_color)]
        )
    raise ValueError("Need either cmap_name or base_color.")


def save_step_colorbar(
    out_path: str,
    vmin: float, vmax: float,
    cmap_name: str = None,
    base_color: str = None,
    figsize=(0.6, 0.36),
    cax_rect=(0.1, 0.12, 0.3, 0.8),
    label: str = "Checkpoint Step",
    tick_values=None,
    tick_labelsize: int = 10,
    labelsize: int = 11,
):
    cmap = make_step_cmap(cmap_name=cmap_name, base_color=base_color)
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig = plt.figure(figsize=figsize)
    cax = fig.add_axes(cax_rect)

    cb = fig.colorbar(sm, cax=cax, orientation="vertical")

    if tick_values is not None:
        cb.set_ticks(tick_values)
        # Convert 5000 to '5k'; keep 0 as '0'.
        cb.set_ticklabels([f'{int(v/1000)}k' if v != 0 else '0' for v in tick_values])

    cb.set_label(label, fontsize=labelsize)
    cb.ax.tick_params(labelsize=tick_labelsize, length=2, pad=1)
    cb.outline.set_linewidth(0.6)

    fig.savefig(out_path, dpi=EXPORT_DPI)
    plt.close(fig)


# === color helpers ===
def rgb_to_hsv_tuple(rgb):
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r, g, b)

def hsv_to_rgb_tuple(hsv):
    h, s, v = hsv
    return colorsys.hsv_to_rgb(h, s, v)

def shift_color_hsv(rgba, dh=0.0, ds=0.0, dv=0.0):
    """Hue shift + saturation/value tweak. rgba can be length 3 or 4."""
    if len(rgba) == 4:
        r, g, b, a = rgba
    else:
        r, g, b = rgba
        a = 1.0
    h, s, v = rgb_to_hsv_tuple((float(r), float(g), float(b)))
    h = (h + dh) % 1.0
    s = float(np.clip(s * (1.0 + ds), 0.0, 1.0))
    v = float(np.clip(v * (1.0 + dv), 0.0, 1.0))
    rr, gg, bb = hsv_to_rgb_tuple((h, s, v))
    return (rr, gg, bb, a)


# === plot lines ===
def plot_ckpt_lines_colorbar(
    out_dir: str,
    xs: List[float],
    ys_by_ckpt: List[List[float]],
    steps: List[int],
    title: str,
    xlabel: str,
    ylabel: str,
    fname: str,
    cmap_name: str = "Blues",
    figsize: tuple = FIGSIZE_LINE,
    draw_colorbar: bool = True,
    use_sci_offset: bool = True,
):
    if not ys_by_ckpt:
        return

    x = np.asarray(xs)
    steps_np = np.asarray(steps, dtype=float)

    vmax = float(steps_np.max())
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=figsize)

    for y, st in zip(ys_by_ckpt, steps):
        yy = np.asarray(y, dtype=float)
        c = cmap(norm(float(st)))
        ax.plot(x, yy, color=c, linewidth=1.2, alpha=0.9)

    if use_sci_offset:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useOffset=True, useMathText=False)

    if draw_colorbar:
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label("checkpoint step")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))

    fig.subplots_adjust(left=0.15, right=1, top=0.92, bottom=0.15)
    # fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=EXPORT_DPI)
    plt.close(fig)


# === grouped scatters ===
def flatten_points(step_list, arr_S_L):
    S, L = arr_S_L.shape
    val_flat = arr_S_L.reshape(-1)
    step_flat = np.repeat(np.array(step_list), L)
    layer_flat = np.tile(np.arange(L), S)
    return val_flat, step_flat, layer_flat

def apply_log_filter(x_raw, y_raw, layer_raw, step_raw, use_log=True):
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    layer_raw = np.asarray(layer_raw, dtype=np.float64)
    step_raw = np.asarray(step_raw, dtype=np.float64)

    mask = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(layer_raw) & np.isfinite(step_raw)
    if use_log:
        mask = mask & (x_raw > 0) & (y_raw > 0)

    x = x_raw[mask]
    y = y_raw[mask]
    layer = layer_raw[mask]
    step = step_raw[mask]
    if use_log:
        x = np.log10(x)
        y = np.log10(y)
    return x, y, layer, step

def scatter_compare_groups(
    x_data, y_data, 
    steps, 
    group_labels,
    title, xlabel, ylabel, out_path,
    colors=['#A82064', '#6420A8', '#2064A8', '#20A864', '#64A820'],
    markers=['o', 's', 'p', 'v', '^'],
    use_log=True,
    figsize=FIGSIZE_SCATTER_GROUP,
    add_ref_vline_zero=False,
):
    """
    x_data: [G, S, L]
    y_data: [G, S, L]
    group_labels: length = G, like ['Tok 0', 'Tok 1', 'Early', 'Mid', 'Rest']
    """
    if x_data.ndim == 2:
        x_data = x_data.unsqueeze(1)
        y_data = y_data.unsqueeze(1)
    G = x_data.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    
    max_step = max(steps)
    
    for i in range(G):
        cmap_name = f"group_{i}_cmap"
        group_cmap = LinearSegmentedColormap.from_list(cmap_name, ['white', colors[i]])

        x_flat, s_flat, _ = flatten_points(steps, x_data[i])
        y_flat, _, _ = flatten_points(steps, y_data[i])
        
        xf, yf, _, sf = apply_log_filter(x_flat, y_flat, np.zeros_like(x_flat), s_flat)
        
        color_values = sf / max_step
        ax.scatter(
            xf, yf, 
            c=color_values,
            cmap=group_cmap,
            vmin=0.0, vmax=1.0,
            s=16 if i == 0 else 12,
            alpha=0.6, 
            marker=markers[i],
            edgecolors='none',
            zorder=G - i
        )

    if use_log:
        ax.set_xlabel(r"$\mathrm{log}_{10}$ " + f"{xlabel}")
        ax.set_ylabel(r"$\mathrm{log}_{10}$ " + f"{ylabel}")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if add_ref_vline_zero:
        ax.axvline(
            0.0,
            color="#A86420",
            linewidth=1.0,
            alpha=0.9,
            zorder=1,
        )
        ax.text(
            0.02,
            0.02,
            r"$\log_{10}1=0$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=8,
            color="#A86420",
        )
        
    ax.set_title(title)

    legend_handles = []
    for i in range(G):
        handle = Line2D([0], [0], 
                       marker=markers[i], 
                       color='w',
                       markerfacecolor=colors[i],
                       markersize=8,
                       label=group_labels[i],
                       markeredgecolor='none')
        legend_handles.append(handle)
    
    ax.legend(handles=legend_handles, loc='best', markerscale=1)

    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.subplots_adjust(left=0.15, right=0.98, top=0.94, bottom=0.11)
    # fig.tight_layout()
    fig.savefig(out_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print(f"[ok] Multi-group plot saved: {out_path}")


def plot_ckpt_group_lines(
    out_path: str,
    x,
    group_lines: dict,
    steps_sorted,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap_name: str,
    group_style: dict,
    group_label_map: dict = None,
    figsize=FIGSIZE_LINE,
):
    """
    group_lines[group] = list of y
        each y is length len(x), aligned with steps_sorted
    group_style[group] = {
        "dh": float, "ds": float, "dv": float,
        "ls": str, "lw": float, "alpha": float
    }
    group_label_map[group] = display name in legend
    """
    if group_label_map is None:
        group_label_map = {}

    fig, ax = plt.subplots(figsize=figsize)

    vmin, vmax = 0, float(max(steps_sorted))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # tok0 first for prominence
    groups_order = list(group_lines.keys())
    if "tok0" in groups_order:
        groups_order.remove("tok0")
        groups_order = ["tok0"] + groups_order

    handles = []
    for g in groups_order:
        ys_list = group_lines[g]
        style = group_style.get(g, {})

        dh = float(style.get("dh", 0.0))
        ds = float(style.get("ds", 0.0))
        dv = float(style.get("dv", 0.0))
        ls = style.get("ls", "-")
        lw = float(style.get("lw", 1.2))
        alpha = float(style.get("alpha", 0.9))

        handles.append(
            Line2D(
                [0], [0],
                color=shift_color_hsv(cmap(1.0), dh=dh, ds=ds, dv=dv),
                linewidth=lw,
                linestyle=ls,
                label=group_label_map.get(g, g),
            )
        )

        for step, y in zip(steps_sorted, ys_list):
            yy = np.asarray(y, dtype=float)

            yy = np.where(np.isfinite(yy), yy, np.nan)

            base = cmap(norm(float(step)))
            col = shift_color_hsv(base, dh=dh, ds=ds, dv=dv)
            ax.plot(x, yy, linestyle=ls, linewidth=lw, alpha=alpha, color=col)

    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim([min(x)-0.5, max(x)+0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)

    fig.subplots_adjust(left=0.17, right=1, top=0.94, bottom=0.11)
    # fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=EXPORT_DPI)
    plt.close(fig)



# Violin
def draw_violin(raw_data_LT: torch.Tensor, n_tokens, x_label, y_label, title, outpath):
    raw_slice = raw_data_LT[:, 0:n_tokens].clamp_min(1e-30)   # [L, T]
    L, T = raw_slice.shape
    data_np = torch.log10(raw_slice).detach().cpu().numpy()  

    # ---------- ratio: token 0 / mean(other tokens), computed per layer ----------
    if n_tokens >= 2:
        rest_mean = raw_slice[:, 1:].mean(dim=1)                      # [L]
        ratio_per_layer = raw_slice[:, 0] / rest_mean                 # [L]
        median_ratio = ratio_per_layer.median().item()
        q1_ratio = torch.quantile(ratio_per_layer, 0.25).item()
        q3_ratio = torch.quantile(ratio_per_layer, 0.75).item()

    # --- build dataframe ---
    df = pd.DataFrame(data_np, columns=list(range(n_tokens)))
    df["Layer"] = np.arange(L)
    df["Depth_Pct"] = (df["Layer"] / (L - 1)) * 100 
    
    df_melted = df.melt(id_vars=["Layer", "Depth_Pct"], var_name="Token_Index", value_name="Log_Norm")
    df_melted["Token_Index"] = df_melted["Token_Index"].astype(int)

    order = list(range(n_tokens))

    highlight_color = "#F84210"   # token 0
    other_color = "#9B57DF"       # all other tokens
    palette = [highlight_color] + [other_color] * (n_tokens - 1)

    fig = plt.figure(figsize=(7.5, 3))
    ax = seaborn.violinplot(
        x="Token_Index", y="Log_Norm", hue="Token_Index",
        data=df_melted, order=order, palette=palette,
        inner=None,
        cut=0, linewidth=0, saturation=0.5,
        legend=False, bw_adjust=0.8
    )

    dot_cmap = "mako"
    seaborn.stripplot(
        x='Token_Index', y='Log_Norm', data=df_melted,
        hue='Depth_Pct', palette=dot_cmap,
        size=4, alpha=0.5, jitter=True, ax=ax, dodge=False
    )

    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.grid(axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_title(title)

    ax_ins = inset_axes(ax, width="25%", height="3%", loc='upper right', 
                        bbox_to_anchor=(0, -0.06, 0.98, 1),
                        bbox_transform=ax.transAxes)
    cb = fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap=dot_cmap),
                      cax=ax_ins, orientation='horizontal', ticklocation='top')
        
    ax_ins.tick_params(axis='x', which='both', length=0, pad=2)
    cb.set_ticks([0, 100])
    cb.set_ticklabels(['0', '100'], fontsize=9)
    cb.outline.set_linewidth(0.0)

    ax_ins.set_title("Layer Depth (%)", fontsize=10, y=0.8)

    if n_tokens >= 2:
        text = rf"Token 0 / mean(others): median=${median_ratio:.1f}$, $\mathrm{{IQR}}=[{q1_ratio:.1f},\,{q3_ratio:.1f}]$"
        ax.text(0.98, 0.86, text, transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if n_tokens > 24:
        tick_step = 2
        ax.set_xticks(list(range(0, n_tokens, tick_step)))
        ax.set_xticklabels(list(range(0, n_tokens, tick_step)))

    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=1, top=0.92, bottom=0.14)
    plt.savefig(outpath, dpi=EXPORT_DPI)
