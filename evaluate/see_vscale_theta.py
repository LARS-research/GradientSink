import os
import re
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# config
# =========================================================
CKPT_DIR = "train/runs/vscale_1B/checkpoints"
CKPT_GLOB = "step_*.pt"   # or "*.pt"
OUTDIR_BASE = "evaluate/outputs/vscale_ckpt_inspect"

FINAL_ONLY = True

#   "theta" -> theta
#   "C"     -> effective C
VALUE_MODE = "theta"

STEP_REGEX = r"step_(\d+)"

DPI = 320
HEATMAP_CMAP = "coolwarm"   # theta
POSITIVE_CMAP = "viridis"   # C
LINE_ALPHA = 0.95
LINE_WIDTH = 1.8
MARKER_SIZE = 3.0

SAVE_NPZ = True
SAVE_JSON = True

ADD_DATE_TAG = True

SAVE_FINAL_CSV = True

USE_MARKER = False

# =========================================================
try:
    from training_tools import load_checkpoint_obj as _custom_load_checkpoint_obj
except Exception:
    _custom_load_checkpoint_obj = None


def load_checkpoint_obj(path):
    if _custom_load_checkpoint_obj is not None:
        return _custom_load_checkpoint_obj(path)
    return torch.load(path, map_location="cpu")


# =========================================================
def extract_step_from_name(path: Path, regex=STEP_REGEX):
    m = re.search(regex, path.name)
    if m:
        return int(m.group(1))
    return None


def extract_model_state(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint object is not a dict, cannot extract model_state.")

    candidate_keys = ["model_state", "state_dict", "model", "module"]
    for k in candidate_keys:
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            vals = list(ckpt_obj[k].values())
            if len(vals) > 0 and isinstance(vals[0], torch.Tensor):
                return ckpt_obj[k]

    vals = list(ckpt_obj.values())
    if len(vals) > 0 and isinstance(vals[0], torch.Tensor):
        return ckpt_obj

    raise KeyError("Could not find model_state/state_dict in checkpoint.")


def to_1d_float_cpu(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.detach().float().cpu().reshape(-1)


def compute_effective_value(theta, c0, mode):
    """
    Return a 1D tensor with shape [num_heads].
    """
    theta = to_1d_float_cpu(theta) if theta is not None else None
    c0 = to_1d_float_cpu(c0) if c0 is not None else None

    if mode == "theta":
        if theta is None:
            raise ValueError("VALUE_MODE='theta' but theta is missing.")
        return theta

    if mode == "C":
        if c0 is None:
            raise ValueError("VALUE_MODE='C' but c0 is missing.")

        if theta is None:
            return c0

        c0_b, theta_b = torch.broadcast_tensors(c0, theta)

        value = c0_b * torch.exp(theta_b)

        return value.reshape(-1)

    raise ValueError(f"Unknown VALUE_MODE: {mode}")


def collect_vradial_from_model_state(model_state):
    layer_dict = {}

    theta_pat = re.compile(r"layers\.(\d+)\..*v_radial\.theta$")
    c0_pat = re.compile(r"layers\.(\d+)\..*v_radial\.c0$")

    for k, v in model_state.items():
        m_theta = theta_pat.search(k)
        if m_theta:
            layer_idx = int(m_theta.group(1))
            layer_dict.setdefault(layer_idx, {"theta": None, "c0": None})
            layer_dict[layer_idx]["theta"] = v
            continue

        m_c0 = c0_pat.search(k)
        if m_c0:
            layer_idx = int(m_c0.group(1))
            layer_dict.setdefault(layer_idx, {"theta": None, "c0": None})
            layer_dict[layer_idx]["c0"] = v
            continue

    if not layer_dict:
        raise KeyError("No v_radial.theta / v_radial.c0 found in model_state.")

    return layer_dict


def sort_and_find_ckpts(ckpt_dir, ckpt_glob):
    ckpt_dir = Path(ckpt_dir)
    paths = list(ckpt_dir.glob(ckpt_glob))
    if len(paths) == 0:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} with glob={ckpt_glob}")

    def sort_key(p):
        step = extract_step_from_name(p)
        step = step if step is not None else float("inf")
        return (step, p.name)

    paths = sorted(paths, key=sort_key)
    return paths


def collect_series(ckpt_paths, value_mode):
    """
    return:
        steps: list[int]
        data: np.ndarray [T, L, H]
        layer_indices: list[int]
        head_count: int
    """
    all_steps = []
    all_data = []
    canonical_layer_indices = None
    canonical_head_count = None
    if FINAL_ONLY:
        ckpt_paths = [ckpt_paths[-1]]
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"[{i+1}/{len(ckpt_paths)}] Loading {ckpt_path}")
        ckpt_obj = load_checkpoint_obj(str(ckpt_path))
        model_state = extract_model_state(ckpt_obj)
        layer_dict = collect_vradial_from_model_state(model_state)

        layer_indices = sorted(layer_dict.keys())

        per_layer_values = []
        per_layer_head_count = None

        for layer_idx in layer_indices:
            theta = layer_dict[layer_idx]["theta"]
            c0 = layer_dict[layer_idx]["c0"]

            value = compute_effective_value(theta, c0, value_mode)  # [H]
            value_np = value.numpy()

            if per_layer_head_count is None:
                per_layer_head_count = value_np.shape[0]
            elif per_layer_head_count != value_np.shape[0]:
                raise ValueError(
                    f"Inconsistent head count within checkpoint {ckpt_path}: "
                    f"got {value_np.shape[0]} vs previous {per_layer_head_count}"
                )

            per_layer_values.append(value_np)

        per_layer_values = np.stack(per_layer_values, axis=0)  # [L, H]

        if canonical_layer_indices is None:
            canonical_layer_indices = layer_indices
        elif canonical_layer_indices != layer_indices:
            raise ValueError(
                f"Layer indices mismatch.\n"
                f"Expected: {canonical_layer_indices}\n"
                f"Got:      {layer_indices}\n"
                f"Checkpoint: {ckpt_path}"
            )

        if canonical_head_count is None:
            canonical_head_count = per_layer_values.shape[1]
        elif canonical_head_count != per_layer_values.shape[1]:
            raise ValueError(
                f"Head count mismatch.\n"
                f"Expected: {canonical_head_count}\n"
                f"Got:      {per_layer_values.shape[1]}\n"
                f"Checkpoint: {ckpt_path}"
            )

        step = extract_step_from_name(ckpt_path)
        if step is None:
            raise ValueError(f"Could not parse step from filename: {ckpt_path.name}")

        all_steps.append(step)
        all_data.append(per_layer_values)

    data = np.stack(all_data, axis=0)  # [T, L, H]
    return all_steps, data, canonical_layer_indices, canonical_head_count


def plot_final_heatmap(steps, data, layer_indices, outdir, value_mode):
    """
    final heatmap: shape [L, H]
    """
    final_arr = data[-1]  # [L, H]

    n_layers, n_heads = final_arr.shape
    fig_w = 5
    fig_h = 5

    plt.figure(figsize=(fig_w, fig_h))

    if value_mode == "theta":
        vmax = np.max(np.abs(final_arr))
        vmin = -vmax
        cmap = HEATMAP_CMAP
        center = 0.0
    else:
        vmin = None
        vmax = None
        cmap = POSITIVE_CMAP
        center = None

    ax = sns.heatmap(
        final_arr,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        xticklabels=[f"{h}" for h in range(n_heads)],
        yticklabels=[f"{l}" for l in layer_indices],
    )

    final_step = steps[-1]
    if value_mode == "theta":
        ax.set_title(r"$\theta_{\ell,h}$" + f" Heatmap (step={final_step})")
    else:
        ax.set_title(f"{value_mode} Heatmap (step={final_step})")
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers")

    plt.tight_layout()
    save_path = Path(outdir) / f"final_heatmap_{value_mode}_step{final_step}.pdf"
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap to {save_path}")

    if SAVE_FINAL_CSV:
        csv_path = Path(outdir) / f"final_heatmap_{value_mode}_step{final_step}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            header = ["layer"] + [f"head_{h}" for h in range(n_heads)]
            f.write(",".join(header) + "\n")
            for i, layer_idx in enumerate(layer_indices):
                row = [str(layer_idx)] + [f"{x:.10g}" for x in final_arr[i]]
                f.write(",".join(row) + "\n")
        print(f"Saved final heatmap csv to {csv_path}")


def plot_per_layer_trajectories(steps, data, layer_indices, outdir, value_mode):
    steps = np.array(steps)
    _, n_layers, n_heads = data.shape

    for li in range(n_layers):
        layer_idx = layer_indices[li]
        layer_data = data[:, li, :]  # [T, H]

        plt.figure(figsize=(10, 5.5))

        for h in range(n_heads):
            y = layer_data[:, h]
            if USE_MARKER:
                plt.plot(
                    steps, y,
                    linewidth=LINE_WIDTH,
                    alpha=LINE_ALPHA,
                    marker="o",
                    markersize=MARKER_SIZE,
                    label=f"H{h}"
                )
            else:
                plt.plot(
                    steps, y,
                    linewidth=LINE_WIDTH,
                    alpha=LINE_ALPHA,
                    label=f"H{h}"
                )

        if value_mode == "theta":
            plt.axhline(0.0, linestyle="--", linewidth=1.2, color="gray", alpha=0.7)

        plt.xlabel("Step")
        plt.ylabel(value_mode)
        plt.title(f"Layer {layer_idx} {value_mode} Trajectories")
        plt.legend(ncol=min(4, n_heads), fontsize=9, frameon=False)
        plt.grid(True, alpha=0.22)
        plt.tight_layout()

        save_path = Path(outdir) / f"layer_{layer_idx:02d}_{value_mode}_trajectories.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close()

        print(f"Saved layer trajectory to {save_path}")


def save_raw_data(steps, data, layer_indices, outdir, value_mode):
    if SAVE_NPZ:
        npz_path = Path(outdir) / f"raw_series_{value_mode}.npz"
        np.savez_compressed(
            npz_path,
            steps=np.array(steps),
            data=data,
            layer_indices=np.array(layer_indices),
        )
        print(f"Saved npz to {npz_path}")

    if SAVE_JSON:
        meta = {
            "ckpt_dir": CKPT_DIR,
            "ckpt_glob": CKPT_GLOB,
            "value_mode": value_mode,
            "num_checkpoints": int(len(steps)),
            "steps": [int(x) for x in steps],
            "num_layers": int(data.shape[1]),
            "num_heads": int(data.shape[2]),
            "output_shape": list(map(int, data.shape)),  # [T, L, H]
        }
        json_path = Path(outdir) / "meta.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved meta json to {json_path}")


def main():
    sns.set_theme(style="whitegrid")

    outdir = os.path.join(OUTDIR_BASE, datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    print(f"Output dir: {outdir}")

    ckpt_paths = sort_and_find_ckpts(CKPT_DIR, CKPT_GLOB)

    print(f"Found {len(ckpt_paths)} checkpoints.")
    print("First few:")
    for p in ckpt_paths[:5]:
        print("  ", p.name)
    if len(ckpt_paths) > 5:
        print("  ...")
        print("  ", ckpt_paths[-1].name)

    steps, data, layer_indices, head_count = collect_series(
        ckpt_paths,
        value_mode=VALUE_MODE,
    )

    print(f"Collected data shape: {data.shape}  [T, L, H]")
    print(f"Layers: {layer_indices}")
    print(f"Heads : {head_count}")
    print(f"Steps : {steps[:5]} ... {steps[-5:] if len(steps) >= 5 else steps}")

    save_raw_data(
        steps=steps,
        data=data,
        layer_indices=layer_indices,
        outdir=outdir,
        value_mode=VALUE_MODE,
    )

    plot_final_heatmap(
        steps=steps,
        data=data,
        layer_indices=layer_indices,
        outdir=outdir,
        value_mode=VALUE_MODE,
    )

    plot_per_layer_trajectories(
        steps=steps,
        data=data,
        layer_indices=layer_indices,
        outdir=outdir,
        value_mode=VALUE_MODE,
    )

    print("Done.")


if __name__ == "__main__":
    main()
