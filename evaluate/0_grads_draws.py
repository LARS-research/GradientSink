import glob
import os
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from draw_helper import *

DIV_EPS = 1e-30

DRAW_SCATTER_GROUP = True
DRAW_LINE = True

EXT = "pdf"

EARLY_TOKENS = 16


# =========================
# config
# =========================
device = "cpu"
collects_path = "evaluate/outputs/ckpt_grads/1B_baseline"

model_name = "1B Baseline"

USE_LOG = True

# HSV encoding:
# layer -> hue in [0, HUE_MAX], step -> saturation in [0,1], value=VALUE
HUE_MAX = 0.8
VALUE = 0.8


# optional: draw lag plots & trajectory lines
DRAW_TRAJECTORY = True   # if True: connect points by step for each layer (still scatter-based)

# =========================
# helpers
# =========================
def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = DIV_EPS) -> torch.Tensor:
    return a / (b + eps)


def group_tensors(x:torch.Tensor):
    """ x: [L,T] size """
    x0 = x[:, 0]
    x1 = x[:, 1]
    x23 = x[:, 2:4].mean(dim=-1)
    x47 = x[:, 4:8].mean(dim=-1)
    xr = x[:, 8:EARLY_TOKENS].mean(dim=-1)
    # all [L]
    return torch.stack([x0, x1, x23, x47, xr], dim=0)

# =========================
# load ckpts
# =========================
folders = glob.glob(os.path.join(collects_path, "ckpts", "ckpt_*"))
ckpts = []
max_step = 0
for f in folders:
    if os.path.isdir(f):
        base = os.path.basename(f)
        try:
            step = int(base.split("_")[-1])
            max_step = max(max_step, step)
            ckpts.append((step, f))
        except ValueError:
            pass
ckpts.sort(key=lambda x: x[0])
if not ckpts:
    raise RuntimeError(f"No ckpt folders under {collects_path}")

steps = []

x_in_SL, x_half_SL = [], []
compress_attn_SL, compress_mlp_SL = [], []
pressure_attn_SL, pressure_mlp_SL = [], []
bloat_attn_SL, bloat_mlp_SL = [], []
change_attn_SL, change_mlp_SL = [], []
D_k_SL, D_v_SL = [], []
D_attn_norm_out_ST = []
D_q_ST, D_k_ST, D_v_ST = [], [], []

drawing_steps = [1000, 3000, 6000, 10000, 15000, 20000, 25000]
for step, ckpt_dir in ckpts:
    if step not in drawing_steps:
        continue
    print(f"checkpoint at {step} processing...")

    pt_path = os.path.join(ckpt_dir, "collects.pt")
    if not os.path.exists(pt_path):
        print(f"[skip] missing {pt_path}")
        continue

    raw = torch.load(pt_path, map_location=device)
    C = raw["collects"]
    meta = raw.get("meta", {})
    # dict_keys(['D_q_sum', 'D_k_sum', 'D_v_sum', 'D_x_in_sum', 'D_attn_norm_out_sum', 'D_x_half_sum', 'D_mlp_norm_out_sum', 'D_x_out_sum', 'D_attn_branch_sum', 'D_x_out_like_sum', 'D_mlp_branch_sum', 'D_q_norm', 'D_k_norm', 'D_v_norm', 'D_x_in_norm', 'D_attn_norm_out_norm', 'D_x_half_norm', 'D_mlp_norm_out_norm', 'D_x_out_norm', 'D_attn_branch_norm', 'D_x_out_like_norm', 'D_mlp_branch_norm', 'x_in_l2', 'x_half_l2', 'x_out_l2', 'attn_out_l2', 'mlp_out_l2']) and "D_v_postscale_norm"
    # B = float(meta.get("global_batch", 512.0))
    # if B <= 0:
        # B = 512.0
    # required keys
    req = [
        "D_q_sum", "D_v_sum", "D_v_sum",
        "D_x_in_sum", "D_x_half_sum", "D_x_out_like_sum",
        "D_attn_branch_sum", "D_mlp_branch_sum",
        "D_attn_norm_out_sum", "D_mlp_norm_out_sum",
        "x_in_l2", "attn_out_l2",
        "x_half_l2", "mlp_out_l2",
        "x_out_l2",
    ]
    for k in req:
        if k not in C:
            raise KeyError(f"Missing key {k} in {pt_path}")
        
    vscale = False
    if "D_v_postscale_norm" in C.keys():
        vscale = True
        req.append("D_v_postscale_norm")

    # --- IMPORTANT ---
    x_in_l2    = C["x_in_l2"].float()       # [L,T]
    attn_out_l2 = C["attn_out_l2"].float()
    x_half_l2   = C["x_half_l2"].float() 
    mlp_out_l2  = C["mlp_out_l2"].float()
    # x_out_l2    = torch.cat([C["x_in_l2"][1:, :, :], C["x_out_l2"].unsqueeze(0)], dim=0).float()
    
    # Gradient Norms
    l2_D_attn_fb = C["D_attn_branch_sum"]              # [L,T]
    l2_D_mlp_fb  = C["D_mlp_branch_sum"]
    l2_D_attn_norm_out = C["D_attn_norm_out_sum"]
    l2_D_mlp_norm_out = C["D_mlp_norm_out_sum"]
    l2_D_x_in = C["D_x_in_sum"]
    l2_D_x_half = C["D_x_half_sum"]
    l2_D_x_out = C["D_x_out_like_sum"]

    # -------------------------
    # build metrics
    # tok 0, tok 1, tok 2-3 mean, tok 4-7 mean, rest mean
    # -------------------------
    # x magnitude
    x_in_groups   = group_tensors(x_in_l2)      # [G,L]
    x_half_groups = group_tensors(x_half_l2)

    # compression: feedback-to-residual over norm_out
    compress_attn = safe_div(l2_D_attn_fb, l2_D_attn_norm_out)  # [L,T]
    compress_attn_groups = group_tensors(compress_attn)
    compress_mlp = safe_div(l2_D_mlp_fb, l2_D_mlp_norm_out)
    compress_mlp_groups = group_tensors(compress_mlp)

    # pressure: attention feedback vs residual grad magnitude (use ||D_x_*|| as residual trunk)
    pressure_attn = safe_div(l2_D_attn_fb, l2_D_x_half)     # [L,T]
    pressure_attn_groups = group_tensors(pressure_attn)
    pressure_mlp = safe_div(l2_D_mlp_fb, l2_D_x_out)        # [L,T]
    pressure_mlp_groups = group_tensors(pressure_mlp)

    # bloat: ||D_*_norm_out|| / ||D_*_out||
    bloat_attn = safe_div(l2_D_attn_norm_out, l2_D_x_half)
    bloat_attn_groups = group_tensors(bloat_attn)
    bloat_mlp = safe_div(l2_D_mlp_norm_out, l2_D_x_out)
    bloat_mlp_groups = group_tensors(bloat_mlp)

    # change:
    change_attn = safe_div(l2_D_x_in, l2_D_x_half)
    change_attn_groups = group_tensors(change_attn)
    change_mlp = safe_div(l2_D_x_half, l2_D_x_out)
    change_mlp_groups = group_tensors(change_mlp)

    # kv
    l2_D_q = C["D_q_sum"]  # [L,T]
    l2_D_k = C["D_k_sum"]
    l2_D_v = C["D_v_sum"]
    D_k_group = group_tensors(l2_D_k)
    D_v_group = group_tensors(l2_D_v)

    # store
    steps.append(step)
    # x magnitude
    x_in_SL.append(x_in_groups.numpy())
    x_half_SL.append(x_half_groups.numpy())
    # compression
    compress_attn_SL.append(compress_attn_groups.numpy())
    compress_mlp_SL.append(compress_mlp_groups.numpy())
    # pressure
    pressure_attn_SL.append(pressure_attn_groups.numpy())
    pressure_mlp_SL.append(pressure_mlp_groups.numpy())
    # bloat
    bloat_attn_SL.append(bloat_attn_groups.numpy())
    bloat_mlp_SL.append(bloat_mlp_groups.numpy())
    # change
    change_attn_SL.append(change_attn_groups.numpy())
    change_mlp_SL.append(change_mlp_groups.numpy())
    # kv
    D_k_SL.append(D_k_group.numpy())
    D_v_SL.append(D_v_group.numpy())
    # grads layer means
    D_attn_norm_out_ST.append(l2_D_attn_norm_out.mean(dim=0).numpy())
    D_q_ST.append(l2_D_q.mean(dim=0).numpy())  # [T]
    D_k_ST.append(l2_D_k.mean(dim=0).numpy())
    D_v_ST.append(l2_D_v.mean(dim=0).numpy())

    # Draw violin
    if step == max_step:
        draw_violin(raw_data_LT=l2_D_attn_norm_out, n_tokens=2 * EARLY_TOKENS,
                    x_label="Token Index", y_label=r"$\log_{10}$ Gradient Norm",
                    title=f"Gradient Distribution at the Attention Input (after RMSNorm) of {model_name}",
                    outpath=os.path.join(collects_path, "violin." + EXT))


# stack to [G,S,L]
steps = list(steps)
if len(steps) == 0:
    raise RuntimeError("No valid collects.pt loaded.")

def stack(lst): return np.stack(lst, axis=0).transpose(1, 0, 2)

if DRAW_SCATTER_GROUP:
    out_dir = os.path.join(collects_path, "scatters")
    os.makedirs(out_dir, exist_ok=True)
    
    # stacks
    x_in_SL   = stack(x_in_SL)
    x_half_SL = stack(x_half_SL)
    compress_attn_SL = stack(compress_attn_SL)
    compress_mlp_SL  = stack(compress_mlp_SL)
    pressure_attn_SL = stack(pressure_attn_SL)
    pressure_mlp_SL  = stack(pressure_mlp_SL)
    bloat_attn_SL = stack(bloat_attn_SL)
    bloat_mlp_SL  = stack(bloat_mlp_SL)
    change_attn_SL = stack(change_attn_SL)
    change_mlp_SL  = stack(change_mlp_SL)
    D_k_SL = stack(D_k_SL)
    D_v_SL = stack(D_v_SL)

    G, S, L = x_in_SL.shape

    # === plot scatters (HSV: layer=hue, step=sat) ===
    def out(name): return os.path.join(out_dir, name)

    # E) token compare
    group_labels = ["token 0", "token 1", "token 2-3", "token 4-7", "rest"]

    scatter_compare_groups(
        compress_attn_SL, x_in_SL,
        steps, group_labels,
        title=r"Gradient Compression vs. Activation",
        xlabel="Gradient Compression of Attention",
        ylabel=r"$\|h_{\mathrm{in}}\|$",
        out_path=out("Ea1_compress." + EXT)
    )

    # scatter_compare_groups(
    #     pressure_attn_SL, x_in_SL,
    #     steps, group_labels,
    #     title="Gradient Pressure vs. Activation",
    #     xlabel="Gradient Pressure of Attention",
    #     ylabel=r"$\|h_{\mathrm{in}}\|$",
    #     out_path=out("Ea2_pressure." + EXT)
    # )

    scatter_compare_groups(
        bloat_attn_SL, x_in_SL,
        steps, group_labels,
        title="Gradient Bloat vs. Activation",
        xlabel="Gradient Bloat of Attention",
        ylabel=r"$\|h_{\mathrm{in}}\|$",
        out_path=out("Ea3_Bloat." + EXT)
    )

    scatter_compare_groups(
        change_attn_SL, x_in_SL,
        steps, group_labels,
        title="Gradient Change vs. Activation",
        xlabel="Gradient Change of Attention",
        ylabel=r"$\|h_{\mathrm{in}}\|$",
        out_path=out("Ea4_Change." + EXT),
        add_ref_vline_zero=True
    )

    # scatter_compare_groups(
    #     D_k_SL, x_in_SL,
    #     steps, group_labels,
    #     title="Evidence 5: Gradient k vs Activation Magnitude",
    #     xlabel="||D_k||",
    #     ylabel="||x_in||",
    #     out_path=out("E5_Dk." + EXT)
    # )

    # scatter_compare_groups(
    #     D_v_SL, x_in_SL,
    #     steps, group_labels,
    #     title="Evidence 6: Gradient v vs Activation Magnitude",
    #     xlabel="||D_v||",
    #     ylabel="||x_in||",
    #     out_path=out("E6_Dv." + EXT)
    # )

    scatter_compare_groups(
        compress_mlp_SL, x_half_SL,
        steps, group_labels,
        title="Gradient Compression vs. Activation",
        xlabel="Gradient Compression of MLP",
        ylabel=r"$\|h_{\mathrm{half}}\|$",
        out_path=out("Em1_compress." + EXT)
    )

    # scatter_compare_groups(
    #     pressure_mlp_SL, x_half_SL,
    #     steps, group_labels,
    #     title="Gradient Pressure vs. Activation",
    #     xlabel="Gradient Pressure of MLP",
    #     ylabel=r"$\|h_{\mathrm{half}}\|$",
    #     out_path=out("Em2_pressure." + EXT)
    # )

    scatter_compare_groups(
        bloat_mlp_SL, x_half_SL,
        steps, group_labels,
        title="Gradient Bloat vs. Activation",
        xlabel="Gradient Bloat of MLP",
        ylabel=r"$\|h_{\mathrm{half}}\|$",
        out_path=out("Em3_Bloat." + EXT)
    )

    scatter_compare_groups(
        change_mlp_SL, x_half_SL,
        steps, group_labels,
        title="Gradient Change vs. Activation",
        xlabel="Gradient Change of MLP",
        ylabel=r"$\|h_{\mathrm{half}}\|$",
        out_path=out("Em4_Change." + EXT),
        add_ref_vline_zero=True
    )

    save_step_colorbar(
        out("scatter_group_steps." + EXT),
        0, max(steps), base_color="#A82064",
        figsize=COLORBAR_SIZE_SCATTER_GROUP,
        cax_rect=COLORBAR_INNER_SCATTER_GROUP,
        tick_values=list(range(0, max(steps)+1, 5000)),
        label="Checkpoint Step"
    )

    print(f"\n[done] All scatter plots saved to: {out_dir}")


if DRAW_LINE:
    # plot the mean gradient for all tokens over layers
    line_dir = os.path.join(collects_path, "lines")
    os.makedirs(line_dir, exist_ok=True)

    # stacks
    # D_q_ST = np.stack(D_q_ST)  # [S,T]
    # D_k_ST = np.stack(D_k_ST)
    # D_v_ST = np.stack(D_v_ST)

    # print(type(D_q_ST), len(D_q_ST), len(D_q_ST[0]))

    S, T = np.stack(D_q_ST).shape

    # plot_ckpt_lines_colorbar(
    #     line_dir, list(range(T)), D_attn_norm_out_ST, steps,
    #     title="Gradient Norm at Attention Input",
    #     xlabel="Token Index", ylabel="Gradient Norm",
    #     fname="LINE_D_attn_norm_out_over_tokens." + EXT,
    #     cmap_name="Blues",
    #     draw_colorbar=False
    # )

    plot_ckpt_lines_colorbar(
        line_dir, list(range(T)), D_q_ST, steps,
        title="Query-Gradient Norm",
        xlabel="Token Index", ylabel="Gradient Norm",
        fname="LINE_Dq_over_tokens." + EXT,
        cmap_name="Blues",
        draw_colorbar=False
    )

    plot_ckpt_lines_colorbar(
        line_dir, list(range(T)), D_k_ST, steps,
        title="Key-Gradient Norm",
        xlabel="Token Index", ylabel="Gradient Norm",
        fname="LINE_Dk_over_tokens." + EXT,
        cmap_name="Blues",
        draw_colorbar=False
    )

    plot_ckpt_lines_colorbar(
        line_dir, list(range(T)), D_v_ST, steps,
        title="Value-Gradient Norm",
        xlabel="Token Index", ylabel="Gradient Norm",
        fname="LINE_Dv_over_tokens." + EXT,
        cmap_name="Blues",
        draw_colorbar=False
    )

    
    save_step_colorbar(
        os.path.join(line_dir, "line_steps." + EXT),
        0, max(steps), cmap_name="Blues",
        figsize=COLORBAR_SIZE_LINE,
        cax_rect=COLORBAR_INNER_LINE,
        tick_values=list(range(0, max(steps)+1, 5000)),
        label="Checkpoint Step"
    )

    print(f"\n[done] All line plots saved to: {line_dir}")
