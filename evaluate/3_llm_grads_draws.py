import os
import torch

from draw_helper import *

DIV_EPS = 1e-30

EARLY_TOKENS = 16 # or None for all other tokens

DRAW_SCATTER_GROUP = True
DRAW_LINE = True

EXT = "pdf"



# =========================
# config
# =========================
device = "cpu"
model_name = "GLM-4-9B"
collects_path = f"evaluate/outputs/llm_grads_check/{model_name}"


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
def token_l2(x: torch.Tensor) -> torch.Tensor:
    """
    Per-(L,T) L2 norm.
    - [L,T,D] -> [L,T]
    - [L,H,T,Dh] -> [L,T]
    - [L,T] -> [L,T]
    IMPORTANT: no EPS inside sqrt.
    """
    if x.dim() == 2:
        return x.float()
    if x.dim() == 3:
        return torch.sqrt((x.float() ** 2).sum(dim=-1))
    if x.dim() == 4:
        return torch.sqrt((x.float() ** 2).sum(dim=(1, 3)))
    raise ValueError(f"Unsupported dim={x.dim()} shape={tuple(x.shape)}")


def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = DIV_EPS) -> torch.Tensor:
    return a / (b + eps)


def group_tensors(x:torch.Tensor):
    """ x: [L,T] size """
    x0 = x[:, 0]
    x1 = x[:, 1]
    x23 = x[:, 2:4].mean(dim=-1)
    x47 = x[:, 4:8].mean(dim=-1)
    if EARLY_TOKENS is not None:
        xr = x[:, 8:EARLY_TOKENS].mean(dim=-1)
    else:
        xr = x[:, 8:].mean(dim=-1)
    # all [L]
    return torch.stack([x0, x1, x23, x47, xr], dim=0)

# =========================
# load stats
# =========================
stats_path = os.path.join(collects_path, "stats.pt")
obj = torch.load(stats_path, map_location=device)
# print(obj.keys())
# dict_keys(['act_mlp_out_mean', 'act_mlp_norm_out_mean', 'act_attn_norm_out_mean', 'act_x_half_mean', 'act_x_in_mean', 'act_attn_out_mean', 'act_x_out_mean', 'grad_mlp_out_mean', 'grad_mlp_norm_out_mean', 'grad_attn_norm_out_mean', 'grad_x_half_mean', 'grad_x_in_mean', 'grad_attn_out_mean', 'grad_x_out_mean', 'grad_q_mean', 'grad_v_mean', 'grad_k_mean', 'loss_mean', 'loss_all'])

D_attn_norm_out = obj["grad_attn_norm_out_mean"]    # [L,T]
D_mlp_norm_out  = obj["grad_mlp_norm_out_mean"]
D_x_in   = obj["grad_x_in_mean"]
D_x_half = obj["grad_x_half_mean"]
D_x_out  = obj["grad_x_out_mean"]

D_q = obj["grad_q_mean"]
D_k = obj["grad_k_mean"]
D_v = obj["grad_v_mean"]

D_attn_fb = obj["grad_attn_fb_mean"]
D_mlp_fb  = obj["grad_mlp_fb_mean"]

x_in_l2 = obj["act_x_in_mean"]      # [L,T]
x_half_l2 = obj["act_x_half_mean"]
x_out_l2 = obj["act_x_out_mean"]
L, T = x_in_l2.shape

# activation
x_in_groups   = group_tensors(x_in_l2)
x_half_groups = group_tensors(x_half_l2)

# compression: feedback-to-residual over norm_out
compress_attn = safe_div(D_attn_fb, D_attn_norm_out)  # [L,T]
compress_attn_groups = group_tensors(compress_attn)
compress_mlp = safe_div(D_mlp_fb, D_mlp_norm_out)
compress_mlp_groups = group_tensors(compress_mlp)

# pressure: attention feedback vs residual grad magnitude (use ||D_x_*|| as residual trunk)
pressure_attn = safe_div(D_attn_fb, D_x_half)     # [L,T]
pressure_attn_groups = group_tensors(pressure_attn)
pressure_mlp = safe_div(D_mlp_fb, D_x_out)        # [L,T]
pressure_mlp_groups = group_tensors(pressure_mlp)

# bloat: ||D_*_norm_out|| / ||D_*_out||
bloat_attn = safe_div(D_attn_norm_out, D_x_half)
bloat_attn_groups = group_tensors(bloat_attn)
bloat_mlp = safe_div(D_mlp_norm_out, D_x_out)
bloat_mlp_groups = group_tensors(bloat_mlp)

# change:
change_attn = safe_div(D_x_in, D_x_half)
change_attn_groups = group_tensors(change_attn)
change_mlp = safe_div(D_x_half, D_x_out)
change_mlp_groups = group_tensors(change_mlp)

# kv
D_k_group = group_tensors(D_k)
D_v_group = group_tensors(D_v)


ma_xout_groups = {}
ma_xout_groups["tok0"] = [x_out_l2[:, 0].tolist()]
ma_xout_groups["mean_rest"] = [x_out_l2[:, 1:EARLY_TOKENS].mean(dim=1).tolist()]
ma_xout_groups["max_rest"] = [x_out_l2[:, 1:EARLY_TOKENS].amax(dim=1).tolist()]


steps = [100]

# Draw MA lines
plot_ckpt_group_lines(
    out_path=os.path.join(collects_path, f"MA_x_out_compare.{EXT}"),
    x=range(L),
    group_lines=ma_xout_groups,
    steps_sorted=steps,
    title="Residual-Stream Output Norm",
    xlabel="Layer Index",
    ylabel="Activation Norm",
    cmap_name="Blues",
    group_style={
        "tok0":      dict(dh=0.00, ds=0.0, dv=0.0, ls="-",  lw=1.70, alpha=1.00),
        "mean_rest": dict(dh=-0.18, ds=0.0, dv=0.0, ls="--", lw=1.20, alpha=0.86),
        "max_rest":  dict(dh=+0.18, ds=0.0, dv=0.0, ls=":",  lw=1.20, alpha=0.86),
    },
    group_label_map={
        "tok0": "token 0",
        "mean_rest": "mean early rest",
        "max_rest": "max early rest",
    },
    figsize=FIGSIZE_COMPARE,
)


# Draw Violin
draw_violin(raw_data_LT=D_attn_norm_out, n_tokens=2 * EARLY_TOKENS, 
            x_label="Token Index", y_label=r"$\log_{10}$ Gradient Norm",
            title=f"Gradient Distribution at the Attention Input (after RMSNorm) of {model_name}",
            outpath=os.path.join(collects_path, "violin." + EXT))


if DRAW_SCATTER_GROUP:
    out_dir = os.path.join(collects_path, "scatters")
    os.makedirs(out_dir, exist_ok=True)

    def out(name): return os.path.join(out_dir, name)

    # E) token compare
    group_labels = ["token 0", "token 1", "token 2-3", "token 4-7", "rest"]

    scatter_compare_groups(
        compress_attn_groups, x_in_groups,
        steps, group_labels,
        title=r"Gradient Compression vs. Activation",
        xlabel="Gradient Compression of Attention",
        ylabel=r"$\|h_{\mathrm{in}}\|$",
        out_path=out("Ea1_compress." + EXT)
    )

    scatter_compare_groups(
        bloat_attn_groups, x_in_groups,
        steps, group_labels,
        title="Gradient Bloat vs. Activation",
        xlabel="Gradient Bloat of Attention",
        ylabel=r"$\|h_{\mathrm{in}}\|$",
        out_path=out("Ea3_Bloat." + EXT)
    )

    scatter_compare_groups(
        change_attn_groups, x_in_groups,
        steps, group_labels,
        title="Gradient Change vs. Activation",
        xlabel="Gradient Change of Attention",
        ylabel=r"$\|h_{\mathrm{in}}\|$",
        out_path=out("Ea4_Change." + EXT),
        add_ref_vline_zero=True
    )

    scatter_compare_groups(
        compress_mlp_groups, x_half_groups,
        steps, group_labels,
        title="Gradient Compression vs. Activation",
        xlabel="Gradient Compression of MLP",
        ylabel=r"$\|h_{\mathrm{half}}\|$",
        out_path=out("Em1_compress." + EXT)
    )

    scatter_compare_groups(
        bloat_mlp_groups, x_half_groups,
        steps, group_labels,
        title="Gradient Bloat vs. Activation",
        xlabel="Gradient Bloat of MLP",
        ylabel=r"$\|h_{\mathrm{half}}\|$",
        out_path=out("Em3_Bloat." + EXT)
    )

    scatter_compare_groups(
        change_mlp_groups, x_half_groups,
        steps, group_labels,
        title="Gradient Change vs. Activation",
        xlabel="Gradient Change of MLP",
        ylabel=r"$\|h_{\mathrm{half}}\|$",
        out_path=out("Em4_Change." + EXT),
        add_ref_vline_zero=True
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
    _, T = D_v.shape
    D_q_ST = D_q.mean(dim=0, keepdim=True).tolist()
    D_k_ST = D_k.mean(dim=0, keepdim=True).tolist()
    D_v_ST = D_v.mean(dim=0, keepdim=True).tolist()

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

    print(f"\n[done] All line plots saved to: {line_dir}")
