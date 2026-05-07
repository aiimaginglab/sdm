import os 
import json
import numpy as np
import pandas as pd

import torch 
from torchvision.utils import make_grid

import PIL.Image
import matplotlib.pyplot as plt


def show_samples(x, nrow=4):
    # x: [B, C, H, W], range: [-1, 1]
    grid = make_grid((x.clamp(-1,1)*0.5+0.5), nrow=nrow)  # range [0,1]
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.show()
    
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def append_jsonl(path, records):
    # list[dict]
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def aggregate_gaplog(jsonl_filepath):
    rows = load_jsonl(jsonl_filepath)
    df = pd.DataFrame(rows)

    num_cols = [
        "r_abs","r_rel","k_abs","k_rel",
        "Dtheta_norm","Dsigma_norm","JD_eps_norm","bracket_norm",
        "ddotx_closed_norm","ddotx_fd_norm","ddotx_recon_err_norm",
        "sigma_cur","sigma_next","sigma_hat","h","batch_size"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "batch_size" not in df.columns:
        df["batch_size"] = 1.0

    # ---- take average per step 
    def wavg(g, col):
        return (g[col]*g["batch_size"]).sum()/g["batch_size"].sum()

    agg = df.groupby("step").apply(
        lambda g: pd.Series({
            "sigma": np.nanmean(g["sigma_hat"].values) if "sigma_hat" in g.columns and g["sigma_hat"].notna().any()
                      else np.nanmean(g["sigma_cur"].values),
            "r_abs": wavg(g,"r_abs") if "r_abs" in g else np.nan,
            "r_rel": wavg(g,"r_rel") if "r_rel" in g else np.nan,
            "k_abs": wavg(g,"k_abs") if "k_abs" in g else np.nan,
            "k_rel": wavg(g,"k_rel") if "k_rel" in g else np.nan,
            "Dtheta_norm": wavg(g,"Dtheta_norm") if "Dtheta_norm" in g else np.nan,
            "Dsigma_norm": wavg(g,"Dsigma_norm") if "Dsigma_norm" in g else np.nan,
            "JD_eps_norm": wavg(g,"JD_eps_norm") if "JD_eps_norm" in g else np.nan,
            "bracket_norm": wavg(g,"bracket_norm") if "bracket_norm" in g else np.nan,
            "ddotx_closed_norm": wavg(g,"ddotx_closed_norm") if "ddotx_closed_norm" in g else np.nan,
            "ddotx_fd_norm": wavg(g,"ddotx_fd_norm") if "ddotx_fd_norm" in g else np.nan,
            "ddotx_recon_err_norm": wavg(g,"ddotx_recon_err_norm") if "ddotx_recon_err_norm" in g else np.nan,
            "count": int(g["batch_size"].sum())
        })
    ).reset_index()

    return agg

def visualize_curvature_metrics_df(
    df, save_path=None, plot_by: str = "sigma"
):
    metrics = [
        "Dtheta_norm","Dsigma_norm","JD_eps_norm","bracket_norm",
        "ddotx_closed_norm","ddotx_fd_norm","ddotx_recon_err_norm",
        "r_abs","r_rel","k_abs","k_rel"
    ]

    if plot_by not in ["step","sigma","log_sigma"]:
        raise ValueError("plot_by must be 'step', 'sigma', or 'log_sigma'")
    if plot_by in ["sigma","log_sigma"] and "sigma" not in df.columns:
        raise ValueError("sigma not found in df; cannot plot by sigma")

    if plot_by == "sigma":
        x_axis = df["sigma"]
        x_label = "σ (noise level)"
    elif plot_by == "log_sigma":
        x_axis = np.log10(df["sigma"].clip(lower=1e-12))
        x_label = "log₁₀(σ)"
    else:
        x_axis = df["step"]
        x_label = "Step index"

    n_rows = int(np.ceil(len(metrics)/2))
    plt.figure(figsize=(10, 3.2*n_rows))
    for i, m in enumerate(metrics, 1):
        if m not in df.columns: continue
        plt.subplot(n_rows, 2, i)
        plt.plot(x_axis, df[m], linewidth=2, label=m)
        plt.xlabel(x_label); plt.ylabel(m); plt.title(f"{m} vs {x_label}")
        plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout()
    if save_path:
        tag = plot_by.replace("_","")
        plt.savefig(f"{save_path}/curvature_metrics_vs_{tag}.png", dpi=200)
    plt.show()

def fit_powerlaw(x, y, mask_extra=None, min_pts=3):
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x>0) & (y>0)
    if mask_extra is not None: mask &= mask_extra
    if mask.sum() < min_pts: return None
    lx = np.log(x[mask]); ly = np.log(y[mask])
    p, b = np.polyfit(lx, ly, 1)  # y ≈ e^b * x^p
    return {'p': p, 'C': np.exp(b), 'mask': mask}

def save_images(x: torch.Tensor, output_dir: str, seeds: torch.Tensor | np.ndarray,
                          subdirs: bool = True, is_latent: bool = False):
    """
    generates image output format as the following.
        x: [B, C, H, W], range: [-1, 1]
    seeds: [B] (batch_idx * B + i)
    outdir: output directory
    subdirs: if True, save folder by 1000 images 
    """
    os.makedirs(output_dir, exist_ok=True)
    B, C, H, W = x.shape
    # (images * 127.5 + 128).clip(0,255).uint8  with images in [-1,1]

    if is_latent:
        images_np = x.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
    else:
        images_u8 = (x.clamp(-1, 1) * 127.5 + 128).round().clamp(0, 255).to(torch.uint8)
        images_np = images_u8.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]

    seeds = np.asarray(seeds, dtype=np.int64)
    for seed, image_np in zip(seeds, images_np):
        image_dir = os.path.join(output_dir, f'{seed - seed % 1000:06d}') if subdirs else output_dir
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f'{seed:06d}.png')
        if image_np.shape[2] == 1:
            PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
        else:
            PIL.Image.fromarray(image_np, 'RGB').save(image_path)

def save_coco_images(
    x: torch.Tensor,
    output_dir: str,
    seeds: torch.Tensor | np.ndarray | None = None,
    file_names: list[str] | None = None,
    subdirs: bool = True,
):
    """
    Save images.

    Args:
        x: [B, C, H, W], range [-1, 1]
        output_dir: output directory
        seeds: [B], optional. Used for seed-based naming and/or subdir bucketing.
        file_names: list[str], optional. Used for file-name-based naming.
                    Example: COCO_val2014_000000123456.jpg
                    Saved as: COCO_val2014_000000123456.png
        subdirs: if True, save folder by 1000 images

    Supported modes:
        1) seeds only
        2) file_names + seeds   <- recommended for COCO eval
    """
    os.makedirs(output_dir, exist_ok=True)

    if seeds is None and file_names is None:
        raise ValueError("Provide at least one of `seeds` or `file_names`.")

    B, C, H, W = x.shape
    images_u8 = (x.clamp(-1, 1) * 127.5 + 128).round().clamp(0, 255).to(torch.uint8)
    images_np = images_u8.permute(0, 2, 3, 1).cpu().numpy()

    if seeds is not None:
        seeds = np.asarray(seeds, dtype=np.int64)
        if len(seeds) != B:
            raise ValueError(f"len(seeds)={len(seeds)} does not match batch size={B}")

    if file_names is not None and len(file_names) != B:
        raise ValueError(f"len(file_names)={len(file_names)} does not match batch size={B}")

    for i, image_np in enumerate(images_np):
        if file_names is not None:
            stem = os.path.splitext(os.path.basename(file_names[i]))[0]
            out_name = f"{stem}.png"
        else:
            out_name = f"{int(seeds[i]):06d}.png"

        if subdirs:
            if seeds is not None:
                bucket = int(seeds[i]) - int(seeds[i]) % 1000
            else:
                bucket = i - i % 1000
            image_dir = os.path.join(output_dir, f"{bucket:06d}")
        else:
            image_dir = output_dir

        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, out_name)

        if image_np.shape[2] == 1:
            PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
        else:
            PIL.Image.fromarray(image_np, "RGB").save(image_path)