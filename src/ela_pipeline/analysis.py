"""Quantitative + visual analysis of the ELA pipeline.

Produces (under ``reports/``):

* PSNR / SSIM tables comparing original vs JPEG-90 recompressed
* Histogram of mean ELA error per image (authentic vs tampered)
* Inside-vs-outside-bbox mean ELA error for tampered images (correlation)
* Heatmap visualisations for a few example images
* ``metrics.json`` summarising aggregate statistics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

from .ela import compute_ela_array
from .utils import ensure_dirs, list_images, load_image_np

logger = logging.getLogger(__name__)


def _mean_error(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def analyse_set(
    *,
    authentic_dir: str | Path,
    tampered_dirs: list[str | Path],
    tamper_manifest_path: str | Path,
    reports_dir: str | Path,
    q_recompress: int = 90,
    ela_scale: float = 12.0,
    sample_visuals: int = 6,
) -> dict:
    authentic_dir = Path(authentic_dir)
    tampered_dirs = [Path(d) for d in tampered_dirs]
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    ensure_dirs(reports_dir, figures_dir)

    with open(tamper_manifest_path) as f:
        manifest = json.load(f)
    manifest_by_stem = {Path(k).stem: v for k, v in manifest.items()}

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    mean_err_authentic: list[float] = []
    mean_err_tampered: list[float] = []
    inside_vs_outside: list[tuple[float, float]] = []  # per tampered image
    per_kind_inside_outside: dict[str, list[tuple[float, float]]] = {}
    per_kind_mean_err: dict[str, list[float]] = {}

    # ---------- Authentic stats ----------
    for p in tqdm(list_images(authentic_dir), desc="analyse authentic"):
        rgb = load_image_np(p)
        raw_diff = compute_ela_array(rgb, q_recompress=q_recompress, return_raw=True)
        # Reconstruct recompressed for PSNR/SSIM
        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=q_recompress, subsampling=0, optimize=False)
        buf.seek(0)
        re_rgb = np.array(Image.open(buf).convert("RGB"))
        psnr_values.append(float(psnr_fn(rgb, re_rgb, data_range=255)))
        ssim_values.append(
            float(ssim_fn(rgb, re_rgb, channel_axis=2, data_range=255))
        )
        mean_err_authentic.append(_mean_error(raw_diff))

    # ---------- Tampered stats ----------
    # Balanced sampler: aim for an equal number of visuals per tampering kind.
    per_kind_visuals: dict[str, int] = {}
    visuals_per_kind = max(1, sample_visuals // 2)
    visual_examples: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int, int]]] = []
    for d in tampered_dirs:
        for p in tqdm(list_images(d), desc=f"analyse {d.name}"):
            meta = manifest_by_stem.get(p.stem)
            if not meta:
                continue
            rgb = load_image_np(p)
            raw = compute_ela_array(rgb, q_recompress=q_recompress, return_raw=True)
            err_per_pixel = raw.mean(axis=2)  # collapse RGB
            mean_err_tampered.append(float(err_per_pixel.mean()))
            kind = meta.get("kind", "unknown")
            per_kind_mean_err.setdefault(kind, []).append(float(err_per_pixel.mean()))

            x1, y1, x2, y2 = meta["tampered_xyxy"]
            H, W = err_per_pixel.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            inside_mask = np.zeros_like(err_per_pixel, dtype=bool)
            inside_mask[y1:y2, x1:x2] = True
            inside_mean = float(err_per_pixel[inside_mask].mean())
            outside_mean = float(err_per_pixel[~inside_mask].mean()) if (~inside_mask).any() else 0.0
            inside_vs_outside.append((inside_mean, outside_mean))
            per_kind_inside_outside.setdefault(kind, []).append((inside_mean, outside_mean))

            if per_kind_visuals.get(kind, 0) < visuals_per_kind:
                # Display-friendly scaled ELA
                display = np.clip(raw * ela_scale, 0, 255).astype(np.uint8)
                visual_examples.append((f"[{kind}] {p.name}", rgb, display, (x1, y1, x2, y2)))
                per_kind_visuals[kind] = per_kind_visuals.get(kind, 0) + 1

    # ---------- Plots ----------
    # 1. Mean ELA error histogram
    plt.figure(figsize=(8, 5))
    if mean_err_authentic:
        plt.hist(mean_err_authentic, bins=30, alpha=0.55, label="authentic", color="#1f77b4")
    if mean_err_tampered:
        plt.hist(mean_err_tampered, bins=30, alpha=0.55, label="tampered", color="#d62728")
    plt.xlabel("Mean per-image ELA error (raw, pre-scale)")
    plt.ylabel("Count")
    plt.title("Distribution of ELA error: authentic vs tampered")
    plt.legend()
    plt.tight_layout()
    hist_path = figures_dir / "ela_error_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # 2. Inside-vs-outside scatter (coloured per tampering kind)
    if per_kind_inside_outside:
        plt.figure(figsize=(7, 6))
        all_vals = []
        for k, vals in per_kind_inside_outside.items():
            all_vals.extend(vals)
        ins_all = np.array([v[0] for v in all_vals])
        outs_all = np.array([v[1] for v in all_vals])
        m = float(max(ins_all.max(), outs_all.max(), 1e-3)) * 1.05
        plt.plot([0, m], [0, m], "k--", alpha=0.3, label="y = x (no signal)")
        palette = {"copy_move": "#d62728", "removal": "#1f77b4"}
        for k, vals in per_kind_inside_outside.items():
            ins = np.array([v[0] for v in vals])
            outs = np.array([v[1] for v in vals])
            plt.scatter(outs, ins, alpha=0.65, color=palette.get(k, "#9467bd"),
                        label=f"{k} (n={len(vals)})", edgecolor="white", linewidth=0.5)
        plt.xlabel("Mean ELA error OUTSIDE bbox")
        plt.ylabel("Mean ELA error INSIDE bbox")
        plt.title("Tampered images: bbox vs background ELA error")
        plt.legend()
        plt.tight_layout()
        sc_path = figures_dir / "inside_vs_outside.png"
        plt.savefig(sc_path, dpi=150)
        plt.close()

    # 2b. Per-kind histograms of mean error
    if per_kind_mean_err:
        plt.figure(figsize=(8, 5))
        if mean_err_authentic:
            plt.hist(mean_err_authentic, bins=30, alpha=0.55,
                     label=f"authentic (n={len(mean_err_authentic)})", color="#2ca02c")
        for k, vals in per_kind_mean_err.items():
            plt.hist(vals, bins=30, alpha=0.55,
                     label=f"{k} (n={len(vals)})",
                     color={"copy_move": "#d62728", "removal": "#1f77b4"}.get(k, "#9467bd"))
        plt.xlabel("Mean per-image ELA error")
        plt.ylabel("Count")
        plt.title("Per-class ELA error distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "ela_error_per_kind.png", dpi=150)
        plt.close()

    # 3. Heatmap visual examples
    if visual_examples:
        n = len(visual_examples)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
        if n == 1:
            axes = np.array([axes])
        for i, (name, rgb, ela_img, (x1, y1, x2, y2)) in enumerate(visual_examples):
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(f"{name}\noriginal", fontsize=8)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(ela_img)
            axes[i, 1].set_title("ELA (×scale)", fontsize=8)
            axes[i, 1].axis("off")
            heat = ela_img.mean(axis=2)
            im = axes[i, 2].imshow(heat, cmap="hot")
            axes[i, 2].add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", lw=1.5
                )
            )
            axes[i, 2].set_title("ELA heatmap + GT bbox", fontsize=8)
            axes[i, 2].axis("off")
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        hp = figures_dir / "ela_heatmap_examples.png"
        plt.savefig(hp, dpi=120)
        plt.close()

    # ---------- Metrics ----------
    inside_arr = np.array([x[0] for x in inside_vs_outside]) if inside_vs_outside else np.array([])
    outside_arr = np.array([x[1] for x in inside_vs_outside]) if inside_vs_outside else np.array([])
    per_kind_summary: dict = {}
    for k, vals in per_kind_inside_outside.items():
        ia = np.array([v[0] for v in vals])
        oa = np.array([v[1] for v in vals])
        per_kind_summary[k] = {
            "n": int(len(vals)),
            "inside_mean": float(ia.mean()) if ia.size else None,
            "outside_mean": float(oa.mean()) if oa.size else None,
            "inside_to_outside_ratio": (
                float(ia.mean() / oa.mean()) if oa.size and oa.mean() > 0 else None
            ),
            "fraction_with_inside_gt_outside": (
                float((ia > oa).mean()) if ia.size else None
            ),
        }
    metrics = {
        "n_authentic": len(mean_err_authentic),
        "n_tampered": len(mean_err_tampered),
        "psnr_orig_vs_recompressed": {
            "mean": float(np.mean(psnr_values)) if psnr_values else None,
            "std": float(np.std(psnr_values)) if psnr_values else None,
        },
        "ssim_orig_vs_recompressed": {
            "mean": float(np.mean(ssim_values)) if ssim_values else None,
            "std": float(np.std(ssim_values)) if ssim_values else None,
        },
        "mean_ela_error": {
            "authentic_mean": float(np.mean(mean_err_authentic)) if mean_err_authentic else None,
            "tampered_mean": float(np.mean(mean_err_tampered)) if mean_err_tampered else None,
        },
        "inside_vs_outside_bbox": {
            "n": int(len(inside_arr)),
            "inside_mean": float(inside_arr.mean()) if inside_arr.size else None,
            "outside_mean": float(outside_arr.mean()) if outside_arr.size else None,
            "inside_to_outside_ratio": (
                float(inside_arr.mean() / outside_arr.mean())
                if inside_arr.size and outside_arr.mean() > 0
                else None
            ),
            "fraction_with_inside_gt_outside": (
                float((inside_arr > outside_arr).mean()) if inside_arr.size else None
            ),
        },
        "per_kind": per_kind_summary,
    }
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote analysis metrics to %s", metrics_path)
    return metrics
