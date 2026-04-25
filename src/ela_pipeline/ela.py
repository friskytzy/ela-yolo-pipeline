"""Error Level Analysis (ELA) image generation.

Pipeline:

1. Load source JPEG with PIL.
2. Re-encode it once at quality ``q_recompress`` (default 90) to a temp buffer.
3. Decode the buffer back to RGB.
4. Compute pixel-wise absolute difference, scale by ``ela_scale`` and clip.
5. Write the result as JPEG (q=95) so downstream YOLO training receives a
   stable representation.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import ensure_dirs, list_images, save_jpeg

logger = logging.getLogger(__name__)


def compute_ela(img: Image.Image, *, q_recompress: int = 90, scale: float = 12.0) -> Image.Image:
    """Compute the ELA representation for a single PIL image."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q_recompress, subsampling=0, optimize=False)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    arr_a = np.asarray(img, dtype=np.int16)
    arr_b = np.asarray(recompressed, dtype=np.int16)
    diff = np.abs(arr_a - arr_b).astype(np.float32) * float(scale)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def compute_ela_array(
    img_rgb: np.ndarray,
    *,
    q_recompress: int = 90,
    scale: float = 12.0,
    return_raw: bool = False,
) -> np.ndarray:
    """Numpy variant. Returns a uint8 RGB array (or float32 raw if requested)."""
    pil = Image.fromarray(img_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q_recompress, subsampling=0, optimize=False)
    buf.seek(0)
    re = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    diff = np.abs(img_rgb.astype(np.int16) - re).astype(np.float32)
    if return_raw:
        return diff
    out = np.clip(diff * float(scale), 0, 255).astype(np.uint8)
    return out


def ela_directory(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    q_recompress: int = 90,
    scale: float = 12.0,
    out_quality: int = 95,
) -> int:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    ensure_dirs(out_dir)
    images = list_images(src_dir)
    n = 0
    for p in tqdm(images, desc=f"ELA {src_dir.name}"):
        try:
            with Image.open(p) as im:
                ela = compute_ela(im, q_recompress=q_recompress, scale=scale)
            save_jpeg(ela, out_dir / p.name, quality=out_quality)
            n += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("ELA failed for %s: %s", p, e)
    logger.info("Wrote %d ELA images to %s", n, out_dir)
    return n
