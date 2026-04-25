"""Generate tampered images from authentic counterparts (1:1 mapping).

Two manipulations are produced for every authentic image:

* ``copy_move``: pick a YOLO bbox region, paste it elsewhere with mild
  Gaussian-blurred edges (controlled by config) so the seam looks natural.
* ``removal``: erase a YOLO bbox region and inpaint with OpenCV
  Telea / Navier-Stokes so the result has no obvious gap.

Tampered images are saved at the **same JPEG quality** as authentic to keep
ELA controls intact, plus a JSON sidecar storing the tampered region polygon
in pixel coordinates so the YOLO label generator has ground truth.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import (
    BBox,
    ensure_dirs,
    list_images,
    parse_yolo_label,
    save_jpeg,
)

logger = logging.getLogger(__name__)

INPAINT_METHODS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


def _filter_bboxes(boxes: list[BBox], min_frac: float, max_frac: float) -> list[BBox]:
    return [b for b in boxes if min_frac <= b.area_frac() <= max_frac]


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _feathered_mask(h: int, w: int, kernel: int, sigma: float) -> np.ndarray:
    """Return a soft alpha mask of shape (h, w) in [0, 1]."""
    mask = np.ones((h, w), dtype=np.float32)
    if kernel > 1:
        mask = cv2.GaussianBlur(mask, (kernel | 1, kernel | 1), sigma)
    return mask


def copy_move(
    img: np.ndarray,
    bbox: BBox,
    *,
    blur_kernel: int,
    blur_sigma: float,
    rng: random.Random,
    min_shift_frac: float,
    max_shift_frac: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (tampered_image_BGR, target_xyxy) for the moved region."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox.to_xyxy(W, H)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 4:
        raise ValueError("bbox too small")

    patch = img[y1:y2, x1:x2].copy()
    mask = _feathered_mask(bh, bw, blur_kernel, blur_sigma)[..., None]

    # Pick a translation that keeps the patch in-bounds and far enough from
    # its source location to avoid trivial overlap.
    for _ in range(60):
        dx = rng.choice([-1, 1]) * rng.uniform(min_shift_frac, max_shift_frac) * W
        dy = rng.choice([-1, 1]) * rng.uniform(min_shift_frac, max_shift_frac) * H
        nx1 = int(x1 + dx)
        ny1 = int(y1 + dy)
        nx2 = nx1 + bw
        ny2 = ny1 + bh
        if 0 <= nx1 and nx2 <= W and 0 <= ny1 and ny2 <= H:
            # require IoU=0 with source
            ix1, iy1 = max(x1, nx1), max(y1, ny1)
            ix2, iy2 = min(x2, nx2), min(y2, ny2)
            if ix1 >= ix2 or iy1 >= iy2:
                break
    else:
        raise RuntimeError("could not find non-overlapping target location")

    out = img.copy()
    region = out[ny1:ny2, nx1:nx2].astype(np.float32)
    blended = patch.astype(np.float32) * mask + region * (1.0 - mask)
    out[ny1:ny2, nx1:nx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out, (nx1, ny1, nx2, ny2)


def removal(
    img: np.ndarray,
    bbox: BBox,
    *,
    method: str,
    inpaint_radius: int,
    mask_dilate: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Object removal via OpenCV inpainting."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox.to_xyxy(W, H)
    if x2 - x1 <= 4 or y2 - y1 <= 4:
        raise ValueError("bbox too small")

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    if mask_dilate > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_dilate, mask_dilate))
        mask = cv2.dilate(mask, kernel, iterations=1)
    flag = INPAINT_METHODS[method]
    inpainted = cv2.inpaint(img, mask, inpaint_radius, flag)
    return inpainted, (x1, y1, x2, y2)


def tamper_dataset(
    *,
    auth_dir: str | Path,
    label_dir: str | Path,
    cm_out_dir: str | Path,
    rm_out_dir: str | Path,
    cfg: dict,
) -> dict:
    """Run both tampering modes over the curated authentic set.

    Returns a manifest dict keyed by output filename containing the pixel
    bbox of the manipulated region (used downstream for YOLO labels).
    """
    auth_dir, label_dir = Path(auth_dir), Path(label_dir)
    cm_out_dir, rm_out_dir = Path(cm_out_dir), Path(rm_out_dir)
    ensure_dirs(cm_out_dir, rm_out_dir)

    rng = random.Random(cfg["dataset"]["random_seed"])

    cm_cfg = cfg["tamper"]["copy_move"]
    rm_cfg = cfg["tamper"]["removal"]
    quality = cfg["dataset"]["jpeg_quality"]
    cm_per = cfg["dataset"]["copy_move_per_image"]
    rm_per = cfg["dataset"]["removal_per_image"]

    manifest: dict[str, dict] = {}

    images = list_images(auth_dir)
    logger.info("Tampering %d authentic images", len(images))

    for img_path in tqdm(images, desc="tamper"):
        label_path = label_dir / f"{img_path.stem}.txt"
        boxes_all = parse_yolo_label(label_path)
        cm_boxes = _filter_bboxes(
            boxes_all, cm_cfg["min_bbox_area_frac"], cm_cfg["max_bbox_area_frac"],
        )
        rm_boxes = _filter_bboxes(
            boxes_all, rm_cfg["min_bbox_area_frac"], rm_cfg["max_bbox_area_frac"],
        )

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("cv2 failed to read %s", img_path)
            continue

        # ---- Copy-move ----
        for k in range(cm_per):
            if not cm_boxes:
                logger.debug("No suitable bbox for copy-move on %s", img_path.name)
                break
            box = rng.choice(cm_boxes)
            try:
                tampered, dest_xyxy = copy_move(
                    bgr,
                    box,
                    blur_kernel=cm_cfg["blur_kernel"],
                    blur_sigma=cm_cfg["blur_sigma"],
                    rng=rng,
                    min_shift_frac=cm_cfg["min_shift_frac"],
                    max_shift_frac=cm_cfg["max_shift_frac"],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("copy_move failed on %s: %s", img_path.name, e)
                continue

            out_name = f"{img_path.stem}_cm{k}.jpg"
            out_path = cm_out_dir / out_name
            save_jpeg(_bgr_to_pil(tampered), out_path, quality=quality)
            manifest[str(out_path)] = {
                "src": str(img_path),
                "kind": "copy_move",
                "tampered_xyxy": list(dest_xyxy),
                "image_size": [bgr.shape[1], bgr.shape[0]],
            }

        # ---- Removal ----
        for k in range(rm_per):
            if not rm_boxes:
                break
            box = rng.choice(rm_boxes)
            try:
                tampered, region_xyxy = removal(
                    bgr,
                    box,
                    method=rm_cfg["method"],
                    inpaint_radius=rm_cfg["inpaint_radius"],
                    mask_dilate=rm_cfg["mask_dilate"],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("removal failed on %s: %s", img_path.name, e)
                continue
            out_name = f"{img_path.stem}_rm{k}.jpg"
            out_path = rm_out_dir / out_name
            save_jpeg(_bgr_to_pil(tampered), out_path, quality=quality)
            manifest[str(out_path)] = {
                "src": str(img_path),
                "kind": "removal",
                "tampered_xyxy": list(region_xyxy),
                "image_size": [bgr.shape[1], bgr.shape[0]],
            }

    manifest_path = Path(cm_out_dir).parent / "tamper_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote tamper manifest with %d entries to %s", len(manifest), manifest_path)
    return manifest
