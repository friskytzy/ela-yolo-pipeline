"""Shared utilities: config loader, IO helpers, deterministic randomness."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def list_images(directory: str | Path, exts: tuple[str, ...] = (".jpg", ".jpeg")) -> list[Path]:
    directory = Path(directory)
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in exts)


def save_jpeg(img: np.ndarray | Image.Image, path: str | Path, quality: int = 95) -> None:
    """Save an image as JPEG with EXACT quality control.

    Uses PIL exclusively so authentic and tampered files share the same
    encoder/quantisation tables — a critical control for ELA experiments.
    Subsampling is forced to 4:4:4 (no chroma subsampling) for parity.
    """
    if isinstance(img, np.ndarray):
        # OpenCV BGR -> PIL RGB conversion if 3-channel
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1]
        img = Image.fromarray(img)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=quality, subsampling=0, optimize=False)


def load_image_pil(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_image_np(path: str | Path) -> np.ndarray:
    """Load image as RGB uint8 numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


@dataclass
class BBox:
    """YOLO-format bbox: cls, xc, yc, w, h all normalised in [0, 1]."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        x1 = int((self.xc - self.w / 2) * img_w)
        y1 = int((self.yc - self.h / 2) * img_h)
        x2 = int((self.xc + self.w / 2) * img_w)
        y2 = int((self.yc + self.h / 2) * img_h)
        return (
            max(0, x1),
            max(0, y1),
            min(img_w, x2),
            min(img_h, y2),
        )

    def area_frac(self) -> float:
        return float(self.w * self.h)

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, cls_id: int = 1) -> "BBox":
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        return cls(cls_id, xc, yc, w, h)


def parse_yolo_label(label_path: str | Path) -> list[BBox]:
    """Parse a YOLO .txt label file into BBox objects. Missing file -> []."""
    p = Path(label_path)
    if not p.exists():
        return []
    boxes: list[BBox] = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        boxes.append(BBox(cls_id, xc, yc, w, h))
    return boxes


def write_yolo_label(boxes: list[BBox], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b.cls} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}\n")
