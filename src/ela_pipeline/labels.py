"""Build the YOLO training set from ELA images + tampered ground truth.

Output structure follows Ultralytics convention so we can feed Ultralytics a
``data.yaml`` directly:

    dataset/yolo/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt   (empty for authentic = pure background)
        labels/val/*.txt
    data.yaml
"""
from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

import yaml

from .utils import BBox, ensure_dirs, list_images, write_yolo_label

logger = logging.getLogger(__name__)


def _empty_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def build_yolo_dataset(
    *,
    ela_authentic_dir: str | Path,
    ela_tampered_dir: str | Path,
    tamper_manifest_path: str | Path,
    out_root: str | Path,
    val_split: float = 0.2,
    seed: int = 42,
    class_names: tuple[str, ...] = ("authentic", "manipulated"),
) -> Path:
    ela_auth = Path(ela_authentic_dir)
    ela_tamp = Path(ela_tampered_dir)
    out_root = Path(out_root)

    images_train = out_root / "images" / "train"
    images_val = out_root / "images" / "val"
    labels_train = out_root / "labels" / "train"
    labels_val = out_root / "labels" / "val"
    ensure_dirs(images_train, images_val, labels_train, labels_val)

    with open(tamper_manifest_path) as f:
        manifest = json.load(f)
    # The manifest is keyed by the *tampered* (non-ELA) path. Map by stem so
    # the ELA file (same stem) can be looked up.
    manifest_by_stem = {Path(k).stem: v for k, v in manifest.items()}

    rng = random.Random(seed)
    items: list[tuple[Path, list[BBox]]] = []

    # Authentic ELA images get an empty label file (= no manipulated regions).
    for p in list_images(ela_auth):
        items.append((p, []))

    # Tampered ELA images get a single bbox of class id 1 (= manipulated).
    for p in list_images(ela_tamp):
        meta = manifest_by_stem.get(p.stem)
        if not meta:
            logger.warning("No manifest entry for tampered ELA %s", p.name)
            continue
        x1, y1, x2, y2 = meta["tampered_xyxy"]
        W, H = meta["image_size"]
        bbox = BBox.from_xyxy(x1, y1, x2, y2, W, H, cls_id=1)
        # Skip obviously broken bboxes
        if bbox.w <= 0 or bbox.h <= 0:
            logger.warning("Bad bbox for %s, skipping", p.name)
            continue
        items.append((p, [bbox]))

    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_split))
    val_items = items[:n_val]
    train_items = items[n_val:]
    logger.info("YOLO split: %d train / %d val", len(train_items), len(val_items))

    for split, split_items, img_dir, lbl_dir in (
        ("train", train_items, images_train, labels_train),
        ("val", val_items, images_val, labels_val),
    ):
        for img_path, boxes in split_items:
            new_img = img_dir / img_path.name
            shutil.copy(img_path, new_img)
            new_lbl = lbl_dir / f"{img_path.stem}.txt"
            if boxes:
                write_yolo_label(boxes, new_lbl)
            else:
                _empty_label(new_lbl)

    data_yaml_path = out_root / "data.yaml"
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    logger.info("Wrote %s", data_yaml_path)
    return data_yaml_path
