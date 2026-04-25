"""Synthetic smoke test that exercises every module WITHOUT Roboflow.

Generates a handful of random JPEGs + YOLO labels, runs tamper -> ELA ->
labels -> analysis on them, and trains YOLO for one epoch on CPU. This is
how the pipeline is validated in CI without external API access.
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ela_pipeline.analysis import analyse_set  # noqa: E402
from ela_pipeline.ela import ela_directory  # noqa: E402
from ela_pipeline.labels import build_yolo_dataset  # noqa: E402
from ela_pipeline.tamper import tamper_dataset  # noqa: E402
from ela_pipeline.utils import (  # noqa: E402
    ensure_dirs,
    save_jpeg,
    seed_everything,
    setup_logging,
    write_yolo_label,
    BBox,
)


def make_synthetic_image(rng: random.Random, size: int = 1024) -> np.ndarray:
    """Texture-rich random image with a few coloured rectangular 'objects'."""
    base = (rng.random() * 80 + 60)
    img = (np.random.RandomState(rng.randint(0, 1_000_000))
           .normal(loc=base, scale=35, size=(size, size, 3))
           .clip(0, 255)
           .astype(np.uint8))
    # Sprinkle a few coloured blocks so YOLO bboxes are meaningful regions.
    out = img.copy()
    boxes = []
    for _ in range(rng.randint(2, 4)):
        bw = rng.randint(int(0.10 * size), int(0.25 * size))
        bh = rng.randint(int(0.10 * size), int(0.25 * size))
        x1 = rng.randint(0, size - bw - 1)
        y1 = rng.randint(0, size - bh - 1)
        colour = np.array([rng.randint(40, 220) for _ in range(3)], dtype=np.uint8)
        # noise overlay so it's not flat
        noise = (np.random.RandomState(rng.randint(0, 1_000_000))
                 .normal(0, 25, size=(bh, bw, 3)).astype(np.int16))
        block = np.clip(colour.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out[y1:y1 + bh, x1:x1 + bw] = block
        boxes.append((x1, y1, x1 + bw, y1 + bh))
    return out, boxes


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="number of synthetic images")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--root", default="dataset_smoke")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    seed_everything(7)
    rng = random.Random(7)

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    auth = root / "authentic"
    lbl = root / "labels"
    cm = root / "tampered" / "copy_move"
    rm = root / "tampered" / "removal"
    ela_a = root / "ela" / "authentic"
    ela_t = root / "ela" / "tampered"
    yolo = root / "yolo"
    reports = root / "reports"
    ensure_dirs(auth, lbl, cm, rm, ela_a, ela_t, yolo, reports)

    # 1) Synthetic authentic images
    for i in range(args.n):
        img, boxes = make_synthetic_image(rng, size=args.size)
        stem = f"auth_{i:05d}"
        save_jpeg(Image.fromarray(img), auth / f"{stem}.jpg", quality=95)
        bboxes = [BBox.from_xyxy(*b, args.size, args.size, cls_id=0) for b in boxes]
        write_yolo_label(bboxes, lbl / f"{stem}.txt")

    # 2) Tamper
    cfg = {
        "dataset": {
            "random_seed": 7, "jpeg_quality": 95,
            "copy_move_per_image": 1, "removal_per_image": 1,
        },
        "tamper": {
            "copy_move": {
                "blur_kernel": 3, "blur_sigma": 0.8,
                "min_bbox_area_frac": 0.005, "max_bbox_area_frac": 0.5,
                "min_shift_frac": 0.10, "max_shift_frac": 0.40,
            },
            "removal": {
                "method": "telea", "inpaint_radius": 5, "mask_dilate": 7,
                "min_bbox_area_frac": 0.005, "max_bbox_area_frac": 0.5,
            },
        },
    }
    manifest = tamper_dataset(
        auth_dir=auth, label_dir=lbl, cm_out_dir=cm, rm_out_dir=rm, cfg=cfg,
    )
    assert manifest, "Tamper manifest is empty"

    # 3) ELA
    ela_directory(auth, ela_a, q_recompress=90, scale=12.0)
    ela_directory(cm, ela_t, q_recompress=90, scale=12.0)
    ela_directory(rm, ela_t, q_recompress=90, scale=12.0)

    # 4) Labels
    manifest_path = root / "tampered" / "tamper_manifest.json"
    build_yolo_dataset(
        ela_authentic_dir=ela_a,
        ela_tampered_dir=ela_t,
        tamper_manifest_path=manifest_path,
        out_root=yolo,
        val_split=0.25,
        seed=7,
    )

    # 5) Analyse
    metrics = analyse_set(
        authentic_dir=auth,
        tampered_dirs=[cm, rm],
        tamper_manifest_path=manifest_path,
        reports_dir=reports,
        q_recompress=90,
        ela_scale=12.0,
        sample_visuals=2,
    )
    print("METRICS:", metrics)

    if not args.skip_train:
        from ela_pipeline.train import train_yolo
        out = train_yolo(
            data_yaml=yolo / "data.yaml",
            base_model="yolov8n.pt",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=2,
            patience=5,
            device="cpu",
            project=str(root / "runs"),
            name="smoke",
        )
        print("TRAIN:", out)


if __name__ == "__main__":
    main()
