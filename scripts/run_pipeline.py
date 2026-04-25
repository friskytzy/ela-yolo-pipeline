"""End-to-end ELA + YOLOv8 pipeline driver.

Stages (each can be skipped with --skip-<stage>):

    1. download   -> Roboflow Universe -> dataset/authentic
    2. tamper     -> dataset/tampered/{copy_move,removal}
    3. ela        -> dataset/ela/{authentic,tampered}
    4. labels     -> dataset/yolo/{images,labels}/* + data.yaml
    5. analyse    -> reports/{metrics.json, figures/*.png}
    6. train      -> runs/ela_yolo/train/* (YOLOv8)

Usage::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --num-images 50 --epochs 30
    python scripts/run_pipeline.py --skip-train         # data + analysis only
    python scripts/run_pipeline.py --only train         # use existing data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable when run as a script
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ela_pipeline.utils import (  # noqa: E402
    ensure_dirs,
    load_config,
    seed_everything,
    setup_logging,
)

logger = logging.getLogger("pipeline")

ALL_STAGES = ("download", "tamper", "ela", "labels", "analyse", "train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--num-images", type=int, default=None,
                   help="Override dataset.num_authentic")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override yolo.epochs")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Override yolo.imgsz")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--only", nargs="*", choices=ALL_STAGES,
                   help="Run only the listed stages")
    p.add_argument("--skip", nargs="*", choices=ALL_STAGES, default=[],
                   help="Skip the listed stages")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)

    if args.num_images is not None:
        cfg["dataset"]["num_authentic"] = args.num_images
    if args.epochs is not None:
        cfg["yolo"]["epochs"] = args.epochs
    if args.imgsz is not None:
        cfg["yolo"]["imgsz"] = args.imgsz
    if args.batch is not None:
        cfg["yolo"]["batch"] = args.batch
    if args.device is not None:
        cfg["yolo"]["device"] = args.device

    seed_everything(cfg["dataset"]["random_seed"])

    paths = cfg["paths"]
    ensure_dirs(
        paths["authentic_dir"],
        paths["tampered_copy_move_dir"],
        paths["tampered_removal_dir"],
        paths["ela_authentic_dir"],
        paths["ela_tampered_dir"],
        paths["labels_dir"],
        paths["reports_dir"],
    )

    stages = set(args.only) if args.only else set(ALL_STAGES)
    stages -= set(args.skip)
    logger.info("Running stages: %s", sorted(stages, key=ALL_STAGES.index))

    # 1) Download authentic from Roboflow
    if "download" in stages:
        from ela_pipeline.download import (
            curate_authentic, download_roboflow_dataset,
        )
        raw = download_roboflow_dataset(cfg)
        n = curate_authentic(
            raw,
            paths["authentic_dir"],
            paths["labels_dir"],
            min_resolution=cfg["dataset"]["min_resolution"],
            num_images=cfg["dataset"]["num_authentic"],
            jpeg_quality=cfg["dataset"]["jpeg_quality"],
        )
        logger.info("download stage: %d authentic images", n)

    # 2) Tamper
    if "tamper" in stages:
        from ela_pipeline.tamper import tamper_dataset
        manifest = tamper_dataset(
            auth_dir=paths["authentic_dir"],
            label_dir=paths["labels_dir"],
            cm_out_dir=paths["tampered_copy_move_dir"],
            rm_out_dir=paths["tampered_removal_dir"],
            cfg=cfg,
        )
        logger.info("tamper stage: %d tampered images", len(manifest))

    # 3) ELA
    if "ela" in stages:
        from ela_pipeline.ela import ela_directory
        q = cfg["dataset"]["ela_recompress_quality"]
        s = cfg["dataset"]["ela_scale"]
        ela_directory(paths["authentic_dir"], paths["ela_authentic_dir"],
                      q_recompress=q, scale=s)
        # Tampered ELA = both copy_move + removal collapsed into one folder
        for d in (paths["tampered_copy_move_dir"], paths["tampered_removal_dir"]):
            ela_directory(d, paths["ela_tampered_dir"],
                          q_recompress=q, scale=s)

    # 4) YOLO labels
    if "labels" in stages:
        from ela_pipeline.labels import build_yolo_dataset
        manifest_path = Path(paths["tampered_copy_move_dir"]).parent / "tamper_manifest.json"
        build_yolo_dataset(
            ela_authentic_dir=paths["ela_authentic_dir"],
            ela_tampered_dir=paths["ela_tampered_dir"],
            tamper_manifest_path=manifest_path,
            out_root=paths["yolo_dataset_dir"],
            val_split=cfg["yolo"]["val_split"],
            seed=cfg["dataset"]["random_seed"],
        )

    # 5) Analyse
    if "analyse" in stages:
        from ela_pipeline.analysis import analyse_set
        manifest_path = Path(paths["tampered_copy_move_dir"]).parent / "tamper_manifest.json"
        m = analyse_set(
            authentic_dir=paths["authentic_dir"],
            tampered_dirs=[paths["tampered_copy_move_dir"], paths["tampered_removal_dir"]],
            tamper_manifest_path=manifest_path,
            reports_dir=paths["reports_dir"],
            q_recompress=cfg["dataset"]["ela_recompress_quality"],
            ela_scale=cfg["dataset"]["ela_scale"],
        )
        logger.info("analysis: %s", m)

    # 6) Train
    if "train" in stages:
        from ela_pipeline.train import train_yolo
        data_yaml = Path(paths["yolo_dataset_dir"]) / "data.yaml"
        if not data_yaml.exists():
            raise SystemExit(f"Missing {data_yaml}; run 'labels' stage first.")
        out = train_yolo(
            data_yaml=data_yaml,
            base_model=cfg["yolo"]["base_model"],
            epochs=cfg["yolo"]["epochs"],
            imgsz=cfg["yolo"]["imgsz"],
            batch=cfg["yolo"]["batch"],
            patience=cfg["yolo"]["patience"],
            device=cfg["yolo"]["device"],
            project=cfg["yolo"]["project"],
            name=cfg["yolo"]["name"],
        )
        logger.info("train stage results: %s", out)


if __name__ == "__main__":
    main()
