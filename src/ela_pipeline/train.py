"""Thin wrapper around Ultralytics YOLOv8 training + evaluation."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train_yolo(
    *,
    data_yaml: str | Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 15,
    device: str = "auto",
    project: str = "runs/ela_yolo",
    name: str = "train",
) -> dict:
    """Train a YOLOv8 model and return key metrics."""
    from ultralytics import YOLO  # lazy import

    model = YOLO(base_model)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    if device != "auto":
        train_kwargs["device"] = device
    results = model.train(**train_kwargs)

    # Validation
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        project=project,
        name=f"{name}_val",
        exist_ok=True,
    )

    out = {
        "save_dir": str(getattr(results, "save_dir", "")),
        "best": str(Path(getattr(results, "save_dir", "")) / "weights" / "best.pt"),
        "metrics": {
            "mAP50": float(getattr(metrics.box, "map50", float("nan"))),
            "mAP50_95": float(getattr(metrics.box, "map", float("nan"))),
            "precision": float(getattr(metrics.box, "mp", float("nan"))),
            "recall": float(getattr(metrics.box, "mr", float("nan"))),
        },
    }
    logger.info("Training complete: %s", out)
    return out
