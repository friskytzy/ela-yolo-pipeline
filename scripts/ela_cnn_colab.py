"""Colab-friendly ELA + CNN binary classifier with Gradio inference."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ela_pipeline.cnn_classifier import (  # noqa: E402
    ElaCnnConfig,
    TrainingConfig,
    evaluate_classifier,
    launch_gradio,
    prepare_ela_dataset,
    train_classifier,
)
from ela_pipeline.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder with real/fake class subfolders")
    parser.add_argument("--work-dir", default="dataset_ela_cnn", help="Generated ELA train/val dataset root")
    parser.add_argument("--reports-dir", default="reports/ela_cnn", help="Metrics and confusion matrix output")
    parser.add_argument("--model-path", default="runs/ela_cnn/model.keras", help="Saved Keras model path")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-side", type=int, default=224, help="Upscale images whose shortest side is smaller")
    parser.add_argument("--source-jpeg-quality", type=int, default=95, help="Canonical JPEG quality for PNG/non-JPEG inputs")
    parser.add_argument("--ela-quality", type=int, default=90, help="JPEG quality for ELA recompression")
    parser.add_argument("--ela-scale", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Use a compact CNN instead of EfficientNetB0")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate ELA folders")
    parser.add_argument("--launch-gradio", action="store_true", help="Launch GUI after training")
    parser.add_argument("--no-share", action="store_true", help="Disable Gradio public share link")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    ela_config = ElaCnnConfig(
        image_size=args.image_size,
        min_side=args.min_side,
        source_jpeg_quality=args.source_jpeg_quality,
        ela_recompress_quality=args.ela_quality,
        ela_scale=args.ela_scale,
        seed=args.seed,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        use_pretrained=not args.no_pretrained,
        use_augmentation=not args.no_augmentation,
    )

    counts = prepare_ela_dataset(args.data_dir, args.work_dir, ela_config, training_config)
    logger.info("Prepared ELA dataset counts: %s", counts)
    if args.prepare_only:
        return

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model, _ = train_classifier(args.work_dir, model_path, ela_config, training_config)
    metrics = evaluate_classifier(model, args.work_dir, args.image_size, args.batch_size, args.reports_dir)
    logger.info("Accuracy: %.4f", metrics["accuracy"])

    if args.launch_gradio:
        launch_gradio(
            model_path=model_path,
            class_names_path=model_path.with_name("class_names.json"),
            ela_config=ela_config,
            share=not args.no_share,
        )


if __name__ == "__main__":
    main()
