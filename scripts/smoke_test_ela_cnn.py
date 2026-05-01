"""Fast synthetic smoke test for the ELA + CNN classifier path."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ela_pipeline.cnn_classifier import (  # noqa: E402
    ElaCnnConfig,
    TrainingConfig,
    compute_ela_image,
    prepare_ela_dataset,
)


def make_real(index: int, size: int) -> Image.Image:
    rng = np.random.default_rng(index)
    base = rng.normal(130, 22, size=(size, size, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(base, mode="RGB")


def make_fake(index: int, size: int) -> Image.Image:
    image = np.asarray(make_real(index, size)).copy()
    patch = np.asarray(make_real(index + 500, size // 3))
    offset = size // 5
    image[offset:offset + patch.shape[0], offset:offset + patch.shape[1]] = patch
    return Image.fromarray(image, mode="RGB")


def create_dataset(root: Path, n: int, size: int) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    for index in range(n):
        make_real(index, size).save(root / "real" / f"real_{index:03d}.jpg", quality=95)
        if index % 2 == 0:
            make_fake(index, size).save(root / "fake" / f"fake_{index:03d}.png")
        else:
            make_fake(index, size).save(root / "fake" / f"fake_{index:03d}.jpg", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset_ela_cnn_smoke")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.root)
    source_root = root / "source"
    ela_root = root / "ela"
    create_dataset(source_root, args.n, args.size)

    config = ElaCnnConfig(image_size=96, min_side=96, source_jpeg_quality=95, ela_recompress_quality=90)
    training = TrainingConfig(epochs=1, batch_size=2, val_split=0.25, use_pretrained=False)
    counts = prepare_ela_dataset(source_root, ela_root, config, training)
    assert counts["train"]["REAL"] > 0
    assert counts["train"]["FAKE"] > 0
    assert counts["val"]["REAL"] > 0
    assert counts["val"]["FAKE"] > 0

    low_res = Image.new("RGB", (32, 48), color=(120, 100, 90))
    ela = compute_ela_image(low_res, config)
    assert min(ela.size) >= config.min_side
    print("ELA-CNN smoke test passed:", counts)


if __name__ == "__main__":
    main()
