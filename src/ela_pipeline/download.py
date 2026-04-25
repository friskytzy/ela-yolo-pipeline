"""Download authentic dataset from Roboflow Universe with strict controls.

We:
1) Pull a YOLO-format dataset version
2) Filter to JPEG images that meet the minimum resolution constraint
3) Re-encode ONCE with PIL @ JPEG quality 95 (subsampling=0) into
   ``dataset/authentic`` so the encoder/quantisation tables are uniform
4) Carry over the YOLO label files so tampering can use real bboxes
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .utils import ensure_dirs, save_jpeg

logger = logging.getLogger(__name__)


class RoboflowDownloadError(RuntimeError):
    pass


def _download_one(workspace: str, project: str, version: int, fmt: str, out_dir: Path, api_key: str) -> Path:
    from roboflow import Roboflow  # lazy import; library is heavy

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Roboflow downloads into its own folder; we point it at our cache.
    cwd = Path.cwd()
    os.chdir(out_dir)
    try:
        ds = ver.download(fmt)
    finally:
        os.chdir(cwd)
    return Path(ds.location)


def download_roboflow_dataset(cfg: dict, raw_cache: str | Path = "dataset/_roboflow_raw") -> Path:
    """Try each candidate dataset; return the first that succeeds."""
    api_key_env = cfg["roboflow"]["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RoboflowDownloadError(
            f"Missing API key: set env var {api_key_env}."
        )

    raw_cache = Path(raw_cache)
    raw_cache.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for cand in cfg["roboflow"]["candidates"]:
        try:
            logger.info(
                "Trying Roboflow dataset %s/%s v%s (%s)",
                cand["workspace"], cand["project"], cand["version"], cand["format"],
            )
            target = raw_cache / f"{cand['workspace']}__{cand['project']}__v{cand['version']}"
            if target.exists() and any(target.iterdir()):
                # Roboflow nests one level deeper, e.g. target/vehicles-2/.
                resolved = _resolve_dataset_root(target)
                if resolved is not None:
                    logger.info("Cache hit: %s", resolved)
                    return resolved
                logger.warning("Cache present at %s but no train/ split found; redownloading", target)
            location = _download_one(
                cand["workspace"], cand["project"], int(cand["version"]),
                cand["format"], target, api_key,
            )
            logger.info("Downloaded into %s", location)
            return location
        except Exception as e:  # noqa: BLE001 — try next candidate
            logger.warning("Candidate failed: %s", e)
            last_err = e
    raise RoboflowDownloadError(f"All candidates failed. Last error: {last_err}")


def _resolve_dataset_root(target: Path) -> Path | None:
    """Find the directory that contains ``train/images`` (or similar)."""
    if (target / "train" / "images").exists():
        return target
    for child in target.iterdir():
        if child.is_dir() and (child / "train" / "images").exists():
            return child
    return None


def _iter_image_label_pairs(roboflow_dir: Path):
    """Walk the Roboflow YOLO export structure and yield (img, label) pairs.

    Roboflow YOLO exports look like::

        roboflow_dir/
            train/images/*.jpg
            train/labels/*.txt
            valid/images/*.jpg
            valid/labels/*.txt
            test/images/*.jpg
            test/labels/*.txt
            data.yaml
    """
    for split in ("train", "valid", "test", "."):
        img_dir = roboflow_dir / split / "images"
        lbl_dir = roboflow_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = lbl_dir / f"{img_path.stem}.txt"
            yield img_path, (label_path if label_path.exists() else None)


def curate_authentic(
    roboflow_dir: Path,
    out_img_dir: str | Path,
    out_label_dir: str | Path,
    *,
    min_resolution: int,
    num_images: int,
    jpeg_quality: int,
) -> int:
    """Re-encode authentic images at fixed JPEG quality and copy labels.

    Returns the number of authentic images written.
    """
    out_img_dir = Path(out_img_dir)
    out_label_dir = Path(out_label_dir)
    ensure_dirs(out_img_dir, out_label_dir)

    accepted = 0
    smaller_seen: list[tuple[Path, Path | None, int, int]] = []

    pairs = list(_iter_image_label_pairs(roboflow_dir))
    logger.info("Found %d candidate image/label pairs in raw export", len(pairs))

    for img_path, label_path in tqdm(pairs, desc="curate authentic"):
        if accepted >= num_images:
            break
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                if min(w, h) < min_resolution:
                    smaller_seen.append((img_path, label_path, w, h))
                    continue
                im = im.convert("RGB")
                stem = f"auth_{accepted:05d}"
                save_jpeg(im, out_img_dir / f"{stem}.jpg", quality=jpeg_quality)
            if label_path is not None:
                shutil.copy(label_path, out_label_dir / f"{stem}.txt")
            accepted += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping %s: %s", img_path, e)

    if accepted < num_images and smaller_seen:
        logger.warning(
            "Only %d/%d images met %dpx; falling back to largest available below threshold",
            accepted, num_images, min_resolution,
        )
        # Sort fallback by largest min-side first; keep all original aspect ratios.
        smaller_seen.sort(key=lambda x: -min(x[2], x[3]))
        for img_path, label_path, w, h in smaller_seen:
            if accepted >= num_images:
                break
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    stem = f"auth_{accepted:05d}"
                    save_jpeg(im, out_img_dir / f"{stem}.jpg", quality=jpeg_quality)
                if label_path is not None:
                    shutil.copy(label_path, out_label_dir / f"{stem}.txt")
                accepted += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("Fallback skip %s: %s", img_path, e)

    logger.info("Wrote %d authentic images to %s", accepted, out_img_dir)
    return accepted
