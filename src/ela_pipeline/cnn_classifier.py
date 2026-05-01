"""ELA + CNN binary classifier utilities for Colab and Gradio."""
from __future__ import annotations

import io
import json
import logging
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_ALIASES = {"real", "authentic", "original", "asli", "genuine", "00_real"}
FAKE_ALIASES = {"fake", "tampered", "manipulated", "manipulasi", "forged", "01_fake"}
OUTPUT_CLASS_DIRS = {"REAL": "00_real", "FAKE": "01_fake"}


@dataclass(frozen=True)
class ElaCnnConfig:
    image_size: int = 224
    min_side: int = 224
    source_jpeg_quality: int = 95
    ela_recompress_quality: int = 90
    ela_scale: float = 12.0
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-4
    val_split: float = 0.2
    use_pretrained: bool = True
    use_augmentation: bool = True
    patience: int = 4


def display_class_name(class_name: str) -> str:
    if class_name.endswith("real"):
        return "REAL"
    if class_name.endswith("fake"):
        return "FAKE"
    return class_name.upper()


def list_supported_images(directory: Path) -> list[Path]:
    return [
        path for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def ensure_min_resolution(image: Image.Image, min_side: int) -> Image.Image:
    width, height = image.size
    current_min_side = min(width, height)
    if current_min_side >= min_side:
        return image
    scale = min_side / float(current_min_side)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def compute_ela_image(image: Image.Image, config: ElaCnnConfig) -> Image.Image:
    image = ensure_min_resolution(image.convert("RGB"), config.min_side)
    canonical = jpeg_roundtrip(image, config.source_jpeg_quality)
    recompressed = jpeg_roundtrip(canonical, config.ela_recompress_quality)

    arr_a = np.asarray(canonical, dtype=np.int16)
    arr_b = np.asarray(recompressed, dtype=np.int16)
    diff = np.abs(arr_a - arr_b).astype(np.float32) * float(config.ela_scale)
    out = np.clip(diff, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _normalise_dir_name(path: Path) -> str:
    return path.name.lower().replace("-", "_").replace(" ", "_")


def _class_for_dir(path: Path) -> str | None:
    name = _normalise_dir_name(path)
    if name in REAL_ALIASES:
        return "REAL"
    if name in FAKE_ALIASES:
        return "FAKE"
    return None


def _direct_class_dirs(container: Path) -> dict[str, list[Path]]:
    class_dirs: dict[str, list[Path]] = {"REAL": [], "FAKE": []}
    for child in sorted(container.iterdir()):
        if not child.is_dir():
            continue
        class_name = _class_for_dir(child)
        if class_name is not None:
            class_dirs[class_name].append(child)
    return class_dirs


def discover_dataset(data_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Discover image folders with optional train/val/test split directories."""
    data_dir = data_dir.expanduser().resolve()
    split_aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    discovered: dict[str, dict[str, list[Path]]] = {}

    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and _normalise_dir_name(child) in split_aliases:
            split_name = split_aliases[_normalise_dir_name(child)]
            split_dirs = _direct_class_dirs(child)
            if split_dirs["REAL"] and split_dirs["FAKE"]:
                discovered[split_name] = split_dirs

    if discovered:
        return discovered

    root_dirs = _direct_class_dirs(data_dir)
    if root_dirs["REAL"] and root_dirs["FAKE"]:
        return {"all": root_dirs}

    expected = "real/ and fake/ (or authentic/ and tampered/) under the dataset root"
    raise ValueError(f"Could not find dataset class folders. Expected {expected}: {data_dir}")


def _split_paths(paths: list[Path], val_split: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_split))) if len(shuffled) > 1 else 0
    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    if not train_paths and val_paths:
        train_paths, val_paths = val_paths, []
    return train_paths, val_paths


def _copy_as_ela(source_paths: list[Path], out_dir: Path, config: ElaCnnConfig) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_path in source_paths:
        image = load_rgb_image(source_path)
        ela = compute_ela_image(image, config)
        out_path = out_dir / f"{source_path.stem}.png"
        suffix = 1
        while out_path.exists():
            out_path = out_dir / f"{source_path.stem}_{suffix}.png"
            suffix += 1
        ela.save(out_path, format="PNG")
        count += 1
    return count


def prepare_ela_dataset(
    data_dir: str | Path,
    out_dir: str | Path,
    ela_config: ElaCnnConfig,
    training_config: TrainingConfig,
) -> dict[str, dict[str, int]]:
    """Generate train/val ELA image folders for Keras binary classification."""
    source_root = Path(data_dir)
    output_root = Path(out_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    discovered = discover_dataset(source_root)
    plan: dict[str, dict[str, list[Path]]] = {"train": {"REAL": [], "FAKE": []}, "val": {"REAL": [], "FAKE": []}}

    if "all" in discovered:
        for class_name in ("REAL", "FAKE"):
            paths: list[Path] = []
            for class_dir in discovered["all"][class_name]:
                paths.extend(list_supported_images(class_dir))
            train_paths, val_paths = _split_paths(paths, training_config.val_split, ela_config.seed)
            plan["train"][class_name] = train_paths
            plan["val"][class_name] = val_paths
    else:
        for split_name in ("train", "val"):
            split_dirs = discovered.get(split_name, {"REAL": [], "FAKE": []})
            for class_name in ("REAL", "FAKE"):
                paths = []
                for class_dir in split_dirs[class_name]:
                    paths.extend(list_supported_images(class_dir))
                plan[split_name][class_name] = paths

    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    for split_name in ("train", "val"):
        for class_name in ("REAL", "FAKE"):
            class_dir = output_root / split_name / OUTPUT_CLASS_DIRS[class_name]
            counts[split_name][class_name] = _copy_as_ela(plan[split_name][class_name], class_dir, ela_config)

    for split_name in ("train", "val"):
        if counts[split_name]["REAL"] == 0 or counts[split_name]["FAKE"] == 0:
            raise ValueError(f"Missing REAL or FAKE samples after preparing {split_name}: {counts[split_name]}")

    metadata = {
        "ela_config": asdict(ela_config),
        "training_config": asdict(training_config),
        "counts": counts,
        "source_root": str(source_root),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Prepared ELA dataset at %s: %s", output_root, counts)
    return counts


def build_classifier(config: ElaCnnConfig, training_config: TrainingConfig):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3))
    x = inputs
    if training_config.use_augmentation:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.04),
                tf.keras.layers.RandomZoom(0.08),
                tf.keras.layers.RandomContrast(0.12),
            ],
            name="augmentation",
        )
        x = augmentation(x)

    if training_config.use_pretrained:
        try:
            base = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=(config.image_size, config.image_size, 3),
            )
            base.trainable = False
            x = tf.keras.applications.efficientnet.preprocess_input(x)
            x = base(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        except Exception as exc:
            logger.warning("EfficientNetB0 unavailable, falling back to small CNN: %s", exc)
            training_config = TrainingConfig(
                epochs=training_config.epochs,
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
                val_split=training_config.val_split,
                use_pretrained=False,
                use_augmentation=training_config.use_augmentation,
                patience=training_config.patience,
            )

    if not training_config.use_pretrained:
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="ela_cnn_binary_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_image_datasets(ela_root: str | Path, image_size: int, batch_size: int):
    import tensorflow as tf

    root = Path(ela_root)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root / "train",
        labels="inferred",
        label_mode="binary",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root / "val",
        labels="inferred",
        label_mode="binary",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), class_names


def train_classifier(
    ela_root: str | Path,
    model_path: str | Path,
    ela_config: ElaCnnConfig,
    training_config: TrainingConfig,
) -> tuple[object, list[str]]:
    import tensorflow as tf

    train_ds, val_ds, class_names = load_image_datasets(
        ela_root, ela_config.image_size, training_config.batch_size,
    )
    model = build_classifier(ela_config, training_config)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=training_config.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, training_config.patience // 2),
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=training_config.epochs, callbacks=callbacks)
    model.save(model_path)
    class_names_path = Path(model_path).with_name("class_names.json")
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    logger.info("Saved model to %s and class names to %s", model_path, class_names_path)
    return model, class_names


def _manual_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        matrix[true_label, pred_label] += 1
    return matrix


def evaluate_classifier(model, ela_root: str | Path, image_size: int, batch_size: int, reports_dir: str | Path) -> dict[str, object]:
    import matplotlib.pyplot as plt

    _, val_ds, class_names = load_image_datasets(ela_root, image_size, batch_size)
    true_labels: list[int] = []
    predicted_probs: list[float] = []
    for batch_images, batch_labels in val_ds:
        probs = model.predict(batch_images, verbose=0).reshape(-1)
        labels = np.asarray(batch_labels).reshape(-1)
        predicted_probs.extend(float(prob) for prob in probs)
        true_labels.extend(int(label) for label in labels)

    y_true = np.asarray(true_labels, dtype=np.int64)
    y_prob = np.asarray(predicted_probs, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    accuracy = float((y_true == y_pred).mean())
    matrix = _manual_confusion_matrix(y_true, y_pred)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    labels = [display_class_name(name) for name in class_names]

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"ELA-CNN Confusion Matrix (accuracy={accuracy:.3f})")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(reports_path / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    metrics: dict[str, object] = {
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }
    (reports_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def predict_image(model, image: Image.Image, class_names: list[str], config: ElaCnnConfig) -> tuple[Image.Image, str, float, dict[str, float]]:
    ela = compute_ela_image(image, config)
    resized = ela.resize((config.image_size, config.image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)[None, ...]
    prob_positive = float(model.predict(arr, verbose=0).reshape(-1)[0])
    predicted_index = 1 if prob_positive >= 0.5 else 0
    confidence = prob_positive if predicted_index == 1 else 1.0 - prob_positive
    label = display_class_name(class_names[predicted_index])
    probabilities = {
        display_class_name(class_names[0]): 1.0 - prob_positive,
        display_class_name(class_names[1]): prob_positive,
    }
    return ela, label, confidence, probabilities


def launch_gradio(
    model_path: str | Path,
    class_names_path: str | Path,
    ela_config: ElaCnnConfig,
    *,
    share: bool = True,
) -> None:
    import gradio as gr
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(Path(class_names_path).read_text(encoding="utf-8"))

    def infer(image: Image.Image | None):
        if image is None:
            return None, "Upload an image first.", {}
        ela, label, confidence, probabilities = predict_image(model, image.convert("RGB"), class_names, ela_config)
        return ela, f"{label} ({confidence:.2%})", probabilities

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Image(type="pil", label="Upload image (JPEG/PNG/low-res supported)"),
        outputs=[
            gr.Image(type="pil", label="ELA preview"),
            gr.Textbox(label="Prediction"),
            gr.Label(label="Class confidence", num_top_classes=2),
        ],
        title="ELA + CNN Image Manipulation Detector",
        description=(
            "Upload an image to see the generated ELA preview and a binary "
            "REAL/FAKE prediction with confidence."
        ),
    )
    demo.launch(share=share, debug=True)
