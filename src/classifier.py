"""Whole-image road-damage classifier (Keras MobileNetV2).

The model takes one photo and returns a probability for each of the four damage
types. Unlike the YOLO detector it does NOT locate defects (no bounding boxes):
it answers "which kind of damage is in this photo?".

Keras 3 runs on the PyTorch backend here, so TensorFlow is not required.
Settings (model path, input size, preprocessing, output class order) are in the
``classifier`` section of config.yaml.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("KERAS_BACKEND", "torch")

logger = logging.getLogger(__name__)

PREPROCESSING = {
    "mobilenet_v2": lambda x: x / 127.5 - 1.0,   # pixels to [-1, 1]
    "unit": lambda x: x / 255.0,                  # pixels to [0, 1]
    "none": lambda x: x,
}


class DamageClassifier:
    """Loads a .keras model (also accepts a '.keras.zip' file name) and classifies images."""

    def __init__(self, model_path: str | Path, class_names: list[str], image_size: int = 224,
                 preprocessing: str = "mobilenet_v2") -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Classifier model not found: {path}")
        if preprocessing not in PREPROCESSING:
            raise ValueError(f"Unknown preprocessing '{preprocessing}'. Use one of {list(PREPROCESSING)}.")
        try:
            import keras
        except ImportError as exc:
            raise ImportError("Keras is not installed. Run: pip install -r requirements.txt") from exc

        self.class_names = list(class_names)
        self.image_size = int(image_size)
        self.preprocess = PREPROCESSING[preprocessing]
        self.model = keras.saving.load_model(self._as_keras_file(path), compile=False)
        outputs = self.model.output_shape[-1]
        if outputs != len(self.class_names):
            raise ValueError(f"Model has {outputs} outputs but config lists {len(self.class_names)} class names.")
        logger.info("Loaded classifier %s (%d classes)", path, outputs)

    @staticmethod
    def _as_keras_file(path: Path) -> Path:
        """Keras only loads files ending in .keras; copy e.g. 'model.keras.zip' to a temp .keras file."""
        if path.suffix == ".keras":
            return path
        target = Path(tempfile.gettempdir()) / f"{path.name.split('.')[0]}_{path.stat().st_mtime_ns}.keras"
        if not target.is_file():
            shutil.copy2(path, target)
        return target

    def _prepare(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """RGB PIL image or BGR array -> preprocessed (1, size, size, 3) float32 batch."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        rgb = ImageOps.exif_transpose(image).convert("RGB").resize((self.image_size, self.image_size),
                                                                    Image.BILINEAR)
        return self.preprocess(np.asarray(rgb, dtype=np.float32))[None]

    def predict(self, image: np.ndarray | Image.Image) -> list[dict[str, Any]]:
        """Return all classes sorted by probability: [{"class_name", "confidence"}, ...]."""
        probs = np.asarray(self.model.predict(self._prepare(image), verbose=0))[0]
        ranked = sorted(zip(self.class_names, probs.tolist()), key=lambda t: t[1], reverse=True)
        return [{"class_name": name, "confidence": round(float(p), 4)} for name, p in ranked]
