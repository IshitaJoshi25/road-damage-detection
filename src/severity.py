"""Severity estimation (Low / Medium / High) for detected road defects.

Why severity needs special care
-------------------------------
RDD2022 provides damage *type* labels (D00, D10, D20, D40, ...) and bounding
boxes only. It contains NO Low/Medium/High severity ground truth. Two
estimators are therefore provided, selected by ``severity.method`` in config.yaml:

1. ``SeverityPredictor`` - a trained image classifier (MobileNetV3 / ResNet18).
   This is the proper solution. Its training data comes from our own labeling:
     a. ``python src/crop_defects.py --source data/yolo/images/train --ground-truth --save-crops``
     b. ``streamlit run label_severity.py`` (rules in docs/severity_rubric.md)
     c. ``python src/train_severity.py`` -> models/severity_classifier.pt

2. ``HeuristicSeverityEstimator`` - a transparent rule-based ESTIMATE used until
   a trained classifier exists. It combines two measurable image cues:
     * size   - how large the defect is relative to the image (crack length for
                linear cracks, area for potholes / alligator cracking);
     * density - fraction of dark pixels inside the box (a proxy for crack
                width / spalling or pothole depth / shadow).
   The thresholds are documented starting values in config.yaml, NOT learned
   from ground truth. Its output is always marked ``method="heuristic"`` and the
   UI labels it as an estimate. train_severity.py reports how well it agrees
   with human labels once they exist.

``create_severity_estimator(cfg)`` returns the right estimator: with
``method: auto`` the trained model is used when its file exists, otherwise the
heuristic.

Checkpoint format for the trained model (a ``torch.save``'d dict)::

    {
        "arch": "mobilenet_v3_small" | "mobilenet_v3_large" | "resnet18",
        "state_dict": model.state_dict(),
        "labels": ["Low", "Medium", "High"],   # index order of the output logits
        "image_size": 224,                     # optional, default 224
        "mean": [0.485, 0.456, 0.406],         # optional, ImageNet defaults
        "std": [0.229, 0.224, 0.225],          # optional
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ("Low", "Medium", "High")
SUPPORTED_ARCHS = ("mobilenet_v3_small", "mobilenet_v3_large", "resnet18")
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DENSITY_BLUR_KERNEL = (5, 5)
METHOD_MODEL = "model"
METHOD_HEURISTIC = "heuristic"


def not_available() -> dict[str, Any]:
    """Result used when no severity estimate can be made."""
    return {"severity": None, "confidence": None, "method": None}


def build_classifier(arch: str, num_classes: int, pretrained: bool = False):
    """Create a torchvision classifier with ``num_classes`` outputs (optionally ImageNet-pretrained)."""
    import torch.nn as nn
    from torchvision import models

    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported severity architecture '{arch}'. Use one of {SUPPORTED_ARCHS}.")
    model = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
    if arch == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def preprocess_crop(crop_bgr: np.ndarray, image_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """BGR crop -> normalised float32 CHW array (shared by training and inference)."""
    import cv2

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    array = (resized.astype(np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return array.transpose(2, 0, 1)


def torch_device(device: str | None) -> str:
    """'cpu' / '0' / 'cuda:0' -> a torch device string."""
    if device in (None, "cpu"):
        return "cpu"
    return f"cuda:{device}" if str(device).isdigit() else str(device)


class SeverityPredictor:
    """Trained severity classifier for a single defect crop.

    If no model path is given, or the file does not exist, the predictor is
    "unavailable" and ``predict`` returns ``severity=None``.
    """

    method = METHOD_MODEL

    def __init__(self, model_path: str | Path | None = None, device: str = "cpu") -> None:
        self.model_path = Path(model_path) if model_path else None
        self.device = torch_device(device)
        self.model = None
        self.labels: list[str] = list(SEVERITY_LEVELS)
        self.image_size = DEFAULT_IMAGE_SIZE
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        if self.model_path is None:
            return
        if not self.model_path.is_file():
            logger.warning("Severity model not found at %s; model-based severity disabled.", self.model_path)
            return
        self._load()

    @property
    def available(self) -> bool:
        """True when a trained severity model is loaded."""
        return self.model is not None

    def _load(self) -> None:
        import torch

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "arch" not in checkpoint:
            raise ValueError(f"{self.model_path} is not a severity checkpoint (see src/severity.py docstring).")
        self.labels = list(checkpoint.get("labels", SEVERITY_LEVELS))
        self.image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
        self.mean = tuple(checkpoint.get("mean", IMAGENET_MEAN))
        self.std = tuple(checkpoint.get("std", IMAGENET_STD))
        model = build_classifier(checkpoint["arch"], len(self.labels))
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.to(self.device).eval()
        logger.info("Loaded severity model %s (%s, labels=%s)", self.model_path, checkpoint["arch"], self.labels)

    def predict(self, crop: np.ndarray, detection: dict | None = None,
                image_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """Classify one BGR crop. ``detection``/``image_shape`` are accepted for interface parity."""
        if not self.available or crop is None or crop.size == 0:
            return not_available()
        import torch

        tensor = torch.from_numpy(preprocess_crop(crop, self.image_size, self.mean, self.std)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor.to(self.device)), dim=1)[0]
        idx = int(probs.argmax())
        return {"severity": self.labels[idx], "confidence": round(float(probs[idx]), 4), "method": self.method}


def dark_pixel_fraction(crop_bgr: np.ndarray, dark_delta: float) -> float:
    """Fraction of pixels noticeably darker than the crop's median grey level."""
    import cv2

    gray = cv2.GaussianBlur(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), DENSITY_BLUR_KERNEL, 0)
    return float((gray < np.median(gray) - dark_delta).mean())


class HeuristicSeverityEstimator:
    """Transparent rule-based severity ESTIMATE (see module docstring).

    score = size_weight * size_score + (1 - size_weight) * density_score, both in [0, 1];
    score < thresholds[0] -> Low, < thresholds[1] -> Medium, otherwise High.
    """

    method = METHOD_HEURISTIC
    available = True

    def __init__(self, params: dict[str, Any]) -> None:
        self.dark_delta = float(params["dark_delta"])
        self.size_ref = {k: float(v) for k, v in params["size_ref"].items()}
        self.density_ref = float(params["density_ref"])
        self.size_weight = float(params["size_weight"])
        self.low_max, self.medium_max = (float(t) for t in params["thresholds"])
        self.linear_classes = set(params.get("linear_classes", []))

    def features(self, crop: np.ndarray, detection: dict, image_shape: tuple[int, int]) -> dict[str, float]:
        """Size and density cues, each normalised to [0, 1]."""
        img_h, img_w = image_shape[:2]
        x1, y1, x2, y2 = detection["bbox"]
        box_w, box_h = max(0, x2 - x1), max(0, y2 - y1)
        if detection.get("class_name") in self.linear_classes:
            size = max(box_w / img_w, box_h / img_h)  # crack length along its main direction
            ref = self.size_ref["linear"]
        else:
            size = float(np.sqrt(box_w * box_h / (img_w * img_h)))  # linear size of an area defect
            ref = self.size_ref["area"]
        density = dark_pixel_fraction(crop, self.dark_delta)
        return {"size": min(size / ref, 1.0), "density": min(density / self.density_ref, 1.0)}

    def predict(self, crop: np.ndarray, detection: dict | None = None,
                image_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """Estimate severity; needs the detection (class_name, bbox) and the full image shape.

        For a whole-image prediction (``detection["whole_image"]`` is True, e.g. from the
        classifier) the defect's size is unknown, so only the density cue is used.
        """
        if crop is None or crop.size == 0 or detection is None:
            return not_available()
        if detection.get("whole_image"):
            score = min(dark_pixel_fraction(crop, self.dark_delta) / self.density_ref, 1.0)
        elif image_shape is None:
            return not_available()
        else:
            f = self.features(crop, detection, image_shape)
            score = self.size_weight * f["size"] + (1.0 - self.size_weight) * f["density"]
        level = "Low" if score < self.low_max else "Medium" if score < self.medium_max else "High"
        return {"severity": level, "confidence": None, "method": self.method, "score": round(score, 3)}


def create_severity_estimator(cfg: dict[str, Any], device: str = "cpu"):
    """Return the estimator selected by ``severity.method``: auto | model | heuristic | none."""
    from src.common import resolve_path

    sev = cfg["severity"]
    method = str(sev.get("method", "auto")).lower()
    model_path = resolve_path(sev.get("model_path"))
    if method in ("auto", METHOD_MODEL):
        if model_path and model_path.is_file():
            return SeverityPredictor(model_path, device=device)
        if method == METHOD_MODEL:
            logger.warning("severity.method is 'model' but %s does not exist; severity disabled.", model_path)
            return SeverityPredictor(None)
    if method in ("auto", METHOD_HEURISTIC):
        return HeuristicSeverityEstimator(sev["heuristic"])
    return SeverityPredictor(None)
