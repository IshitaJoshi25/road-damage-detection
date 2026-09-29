"""Reusable detector inference: loading, prediction, drawing and severity hookup.

Used by the Streamlit app, evaluate_detector.py and crop_defects.py so that no
inference logic lives in the UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ultralytics import YOLO

    from src.severity import HeuristicSeverityEstimator, SeverityPredictor

    SeverityEstimator = SeverityPredictor | HeuristicSeverityEstimator

logger = logging.getLogger(__name__)

DEFAULT_BOX_THICKNESS = 2
DEFAULT_FONT_SCALE = 0.6
LABEL_PADDING_PX = 4
# Fixed per-class colours (BGR), in the project palette order: blue, orange, aqua, yellow.
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (214, 120, 42),
    1: (52, 104, 235),
    2: (122, 175, 27),
    3: (0, 161, 237),
}
FALLBACK_COLOR_BGR = (129, 135, 137)
DARK_TEXT_BGR = (11, 11, 11)
LIGHT_TEXT_BGR = (255, 255, 255)
LUMINANCE_TEXT_THRESHOLD = 150


def load_detector(weights_path: str | Path) -> "YOLO":
    """Load an Ultralytics YOLO model from a weights file."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detector weights not found: {path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    return YOLO(str(path))


def to_bgr_array(image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    """Convert a path, PIL image (RGB) or BGR array into a BGR uint8 array."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        array = cv2.imread(str(image))
        if array is None:
            raise ValueError(f"Could not read image: {image}")
        return array
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def predict_image(
    model: "YOLO",
    image: np.ndarray | Image.Image | str | Path,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    imgsz: int | None = None,
) -> list[dict[str, Any]]:
    """Run the detector on one image.

    Returns a list of dicts sorted by confidence::

        {"class_id": 3, "class_name": "pothole", "confidence": 0.94, "bbox": [x1, y1, x2, y2]}

    Bounding boxes are integer pixel coordinates in the input image.
    """
    kwargs: dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    result = model.predict(to_bgr_array(image), **kwargs)[0]

    detections = []
    for xyxy, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        class_id = int(cls)
        detections.append({
            "class_id": class_id,
            "class_name": str(result.names.get(class_id, class_id)),
            "confidence": round(float(score), 4),
            "bbox": [int(round(v)) for v in xyxy],
        })
    return sorted(detections, key=lambda d: d["confidence"], reverse=True)


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return DARK_TEXT_BGR if luminance > LUMINANCE_TEXT_THRESHOLD else LIGHT_TEXT_BGR


def format_label(det: dict[str, Any], show_confidence: bool = True) -> str:
    """Label text drawn above a box, including severity only when available."""
    label = f"{det['class_name']} {det['confidence']:.2f}" if show_confidence else det["class_name"]
    if det.get("severity"):
        label += f" | {det['severity']}"
        if det.get("severity_method") == "heuristic":
            label += " (est.)"
    return label


def draw_predictions(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    box_thickness: int = DEFAULT_BOX_THICKNESS,
    font_scale: float = DEFAULT_FONT_SCALE,
    show_confidence: bool = True,
) -> np.ndarray:
    """Return a copy of a BGR image with boxes and 'class confidence' labels drawn.

    Use ``show_confidence=False`` for human annotations, which have no confidence.
    """
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, box_thickness - 1)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS_BGR.get(det["class_id"], FALLBACK_COLOR_BGR)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

        label = format_label(det, show_confidence)
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        top = y1 - th - baseline - LABEL_PADDING_PX
        if top < 0:  # no room above the box: draw the label inside it
            top = y1
        x1 = max(0, min(x1, canvas.shape[1] - tw - LABEL_PADDING_PX))  # keep label inside the image
        cv2.rectangle(canvas, (x1, top), (x1 + tw + LABEL_PADDING_PX, top + th + baseline + LABEL_PADDING_PX),
                      color, cv2.FILLED)
        cv2.putText(canvas, label, (x1 + LABEL_PADDING_PX // 2, top + th + LABEL_PADDING_PX // 2),
                    font, font_scale, _text_color(color), text_thickness, cv2.LINE_AA)
    return canvas


def crop_detection(image: np.ndarray, bbox: list[int], padding: float = 0.0) -> np.ndarray:
    """Crop a box from a BGR image, expanded by ``padding`` (fraction of box size) and clipped."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(w, int(round(x2 + pad_x))), min(h, int(round(y2 + pad_y)))
    return image[top:bottom, left:right].copy()


def attach_severity(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    severity_estimator: "SeverityEstimator | None",
    padding: float = 0.0,
) -> list[dict[str, Any]]:
    """Add 'severity', 'severity_confidence' and 'severity_method' to each detection.

    'severity_method' is "model" (trained classifier), "heuristic" (rule-based
    estimate) or None (no severity available).
    """
    for det in detections:
        result = {"severity": None, "confidence": None, "method": None}
        if severity_estimator is not None and severity_estimator.available:
            crop = crop_detection(image, det["bbox"], padding)
            result = severity_estimator.predict(crop, detection=det, image_shape=image.shape[:2])
        det["severity"] = result["severity"]
        det["severity_confidence"] = result["confidence"]
        det["severity_method"] = result["method"]
    return detections


def analyze_image(
    model: "YOLO",
    image: np.ndarray | Image.Image | str | Path,
    severity_estimator: "SeverityEstimator | None" = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    crop_padding: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Full pipeline: detect defects, then (optionally) estimate severity per crop.

    Returns the BGR image and the detections with severity fields attached.
    """
    bgr = to_bgr_array(image)
    detections = predict_image(model, bgr, conf=conf, iou=iou, device=device)
    return bgr, attach_severity(bgr, detections, severity_estimator, crop_padding)
