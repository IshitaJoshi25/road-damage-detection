"""Evaluate the trained detector on the held-out test split.

Reports precision, recall, mAP50 and mAP50-95 (overall and per class), and
saves Ultralytics plots (confusion matrix, PR curves) plus example predictions
under ``reports/evaluation/``.

Usage:
    python src/evaluate_detector.py [--weights models/road_damage_yolo.pt] [--split test]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import get_image_extensions, load_config, resolve_device, resolve_path, resolve_weights, setup_logging

logger = logging.getLogger("evaluate_detector")

EVAL_SUBDIR = "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--weights", default=None, help="Detector weights (default: inference.weights / latest run)")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    return parser.parse_args()


def metrics_tables(metrics: Any, names: dict[int, str]) -> tuple[dict[str, float], pd.DataFrame]:
    """Extract overall and per-class metrics from an Ultralytics DetMetrics object."""
    box = metrics.box
    overall = {"precision": float(box.mp), "recall": float(box.mr),
               "mAP50": float(box.map50), "mAP50-95": float(box.map)}
    rows = []
    for i, class_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows.append({"class_id": int(class_id), "class_name": names[int(class_id)], "precision": float(p),
                     "recall": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)})
    return overall, pd.DataFrame(rows)


def save_example_predictions(model: Any, image_dir: Path, out_dir: Path, count: int, cfg: dict[str, Any],
                             device: str, seed: int) -> int:
    """Run the detector on a few images and save annotated copies."""
    import cv2

    from src.inference import draw_predictions, predict_image

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in get_image_extensions(cfg))
    if not images:
        return 0
    random.Random(seed).shuffle(images)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in images[:count]:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Could not read %s", path)
            continue
        detections = predict_image(model, image, conf=cfg["model"]["confidence"], iou=cfg["model"]["iou"],
                                   device=device)
        annotated = draw_predictions(image, detections, cfg["inference"]["box_thickness"],
                                     cfg["inference"]["font_scale"])
        cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), annotated)
    return min(count, len(images))


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    split = args.split or ev.get("split", "test")
    data_yaml = resolve_path(cfg["dataset"]["yolo_dir"]) / "data.yaml"
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    device = resolve_device(args.device or cfg["training"].get("device", "auto"))

    try:
        weights = resolve_weights(cfg, args.weights)
        if not data_yaml.is_file():
            raise FileNotFoundError(f"{data_yaml} not found. Run src/split_dataset.py first.")
        split_dir = data_yaml.parent / "images" / split
        if not split_dir.is_dir() or not any(split_dir.iterdir()):
            raise FileNotFoundError(f"No images in {split_dir}; cannot evaluate split '{split}'.")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    from src.inference import load_detector

    model = load_detector(weights)
    names = {int(k): v for k, v in model.names.items()}
    logger.info("Evaluating %s on split '%s' (device=%s)", weights, split, device)
    metrics = model.val(
        data=data_yaml.as_posix(), split=split, device=device,
        imgsz=args.imgsz or cfg["training"]["image_size"], batch=args.batch or cfg["training"]["batch"],
        conf=ev["confidence"], iou=ev["iou"], plots=True,
        project=reports_dir.as_posix(), name=EVAL_SUBDIR, exist_ok=True,
    )
    eval_dir = Path(getattr(metrics, "save_dir", reports_dir / EVAL_SUBDIR))
    overall, per_class = metrics_tables(metrics, names)

    logger.info("=" * 60)
    logger.info("Overall (%s): P=%.3f  R=%.3f  mAP50=%.3f  mAP50-95=%.3f", split, overall["precision"],
                overall["recall"], overall["mAP50"], overall["mAP50-95"])
    if not per_class.empty:
        logger.info("Per class:\n%s", per_class.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    logger.info("=" * 60)

    eval_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(eval_dir / "per_class_metrics.csv", index=False)
    (eval_dir / "metrics.json").write_text(json.dumps(
        {"weights": weights.as_posix(), "split": split, "overall": overall,
         "per_class": per_class.to_dict(orient="records")}, indent=2), encoding="utf-8")

    n_examples = save_example_predictions(model, split_dir, eval_dir / "examples", int(ev["num_example_images"]),
                                          cfg, device, int(cfg["split"]["seed"]))
    plots = sorted(p.name for p in eval_dir.glob("*.png"))
    logger.info("Saved metrics.json, per_class_metrics.csv, %d example predictions and plots %s in %s",
                n_examples, plots, eval_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
