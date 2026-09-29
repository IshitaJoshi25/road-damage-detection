"""Save each road defect as a crop (+ metadata.csv) for severity labeling.

Two modes:

* detector (default) - run the trained detector on images and crop its detections.
* --ground-truth     - crop the annotated boxes of a prepared YOLO split. Needs no
                       trained detector, so labeling can start while training runs.

Usage:
    python src/crop_defects.py --source data/yolo/images/train --ground-truth --max-per-class 150 --save-crops
    python src/crop_defects.py --source data/yolo/images/test                   # detect only, print summary
    python src/crop_defects.py --source data/raw/RDD2022_India/India/test/images --save-crops
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd
from tqdm import tqdm

from src.common import (
    get_class_names,
    get_image_extensions,
    load_config,
    resolve_device,
    resolve_path,
    resolve_weights,
    setup_logging,
    to_project_relative,
)
from src.inference import crop_detection, load_detector, predict_image

logger = logging.getLogger("crop_defects")

METADATA_NAME = "metadata.csv"
METADATA_COLUMNS = ["image_name", "image_path", "image_width", "image_height", "crop_path", "class_id",
                    "class_name", "confidence", "x1", "y1", "x2", "y2", "source"]
SOURCE_DETECTOR = "detector"
SOURCE_GROUND_TRUTH = "ground_truth"


def list_images(source: Path, extensions: set[str]) -> list[Path]:
    """Return a single image or all images (recursively) under a directory."""
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    raise FileNotFoundError(f"Source not found: {source}")


def save_crops(image: Any, image_path: Path, detections: list[dict[str, Any]], crops_dir: Path,
               padding: float, min_size: int, source: str) -> list[dict[str, Any]]:
    """Save crops as <imageName>_<detection|gt>_<NNN>.jpg and return metadata rows.

    Each detection may carry an 'index' (used in the filename); otherwise its position is used.
    """
    tag = "gt" if source == SOURCE_GROUND_TRUTH else "detection"
    img_h, img_w = image.shape[:2]
    rows = []
    for pos, det in enumerate(detections, start=1):
        idx = det.get("index", pos)
        crop = crop_detection(image, det["bbox"], padding)
        if crop.shape[0] < min_size or crop.shape[1] < min_size:
            logger.debug("%s box %d: crop %dx%d below %dpx; not saved", image_path.name, idx,
                         crop.shape[1], crop.shape[0], min_size)
            continue
        crop_path = crops_dir / f"{image_path.stem}_{tag}_{idx:03d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        x1, y1, x2, y2 = det["bbox"]
        rows.append({"image_name": image_path.name, "image_path": to_project_relative(image_path),
                     "image_width": img_w, "image_height": img_h, "crop_path": to_project_relative(crop_path),
                     "class_id": det["class_id"], "class_name": det["class_name"],
                     "confidence": det.get("confidence"), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "source": source})
    return rows


def write_metadata(metadata_path: Path, new_rows: list[dict[str, Any]], replaced_images: set[str]) -> int:
    """Merge new rows into metadata.csv.

    Old rows are replaced when they have the same crop_path, or (detector mode)
    belong to an image that was re-processed.
    """
    new_df = pd.DataFrame(new_rows, columns=METADATA_COLUMNS)
    if metadata_path.is_file():
        old = pd.read_csv(metadata_path)
        stale = old["crop_path"].isin(new_df["crop_path"])
        if "source" in old:
            stale |= (old["source"] == SOURCE_DETECTOR) & old["image_name"].isin(replaced_images)
        new_df = pd.concat([old[~stale], new_df], ignore_index=True)
    new_df.to_csv(metadata_path, index=False)
    return len(new_df)


def labels_dir_for(images_dir: Path) -> Path:
    """data/yolo/images/<split> -> data/yolo/labels/<split>."""
    return images_dir.parent.parent / "labels" / images_dir.name


def read_ground_truth(images: list[Path], labels_dir: Path) -> list[dict[str, Any]]:
    """Read YOLO label files; returns one record per annotated box (normalised coordinates)."""
    records = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, (cx, cy, w, h) = int(parts[0]), map(float, parts[1:])
            records.append({"image_path": image_path, "index": idx, "class_id": class_id, "box": (cx, cy, w, h)})
    return records


def sample_per_class(records: list[dict[str, Any]], max_per_class: int | None, seed: int) -> list[dict[str, Any]]:
    """Randomly keep at most ``max_per_class`` boxes of each class (reproducible)."""
    if not max_per_class:
        return records
    by_class: dict[int, list] = defaultdict(list)
    for rec in records:
        by_class[rec["class_id"]].append(rec)
    rng = random.Random(seed)
    kept = []
    for class_id in sorted(by_class):
        items = by_class[class_id]
        kept.extend(rng.sample(items, min(max_per_class, len(items))))
    return kept


def to_pixel_bbox(box: tuple[float, float, float, float], width: int, height: int) -> list[int]:
    cx, cy, w, h = box
    return [int(round((cx - w / 2) * width)), int(round((cy - h / 2) * height)),
            int(round((cx + w / 2) * width)), int(round((cy + h / 2) * height))]


def run_ground_truth(args: argparse.Namespace, cfg: dict[str, Any], crops_dir: Path) -> tuple[list, set, int]:
    """Crop annotated boxes from a YOLO split."""
    source = resolve_path(args.source)
    labels_dir = labels_dir_for(source)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"--ground-truth needs YOLO labels at {labels_dir} "
                                "(use --source data/yolo/images/<split>).")
    names = get_class_names(cfg)
    records = read_ground_truth(list_images(source, get_image_extensions(cfg)), labels_dir)
    records = sample_per_class(records, args.max_per_class, int(cfg["split"]["seed"]))
    counts = pd.Series([names[r["class_id"]] for r in records]).value_counts().to_dict() if records else {}
    logger.info("Ground-truth boxes selected: %d %s", len(records), counts)

    by_image: dict[Path, list] = defaultdict(list)
    for rec in records:
        by_image[rec["image_path"]].append(rec)
    rows: list[dict[str, Any]] = []
    for image_path, recs in tqdm(sorted(by_image.items()), desc="Cropping", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Could not read %s; skipped", image_path)
            continue
        h, w = image.shape[:2]
        dets = [{"index": r["index"], "class_id": r["class_id"], "class_name": names[r["class_id"]],
                 "bbox": to_pixel_bbox(r["box"], w, h)} for r in sorted(recs, key=lambda r: r["index"])]
        if args.save_crops:
            rows.extend(save_crops(image, image_path, dets, crops_dir, float(cfg["severity"]["crop_padding"]),
                                   int(cfg["severity"]["min_crop_size_px"]), SOURCE_GROUND_TRUTH))
    return rows, set(), len(records)


def run_detector(args: argparse.Namespace, cfg: dict[str, Any], crops_dir: Path) -> tuple[list, set, int]:
    """Run the detector and crop its detections."""
    weights = resolve_weights(cfg, args.weights)
    images = list_images(resolve_path(args.source), get_image_extensions(cfg))
    if not images:
        raise FileNotFoundError(f"No images found in {args.source}")
    model = load_detector(weights)
    device = resolve_device(args.device or cfg["training"].get("device", "auto"))
    conf = args.conf if args.conf is not None else cfg["model"]["confidence"]
    iou = args.iou if args.iou is not None else cfg["model"]["iou"]

    rows: list[dict[str, Any]] = []
    processed: set[str] = set()
    total = 0
    for path in tqdm(images, desc="Detecting", unit="img"):
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Could not read %s; skipped", path)
            continue
        detections = predict_image(model, image, conf=conf, iou=iou, device=device)
        total += len(detections)
        processed.add(path.name)
        if args.save_crops:
            rows.extend(save_crops(image, path, detections, crops_dir, float(cfg["severity"]["crop_padding"]),
                                   int(cfg["severity"]["min_crop_size_px"]), SOURCE_DETECTOR))
    logger.info("Processed %d images, %d detections.", len(processed), total)
    return rows, processed, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--source", required=True, help="Image file or directory of images")
    parser.add_argument("--ground-truth", action="store_true",
                        help="Crop annotated boxes of a YOLO split (--source data/yolo/images/<split>)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Ground-truth mode: randomly keep at most N boxes per class")
    parser.add_argument("--weights", default=None, help="Detector weights (default: inference.weights)")
    parser.add_argument("--save-crops", action="store_true", help="Save crops + metadata.csv to severity.crops_dir")
    parser.add_argument("--output-dir", default=None, help="Override severity.crops_dir")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    crops_dir = resolve_path(args.output_dir or cfg["severity"]["crops_dir"])
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
    try:
        runner = run_ground_truth if args.ground_truth else run_detector
        rows, processed, _ = runner(args, cfg, crops_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.save_crops:
        n = write_metadata(crops_dir / METADATA_NAME, rows, processed)
        logger.info("Saved %d crops to %s (metadata.csv now has %d rows).", len(rows), crops_dir, n)
        logger.info("Next: label them with `streamlit run label_severity.py`.")
    else:
        logger.info("Crops not saved (use --save-crops to save them).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
