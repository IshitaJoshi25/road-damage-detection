"""Visually check the RDD2022 annotations (and our conversion of them).

For a random sample of boxes per class this writes to reports/annotation_samples/:

    <class>/sheet.jpg        contact sheet: one tile per box (crop with margin + id)
    <class>/<image>.jpg      full image with ALL of its original XML boxes drawn
    review.csv               one row per sampled box - fill in the 'verdict' column
    box_size_stats.csv       box size statistics per class (all boxes, not just the sample)

It also checks two things that would make labels look wrong even though the
XML is correct:
  * EXIF rotation - photos with an EXIF orientation tag are shown rotated by
    OpenCV/YOLO, while annotation tools often ignore the tag, so boxes shift.
  * Conversion   - if data/interim/labels exists, our YOLO boxes are drawn as
    thin white rectangles; they must sit exactly on the coloured XML boxes.

Usage:
    python src/visualize_annotations.py                       # 40 boxes per class
    python src/visualize_annotations.py --classes D40 --n 80  # 80 potholes
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
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.common import (
    get_class_names,
    get_image_extensions,
    get_source_code_map,
    load_config,
    resolve_path,
    setup_logging,
    to_project_relative,
)
from src.inference import CLASS_COLORS_BGR, draw_predictions
from src.voc_dataset import VocParseError, discover_dataset, parse_voc_xml

logger = logging.getLogger("visualize_annotations")

EXIF_ORIENTATION_TAG = 274
TILE_SIZE = 220
TILE_MARGIN = 0.35            # context around each box in a tile (fraction of box size)
SHEET_COLUMNS = 8
TILE_LABEL_HEIGHT = 22
TINY_BOX_PX = 32              # boxes smaller than this (shorter side) are "tiny"
CONVERTED_COLOR = (255, 255, 255)


def exif_orientation(image_path: Path) -> int:
    """EXIF orientation tag (1 = normal); 1 if absent or unreadable."""
    try:
        with Image.open(image_path) as img:
            return int(img.getexif().get(EXIF_ORIENTATION_TAG, 1))
    except (OSError, ValueError):
        return 1


def collect_boxes(pairs, code_map: dict[str, int], names: dict[int, str]) -> tuple[list[dict], list[str]]:
    """All target-class boxes from the XML files: one record per box."""
    boxes, problems = [], []
    for pair in tqdm(pairs, desc="Reading XML", unit="file"):
        try:
            ann, _ = parse_voc_xml(pair.xml_path)
        except VocParseError as exc:
            problems.append(f"{pair.xml_path.name}: {exc}")
            continue
        for idx, obj in enumerate(ann.objects, start=1):
            code = obj.name.strip().upper()
            if code not in code_map:
                continue
            x1, x2 = sorted((obj.xmin, obj.xmax))  # some XML boxes have swapped min/max
            y1, y2 = sorted((obj.ymin, obj.ymax))
            boxes.append({"uid": pair.uid, "image_path": pair.image_path, "box_index": idx, "code": code,
                          "class_id": code_map[code], "class_name": names[code_map[code]],
                          "bbox": [int(round(v)) for v in (x1, y1, x2, y2)],
                          "xml_width": ann.width, "xml_height": ann.height})
    return boxes, problems


def size_stats(boxes: list[dict]) -> pd.DataFrame:
    """Per-class box size statistics (pixels)."""
    rows = []
    frame = pd.DataFrame([{"class_name": b["class_name"], "w": b["bbox"][2] - b["bbox"][0],
                           "h": b["bbox"][3] - b["bbox"][1]} for b in boxes])
    if frame.empty:
        return frame
    for name, g in frame.groupby("class_name"):
        short = g[["w", "h"]].min(axis=1)
        rows.append({"class_name": name, "boxes": len(g), "median_width_px": int(g.w.median()),
                     "median_height_px": int(g.h.median()),
                     f"tiny_boxes_pct(<{TINY_BOX_PX}px)": round(100 * (short < TINY_BOX_PX).mean(), 1),
                     "largest_box_px": f"{int(g.w.max())}x{int(g.h.max())}"})
    return pd.DataFrame(rows)


def read_converted(labels_dir: Path | None, uid: str, width: int, height: int) -> list[list[int]]:
    """Our converted YOLO boxes for an image, in pixels ([] if not available)."""
    if labels_dir is None:
        return []
    path = labels_dir / f"{uid}.txt"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            cx, cy, w, h = (float(v) for v in parts[1:])
            out.append([int(round((cx - w / 2) * width)), int(round((cy - h / 2) * height)),
                        int(round((cx + w / 2) * width)), int(round((cy + h / 2) * height))])
    return out


def draw_full_image(image: np.ndarray, image_boxes: list[dict], converted: list[list[int]]) -> np.ndarray:
    """All XML boxes (coloured, labelled with their id) plus converted boxes (thin white)."""
    dets = [{"class_id": b["class_id"], "class_name": f"#{b['box_index']} {b['class_name']}", "confidence": 1.0,
             "bbox": b["bbox"]} for b in image_boxes]
    canvas = draw_predictions(image, dets, box_thickness=3, show_confidence=False)
    for x1, y1, x2, y2 in converted:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), CONVERTED_COLOR, 1, cv2.LINE_AA)
    return canvas


def make_tile(image: np.ndarray, box: dict, tile_id: str) -> np.ndarray:
    """Square tile: the box with some context, box outlined, id written underneath."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box["bbox"]
    mx, my = int((x2 - x1) * TILE_MARGIN) + 8, int((y2 - y1) * TILE_MARGIN) + 8
    left, top = min(max(0, x1 - mx), w - 1), min(max(0, y1 - my), h - 1)
    right, bottom = max(min(w, x2 + mx), left + 1), max(min(h, y2 + my), top + 1)
    crop = np.ascontiguousarray(image[top:bottom, left:right])
    color = CLASS_COLORS_BGR.get(box["class_id"], (128, 128, 128))
    cv2.rectangle(crop, (x1 - left, y1 - top), (x2 - left, y2 - top), color, 2)
    scale = TILE_SIZE / max(crop.shape[:2]) if crop.size else 1.0
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    tile = np.full((TILE_SIZE + TILE_LABEL_HEIGHT, TILE_SIZE, 3), 30, np.uint8)
    oy, ox = (TILE_SIZE - crop.shape[0]) // 2, (TILE_SIZE - crop.shape[1]) // 2
    tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    cv2.putText(tile, f"{tile_id}  {x2 - x1}x{y2 - y1}px", (4, TILE_SIZE + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def make_sheet(tiles: list[np.ndarray]) -> np.ndarray:
    rows = [tiles[i:i + SHEET_COLUMNS] for i in range(0, len(tiles), SHEET_COLUMNS)]
    blank = np.full_like(tiles[0], 30)
    return np.vstack([np.hstack(r + [blank] * (SHEET_COLUMNS - len(r))) for r in rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--raw-dir", default=None, help="Override dataset.raw_dir")
    parser.add_argument("--classes", nargs="*", default=None, help="Codes or names, e.g. D40 pothole (default: all)")
    parser.add_argument("--n", type=int, default=40, help="Boxes sampled per class")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default: split.seed)")
    parser.add_argument("--out", default=None, help="Output folder (default: reports/annotation_samples)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    names, code_map = get_class_names(cfg), get_source_code_map(cfg)
    raw_dir = resolve_path(args.raw_dir or cfg["dataset"]["raw_dir"])
    out_dir = resolve_path(args.out) if args.out else resolve_path(cfg["paths"]["reports_dir"]) / "annotation_samples"
    interim_labels = resolve_path(cfg["dataset"]["interim_dir"]) / "labels"
    labels_dir = interim_labels if interim_labels.is_dir() else None
    wanted = {c.upper() for c in args.classes} if args.classes else None

    try:
        result = discover_dataset(raw_dir, get_image_extensions(cfg), cfg["dataset"]["annotation_extension"])
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    if not result.pairs:
        logger.error("No image/XML pairs under %s. Run src/inspect_dataset.py to see what was found.", raw_dir)
        return 1

    rotated = [p for p in tqdm(result.pairs, desc="Checking EXIF", unit="img") if exif_orientation(p.image_path) != 1]
    boxes, problems = collect_boxes(result.pairs, code_map, names)
    stats = size_stats(boxes)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "box_size_stats.csv", index=False)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for b in boxes:
        if wanted is None or b["code"] in wanted or b["class_name"].upper() in wanted:
            by_class[b["class_name"]].append(b)
    rng = random.Random(args.seed if args.seed is not None else int(cfg["split"]["seed"]))
    boxes_by_uid: dict[str, list[dict]] = defaultdict(list)
    for b in boxes:
        boxes_by_uid[b["uid"]].append(b)

    review_rows, mismatches = [], 0
    for class_name in sorted(by_class):
        sample = rng.sample(by_class[class_name], min(args.n, len(by_class[class_name])))
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for i, box in enumerate(tqdm(sample, desc=f"Drawing {class_name}", unit="box"), start=1):
            image = cv2.imread(str(box["image_path"]))
            if image is None:
                continue
            h, w = image.shape[:2]
            tile_id = f"{class_name[:3].upper()}-{i:03d}"
            tiles.append(make_tile(image, box, tile_id))
            converted = read_converted(labels_dir, box["uid"], w, h)
            clipped = np.clip(box["bbox"], 0, [w, h, w, h])  # the converter clips boxes to the image
            if labels_dir and converted and not any(np.abs(np.subtract(c, clipped)).max() <= 2 for c in converted):
                mismatches += 1
            full_path = class_dir / f"{box['uid']}.jpg"
            cv2.imwrite(str(full_path), draw_full_image(image, boxes_by_uid[box["uid"]], converted))
            x1, y1, x2, y2 = box["bbox"]
            review_rows.append({"tile_id": tile_id, "class_name": class_name, "code": box["code"],
                                "image_name": box["image_path"].name, "box_index": box["box_index"],
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "width_px": x2 - x1, "height_px": y2 - y1,
                                "full_image": to_project_relative(full_path), "verdict": "", "comment": ""})
        if tiles:
            cv2.imwrite(str(class_dir / "sheet.jpg"), make_sheet(tiles))
    pd.DataFrame(review_rows).to_csv(out_dir / "review.csv", index=False)

    logger.info("=" * 70)
    logger.info("Image/XML pairs: %d | target-class boxes: %d", len(result.pairs), len(boxes))
    logger.info("Box sizes (all boxes):\n%s", stats.to_string(index=False) if not stats.empty else "(none)")
    if rotated:
        logger.warning("EXIF rotation: %d images have an orientation tag != 1 (e.g. %s). Their boxes may be "
                       "shifted/rotated - check their full images carefully.", len(rotated), rotated[0].image_path.name)
    else:
        logger.info("EXIF rotation: none found - boxes are not affected by photo rotation.")
    if labels_dir is None:
        logger.info("Conversion check skipped (run src/convert_voc_to_yolo.py first to include it).")
    elif mismatches:
        logger.warning("Conversion check: %d sampled boxes have NO matching converted YOLO box.", mismatches)
    else:
        logger.info("Conversion check: every sampled XML box has a matching converted YOLO box.")
    if problems:
        logger.warning("%d XML files could not be read (e.g. %s)", len(problems), problems[0])
    logger.info("Open %s: start with <class>/sheet.jpg, then fill the 'verdict' column in review.csv "
                "(ok / wrong_class / bad_box / not_damage).", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
