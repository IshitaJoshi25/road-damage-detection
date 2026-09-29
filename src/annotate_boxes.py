"""Create Pascal VOC XML files for extra images sorted into class folders.

Two modes:

* --whole-image : one box covering the entire image (only valid when the defect
                  fills the photo, e.g. top-down close-ups of alligator cracking).
* interactive   : an OpenCV window where you draw a box around each defect.

XML files are written next to each image (same name, .xml), in the same format
as RDD2022, so convert_voc_to_yolo.py reads them unchanged. Images that are too
small, or exact copies of an image already handled, are skipped automatically.
Images you skip (diagrams, collages, no visible defect) get no XML and are
therefore not used. Progress is kept in <folder>/annotation_log.csv, so you can
stop at any time and continue later.

Usage:
    python src/annotate_boxes.py --folder data/raw/extra/alligator_cracks --class alligator_crack --whole-image
    python src/annotate_boxes.py --folder data/raw/extra/potholes --class pothole

Keys in the interactive window:
    mouse drag   draw a box (class = current class, shown at the top)
    1 2 3 4      current class: 1 longitudinal, 2 transverse, 3 alligator, 4 pothole
    u            undo the last box
    Enter/Space  save boxes and go to the next image
    s            skip image (not usable: diagram, collage, no defect, wrong class...)
    b            back to the previous image
    q / Esc      save progress and quit
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.common import get_class_names, get_image_extensions, get_source_code_map, load_config, resolve_path, setup_logging
from src.inference import CLASS_COLORS_BGR

logger = logging.getLogger("annotate_boxes")

LOG_NAME = "annotation_log.csv"
LOG_COLUMNS = ["file", "status", "boxes", "note", "md5", "updated"]
MIN_IMAGE_SIDE_PX = 100      # smaller images are skipped automatically
MIN_BOX_SIDE_PX = 4          # ignore accidental clicks
WINDOW = "annotate - drag boxes | 1-4 class | u undo | Enter next | s skip | b back | q quit"
MAX_DISPLAY_SIDE = 1000      # large photos are shown scaled down
KEY_ENTER = (13, 10, 32)
KEY_ESC = 27


def load_image_upright(path: Path) -> np.ndarray | None:
    """Load as BGR with EXIF rotation applied (the orientation YOLO trains on)."""
    try:
        with Image.open(path) as img:
            rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except OSError:
        return None


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def xml_path_for(image_path: Path) -> Path:
    """<stem>.xml, or <file name>.xml if another image in the folder has the same stem."""
    clash = any(p != image_path and p.stem == image_path.stem and p.suffix.lower() != ".xml"
                for p in image_path.parent.glob(f"{glob_escape(image_path.stem)}.*"))
    return image_path.with_name(image_path.name + ".xml") if clash else image_path.with_suffix(".xml")


def glob_escape(text: str) -> str:
    return "".join(f"[{c}]" if c in "[]*?" else c for c in text)


def remove_xml(image_path: Path) -> None:
    for candidate in (image_path.with_suffix(".xml"), image_path.with_name(image_path.name + ".xml")):
        if candidate.is_file() and candidate == xml_path_for(image_path):
            candidate.unlink()


def write_voc_xml(image_path: Path, width: int, height: int, boxes: list[tuple[str, list[int]]]) -> Path:
    """Write <image>.xml next to the image; boxes are (code, [x1, y1, x2, y2])."""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = image_path.parent.name
    ET.SubElement(root, "filename").text = image_path.name
    size = ET.SubElement(root, "size")
    for tag, value in (("width", width), ("height", height), ("depth", 3)):
        ET.SubElement(size, tag).text = str(value)
    for code, (x1, y1, x2, y2) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = code
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        for tag, value in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
            ET.SubElement(bnd, tag).text = str(int(value))
    ET.indent(root)
    xml_path = xml_path_for(image_path)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


def read_voc_boxes(xml_path: Path) -> list[tuple[str, list[int]]]:
    """Boxes of an existing XML (to show them again when going back)."""
    if not xml_path.is_file():
        return []
    boxes = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        b = obj.find("bndbox")
        boxes.append((obj.findtext("name", ""), [int(float(b.findtext(t))) for t in ("xmin", "ymin", "xmax", "ymax")]))
    return boxes


class AnnotationLog:
    """annotation_log.csv: one row per image with its status."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / LOG_NAME
        self.df = pd.read_csv(self.path, dtype=str).fillna("") if self.path.is_file() else pd.DataFrame(columns=LOG_COLUMNS)

    def status(self, name: str) -> str | None:
        rows = self.df[self.df["file"] == name]
        return rows["status"].iloc[0] if len(rows) else None

    def set(self, name: str, status: str, boxes: int = 0, note: str = "", md5: str = "") -> None:
        self.df = self.df[self.df["file"] != name]
        row = {"file": name, "status": status, "boxes": boxes, "note": note, "md5": md5,
               "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)


def known_hashes(extra_root: Path, current_folder: Path) -> dict[str, str]:
    """md5 -> 'folder/file' of images already labeled or skipped in any class folder."""
    hashes: dict[str, str] = {}
    for log_path in extra_root.rglob(LOG_NAME):
        df = pd.read_csv(log_path, dtype=str).fillna("")
        for row in df.itertuples(index=False):
            if row.md5 and row.status in ("labeled", "whole_image"):
                hashes.setdefault(row.md5, f"{log_path.parent.name}/{row.file}")
    return hashes


def other_folder_hashes(folder: Path, extensions: set[str]) -> dict[str, str]:
    """md5 -> 'folder/file' for images in the sibling class folders."""
    hashes: dict[str, str] = {}
    for sibling in folder.parent.iterdir():
        if sibling.is_dir() and sibling != folder:
            for f in sibling.iterdir():
                if f.is_file() and f.suffix.lower() in extensions:
                    hashes.setdefault(file_md5(f), f"{sibling.name}/{f.name}")
    return hashes


def auto_skip_reason(path: Path, image: np.ndarray | None, md5: str, seen: dict[str, str]) -> str | None:
    if image is None:
        return "unreadable"
    if min(image.shape[:2]) < MIN_IMAGE_SIDE_PX:
        return f"too small ({image.shape[1]}x{image.shape[0]})"
    if md5 in seen and seen[md5] != f"{path.parent.name}/{path.name}":
        return f"duplicate of {seen[md5]}"
    return None


def run_whole_image(images: list[Path], code: str, log: AnnotationLog, seen: dict[str, str],
                    in_other_folders: dict[str, str]) -> None:
    """One full-image box per image. Images that also appear in another class folder are
    ambiguous (the folder cannot tell which class is right) and are skipped."""
    done = skipped = 0
    for path in images:
        image = load_image_upright(path)
        md5 = file_md5(path)
        reason = auto_skip_reason(path, image, md5, seen)
        if not reason and md5 in in_other_folders:
            reason = f"ambiguous: same image is also in {in_other_folders[md5]}"
        if not reason and md5 in seen and seen[md5] != f"{path.parent.name}/{path.name}":
            reason = f"duplicate of {seen[md5]}"
        if reason:
            remove_xml(path)
            log.set(path.name, "auto_skipped", note=reason, md5=md5)
            skipped += 1
            continue
        h, w = image.shape[:2]
        write_voc_xml(path, w, h, [(code, [0, 0, w, h])])
        log.set(path.name, "whole_image", boxes=1, md5=md5)
        seen[md5] = f"{path.parent.name}/{path.name}"
        done += 1
    logger.info("Whole-image XML written for %d images, %d skipped automatically.", done, skipped)


class BoxDrawer:
    """Interactive OpenCV box drawing for one image."""

    def __init__(self, image: np.ndarray, boxes: list[tuple[str, list[int]]], code: str,
                 code_by_key: dict[int, str], names_by_code: dict[str, str], title: str) -> None:
        self.image = image
        self.scale = min(1.0, MAX_DISPLAY_SIDE / max(image.shape[:2]))
        self.boxes = list(boxes)
        self.code = code
        self.code_by_key = code_by_key
        self.names_by_code = names_by_code
        self.title = title
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start, self.current = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            (sx, sy), inv = self.start, 1.0 / self.scale
            h, w = self.image.shape[:2]
            x1, x2 = sorted((int(sx * inv), int(x * inv)))
            y1, y2 = sorted((int(sy * inv), int(y * inv)))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 >= MIN_BOX_SIDE_PX and y2 - y1 >= MIN_BOX_SIDE_PX:
                self.boxes.append((self.code, [x1, y1, x2, y2]))
            self.start = self.current = None

    def render(self) -> np.ndarray:
        canvas = cv2.resize(self.image, None, fx=self.scale, fy=self.scale) if self.scale < 1 else self.image.copy()
        class_ids = {c: i for i, c in enumerate(self.names_by_code)}
        for code, (x1, y1, x2, y2) in self.boxes:
            color = CLASS_COLORS_BGR.get(class_ids.get(code, -1), (200, 200, 200))
            p1, p2 = (int(x1 * self.scale), int(y1 * self.scale)), (int(x2 * self.scale), int(y2 * self.scale))
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, self.names_by_code.get(code, code), (p1[0] + 3, p1[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        if self.start and self.current:
            cv2.rectangle(canvas, self.start, self.current, (0, 255, 255), 1)
        keys = "  ".join(f"{chr(k)}={self.names_by_code[c]}" for k, c in self.code_by_key.items())
        header = f"{self.title} | drawing: {self.names_by_code[self.code]} | {keys}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), cv2.FILLED)
        cv2.putText(canvas, header, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def run(self) -> str:
        """Returns 'next', 'skip', 'back' or 'quit'."""
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return "quit"
                continue
            if key in KEY_ENTER:
                return "next"
            if key in (ord("q"), KEY_ESC):
                return "quit"
            if key == ord("s"):
                return "skip"
            if key == ord("b"):
                return "back"
            if key == ord("u") and self.boxes:
                self.boxes.pop()
            if key in self.code_by_key:
                self.code = self.code_by_key[key]


def run_interactive(images: list[Path], code: str, log: AnnotationLog, seen: dict[str, str],
                    code_by_key: dict[int, str], names_by_code: dict[str, str], redo: bool) -> None:
    todo = [p for p in images if redo or log.status(p.name) is None]
    logger.info("%d of %d images still to annotate in this folder.", len(todo), len(images))
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    i = 0
    while 0 <= i < len(todo):
        path = todo[i]
        image = load_image_upright(path)
        md5 = file_md5(path)
        reason = auto_skip_reason(path, image, md5, seen)
        if reason:
            log.set(path.name, "auto_skipped", note=reason, md5=md5)
            logger.info("Auto-skipped %s: %s", path.name, reason)
            i += 1
            continue
        drawer = BoxDrawer(image, read_voc_boxes(xml_path_for(path)), code, code_by_key, names_by_code,
                           f"[{i + 1}/{len(todo)}] {path.name}")
        action = drawer.run()
        if action == "quit":
            break
        if action == "back":
            i = max(0, i - 1)
            continue
        if action == "skip" or not drawer.boxes:
            remove_xml(path)
            log.set(path.name, "skipped", note="skipped by annotator" if action == "skip" else "no boxes", md5=md5)
        else:
            h, w = image.shape[:2]
            write_voc_xml(path, w, h, drawer.boxes)
            log.set(path.name, "labeled", boxes=len(drawer.boxes), md5=md5)
            seen[md5] = f"{path.parent.name}/{path.name}"
        i += 1
    cv2.destroyAllWindows()


def summarize(folder: Path, log: AnnotationLog) -> None:
    counts = log.df["status"].value_counts().to_dict()
    boxes = pd.to_numeric(log.df["boxes"], errors="coerce").fillna(0).sum()
    logger.info("%s: %s | %d boxes in total. Log: %s", folder.name, counts, int(boxes), log.path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--folder", required=True, help="Folder with images of one class")
    parser.add_argument("--class", dest="class_name", required=True,
                        help="Default class: longitudinal_crack | transverse_crack | alligator_crack | pothole")
    parser.add_argument("--whole-image", action="store_true", help="One box = whole image (close-ups only)")
    parser.add_argument("--redo", action="store_true", help="Revisit images that were already handled")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    names = get_class_names(cfg)
    code_of = {names[cid]: code for code, cid in get_source_code_map(cfg).items()}
    if args.class_name not in code_of:
        logger.error("Unknown class '%s'. Use one of: %s", args.class_name, ", ".join(code_of))
        return 1
    names_by_code = {code_of[names[cid]]: names[cid] for cid in sorted(names)}
    code_by_key = {ord(str(i + 1)): code for i, code in enumerate(names_by_code)}

    folder = resolve_path(args.folder)
    if not folder or not folder.is_dir():
        logger.error("Folder not found: %s", args.folder)
        return 1
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in get_image_extensions(cfg))
    if not images:
        logger.error("No images in %s", folder)
        return 1
    log = AnnotationLog(folder)
    seen = known_hashes(folder.parent, folder)
    if args.whole_image:
        run_whole_image(images, code_of[args.class_name], log, seen,
                        other_folder_hashes(folder, get_image_extensions(cfg)))
    else:
        run_interactive(images, code_of[args.class_name], log, seen, code_by_key, names_by_code, args.redo)
    summarize(folder, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
