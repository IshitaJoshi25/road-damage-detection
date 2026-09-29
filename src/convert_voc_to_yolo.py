"""Convert RDD2022 Pascal VOC XML annotations to YOLO label files.

For every image with a matching XML file, writes ``<interim_dir>/labels/<uid>.txt``
containing zero or more lines ``class_id cx cy w h`` (normalised to [0, 1]) and
records the pair in ``<interim_dir>/manifest.csv`` for split_dataset.py.

Folders listed in ``dataset.extra_dirs`` (your own labelled images) are converted
too and marked ``source=extra`` in the manifest; split_dataset.py puts them in
the training split only, so validation/test remain pure RDD2022 India.

Usage:
    python src/convert_voc_to_yolo.py [--config config.yaml] [--force] [--no-extra]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm

from src.common import (
    get_class_names,
    get_image_extensions,
    get_source_code_map,
    load_config,
    resolve_path,
    setup_logging,
)
from src.voc_dataset import ImageXmlPair, VocObject, VocParseError, discover_dataset, parse_voc_xml, read_image_size

logger = logging.getLogger("convert_voc_to_yolo")

MANIFEST_NAME = "manifest.csv"
SOURCE_RDD = "rdd2022"
SOURCE_EXTRA = "extra"
EXTRA_UID_PREFIX = "extra__"
LABELS_SUBDIR = "labels"
YOLO_FLOAT_FORMAT = "{:.6f}"


@dataclass
class ConversionStats:
    """Counters accumulated over the whole conversion."""

    images_written: int = 0
    images_skipped: int = 0
    boxes_written: Counter = field(default_factory=Counter)
    ignored_labels: Counter = field(default_factory=Counter)
    boxes_clipped: int = 0
    boxes_swapped: int = 0
    boxes_dropped: int = 0
    issues: list[dict] = field(default_factory=list)

    def add_issue(self, file: Path, severity: str, message: str) -> None:
        self.issues.append({"file": str(file), "severity": severity, "message": message})


def to_yolo_box(
    obj: VocObject, width: int, height: int, min_size: float, stats: ConversionStats, xml_path: Path
) -> tuple[float, float, float, float] | None:
    """Validate/clip a VOC box and return normalised (cx, cy, w, h), or None if unusable."""
    x1, y1, x2, y2 = obj.xmin, obj.ymin, obj.xmax, obj.ymax
    if x1 > x2 or y1 > y2:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        stats.boxes_swapped += 1
        stats.add_issue(xml_path, "warning", f"{obj.name}: min/max coordinates swapped; fixed")

    cx1, cy1 = min(max(x1, 0.0), width), min(max(y1, 0.0), height)
    cx2, cy2 = min(max(x2, 0.0), width), min(max(y2, 0.0), height)
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        stats.boxes_clipped += 1
        stats.add_issue(
            xml_path, "warning",
            f"{obj.name}: box ({x1:g},{y1:g},{x2:g},{y2:g}) outside {width}x{height}; clipped",
        )

    bw, bh = cx2 - cx1, cy2 - cy1
    if bw < min_size or bh < min_size:
        stats.boxes_dropped += 1
        stats.add_issue(xml_path, "error", f"{obj.name}: degenerate box {bw:g}x{bh:g}px after clipping; dropped")
        return None
    return (cx1 + bw / 2) / width, (cy1 + bh / 2) / height, bw / width, bh / height


def image_dimensions(pair: ImageXmlPair, xml_w: int | None, xml_h: int | None, stats: ConversionStats):
    """Return the real image (width, height); report mismatches with the XML <size>."""
    try:
        width, height = read_image_size(pair.image_path)
    except OSError as exc:
        stats.add_issue(pair.image_path, "error", f"unreadable image ({exc}); skipped")
        return None
    if xml_w and xml_h and (xml_w, xml_h) != (width, height):
        stats.add_issue(pair.xml_path, "warning",
                        f"XML size {xml_w}x{xml_h} differs from image {width}x{height}; using image size")
    return width, height


def convert_pair(pair: ImageXmlPair, code_map: dict[str, int], labels_dir: Path,
                 min_size: float, stats: ConversionStats) -> dict | None:
    """Convert one image/XML pair; return a manifest row or None if skipped."""
    try:
        ann, problems = parse_voc_xml(pair.xml_path)
    except VocParseError as exc:
        stats.add_issue(pair.xml_path, "error", f"{exc}; image skipped")
        return None
    for msg in problems:
        stats.add_issue(pair.xml_path, "error", f"{msg}; object skipped")

    dims = image_dimensions(pair, ann.width, ann.height, stats)
    if dims is None:
        return None
    width, height = dims

    lines: list[str] = []
    counts: Counter = Counter()
    for obj in ann.objects:
        code = obj.name.strip().upper()
        if code not in code_map:
            stats.ignored_labels[code] += 1
            continue
        box = to_yolo_box(obj, width, height, min_size, stats, pair.xml_path)
        if box is None:
            continue
        class_id = code_map[code]
        counts[class_id] += 1
        lines.append(" ".join([str(class_id)] + [YOLO_FLOAT_FORMAT.format(v) for v in box]))

    (labels_dir / f"{pair.uid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    stats.boxes_written.update(counts)
    row = {"uid": pair.uid, "image_path": pair.image_path.as_posix(), "xml_path": pair.xml_path.as_posix(),
           "width": width, "height": height, "num_boxes": len(lines)}
    row.update({f"n_class_{cid}": counts.get(cid, 0) for cid in sorted(set(code_map.values()))})
    return row


def collect_sources(raw_dir: Path, extra_dirs: list[Path], cfg: dict) -> list[tuple[str, Path, list]]:
    """Discover RDD2022 pairs plus pairs from each extra folder (images without XML are ignored)."""
    exts, ann_ext = get_image_extensions(cfg), cfg["dataset"]["annotation_extension"]
    rdd = discover_dataset(raw_dir, exts, ann_ext)
    inside_extra = lambda p: any(d in p.parents for d in extra_dirs)  # noqa: E731
    rdd_pairs = [p for p in rdd.pairs if not inside_extra(p.image_path)]
    if not rdd_pairs:
        raise FileNotFoundError(f"No RDD2022 image/XML pairs found under {raw_dir}. Run src/inspect_dataset.py first.")
    sources = [(SOURCE_RDD, raw_dir, rdd_pairs)]
    for extra in extra_dirs:
        if not extra.is_dir():
            raise FileNotFoundError(f"Extra dataset folder not found: {extra} (dataset.extra_dirs in config.yaml)")
        found = discover_dataset(extra, exts, ann_ext, uid_prefix=EXTRA_UID_PREFIX)
        logger.info("Extra folder %s: %d images, %d with XML (images without XML are not used).",
                    extra, len(found.images), len(found.pairs))
        sources.append((SOURCE_EXTRA, extra, found.pairs))
    return sources


def prepare_output_dir(interim_dir: Path, force: bool) -> Path:
    """Create (or, with --force, recreate) the interim labels directory."""
    labels_dir = interim_dir / LABELS_SUBDIR
    if labels_dir.exists() and any(labels_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{labels_dir} already contains labels. Re-run with --force to overwrite.")
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return labels_dir


def log_summary(stats: ConversionStats, names: dict[int, str], num_pairs: int) -> None:
    """Print the end-of-run summary."""
    logger.info("=" * 60)
    logger.info("Conversion summary")
    logger.info("  matched pairs          : %d", num_pairs)
    logger.info("  label files written    : %d", stats.images_written)
    logger.info("  images skipped         : %d", stats.images_skipped)
    for cid, name in names.items():
        logger.info("  boxes %-17s: %d", name, stats.boxes_written.get(cid, 0))
    logger.info("  boxes clipped to image : %d", stats.boxes_clipped)
    logger.info("  boxes with swapped mins: %d", stats.boxes_swapped)
    logger.info("  degenerate boxes dropped: %d", stats.boxes_dropped)
    ignored = sum(stats.ignored_labels.values())
    logger.info("  annotations ignored (non-target classes): %d %s", ignored, dict(stats.ignored_labels) or "")
    logger.info("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--raw-dir", default=None, help="Override dataset.raw_dir")
    parser.add_argument("--force", action="store_true", help="Overwrite existing converted labels")
    parser.add_argument("--no-extra", action="store_true", help="Ignore dataset.extra_dirs")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    raw_dir = resolve_path(args.raw_dir or cfg["dataset"]["raw_dir"])
    interim_dir = resolve_path(cfg["dataset"]["interim_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    code_map = get_source_code_map(cfg)
    min_size = float(cfg["dataset"].get("min_box_size_px", 1.0))

    extra_dirs = [] if args.no_extra else [resolve_path(d) for d in cfg["dataset"].get("extra_dirs") or []]
    try:
        sources = collect_sources(raw_dir, extra_dirs, cfg)
        labels_dir = prepare_output_dir(interim_dir, args.force)
    except (FileNotFoundError, FileExistsError) as exc:
        logger.error("%s", exc)
        return 1

    stats = ConversionStats()
    rows = []
    total_pairs = sum(len(pairs) for _, _, pairs in sources)
    for source, root, pairs in sources:
        logger.info("Converting %d image/XML pairs from %s (%s)", len(pairs), root, source)
        for pair in tqdm(pairs, desc=f"Converting {source}", unit="img"):
            row = convert_pair(pair, code_map, labels_dir, min_size, stats)
            if row is None:
                stats.images_skipped += 1
            else:
                stats.images_written += 1
                row["source"] = source
                rows.append(row)

    pd.DataFrame(rows).to_csv(interim_dir / MANIFEST_NAME, index=False)
    reports_dir.mkdir(parents=True, exist_ok=True)
    issues_path = reports_dir / "conversion_issues.csv"
    pd.DataFrame(stats.issues, columns=["file", "severity", "message"]).to_csv(issues_path, index=False)
    for issue in stats.issues[:10]:
        logger.warning("%s: %s", Path(issue["file"]).name, issue["message"])
    if len(stats.issues) > 10:
        logger.warning("... %d more issues, see %s", len(stats.issues) - 10, issues_path)

    log_summary(stats, get_class_names(cfg), total_pairs)
    if extra_dirs:
        per_source = pd.DataFrame(rows).groupby("source")[[c for c in rows[0] if c.startswith("n_class_")]].sum()
        per_source.columns = [get_class_names(cfg)[int(c.split("_")[-1])] for c in per_source.columns]
        logger.info("Boxes per source (extra = training only):\n%s", per_source.to_string())
    logger.info("Labels: %s", labels_dir)
    logger.info("Manifest: %s", interim_dir / MANIFEST_NAME)
    logger.info("Issues report: %s", issues_path)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
