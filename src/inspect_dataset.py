"""Inspect the raw RDD2022 download: file discovery, matching and class distribution.

Usage:
    python src/inspect_dataset.py [--config config.yaml] [--raw-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import (
    get_class_names,
    get_image_extensions,
    get_source_code_map,
    load_config,
    resolve_path,
    setup_logging,
)
from src.voc_dataset import DiscoveryResult, VocParseError, discover_dataset, parse_voc_xml, read_image_size

logger = logging.getLogger("inspect_dataset")

MAX_LISTED_FILES = 10
# Colours from the project's chart palette: target classes vs. ignored labels.
TARGET_BAR_COLOR = "#2a78d6"
IGNORED_BAR_COLOR = "#898781"
TEXT_COLOR = "#52514e"


def summarize_directories(result: DiscoveryResult, root: Path) -> None:
    """Log which folders contain images and which contain XML annotations."""
    for label, files in (("Image", result.images), ("Annotation (XML)", result.xmls)):
        counts = Counter(str(p.parent.relative_to(root)) for p in files)
        logger.info("%s directories:", label)
        for folder, n in counts.most_common():
            logger.info("    %-60s %6d files", folder, n)


def log_file_list(title: str, files: list[Path], root: Path) -> None:
    """Log up to MAX_LISTED_FILES entries of a file list."""
    if not files:
        return
    logger.warning("%s (%d), first %d:", title, len(files), min(len(files), MAX_LISTED_FILES))
    for p in files[:MAX_LISTED_FILES]:
        logger.warning("    %s", p.relative_to(root))


def collect_annotation_stats(result: DiscoveryResult) -> tuple[Counter, list[tuple[int, int]], list[str]]:
    """Parse all XML files; return label counts, image sizes and problem descriptions."""
    label_counts: Counter = Counter()
    sizes: list[tuple[int, int]] = []
    problems: list[str] = []
    image_for_xml = {p.xml_path: p.image_path for p in result.pairs}

    for xml_path in result.xmls:
        try:
            ann, obj_problems = parse_voc_xml(xml_path)
        except VocParseError as exc:
            problems.append(f"{xml_path}: {exc}")
            continue
        problems.extend(f"{xml_path}: {msg}" for msg in obj_problems)
        label_counts.update(o.name.strip().upper() for o in ann.objects)

        image_path = image_for_xml.get(xml_path)
        if image_path is None:
            continue
        try:
            sizes.append(read_image_size(image_path))
        except OSError as exc:
            problems.append(f"{image_path}: unreadable image ({exc})")
    return label_counts, sizes, problems


def build_distribution_table(label_counts: Counter, cfg: dict) -> pd.DataFrame:
    """Create a class distribution table including ignored (non-target) labels."""
    names = get_class_names(cfg)
    code_map = get_source_code_map(cfg)
    rows = []
    for code, class_id in sorted(code_map.items(), key=lambda kv: kv[1]):
        rows.append({"code": code, "class_id": class_id, "class_name": names[class_id],
                     "instances": label_counts.get(code, 0), "used": True})
    for code, n in sorted(label_counts.items()):
        if code not in code_map:
            rows.append({"code": code, "class_id": None, "class_name": "(ignored)", "instances": n, "used": False})
    df = pd.DataFrame(rows)
    df["class_id"] = df["class_id"].astype("Int64")
    total = df["instances"].sum()
    df["percent"] = (100.0 * df["instances"] / total).round(2) if total else 0.0
    return df


def plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Save a bar chart of instances per label code."""
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(df)), 4.5))
    colors = [TARGET_BAR_COLOR if used else IGNORED_BAR_COLOR for used in df["used"]]
    labels = [f"{c}\n{n.replace('_', chr(10))}" if u else f"{c}\n(ignored)"
              for c, n, u in zip(df["code"], df["class_name"], df["used"])]
    bars = ax.bar(labels, df["instances"], color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{v:,}" for v in df["instances"]], padding=3, fontsize=9, color=TEXT_COLOR)
    ax.set_ylabel("Instances", color=TEXT_COLOR)
    ax.set_title("RDD2022 India - annotated instances per class", loc="left")
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def log_size_stats(sizes: list[tuple[int, int]]) -> None:
    """Log image width/height statistics."""
    if not sizes:
        logger.info("No image dimensions collected.")
        return
    df = pd.DataFrame(sizes, columns=["width", "height"])
    logger.info("Image dimensions (%d images):", len(df))
    logger.info("    width : min %d  max %d  mean %.1f", df.width.min(), df.width.max(), df.width.mean())
    logger.info("    height: min %d  max %d  mean %.1f", df.height.min(), df.height.max(), df.height.mean())
    for (w, h), n in Counter(sizes).most_common(5):
        logger.info("    %dx%d: %d images", w, h, n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--raw-dir", default=None, help="Override dataset.raw_dir")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    root = resolve_path(args.raw_dir or cfg["dataset"]["raw_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])

    try:
        result = discover_dataset(root, get_image_extensions(cfg), cfg["dataset"]["annotation_extension"])
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Scanned: %s", root)
    logger.info("Images found              : %d", len(result.images))
    logger.info("XML files found           : %d", len(result.xmls))
    logger.info("Matched image/XML pairs   : %d", len(result.pairs))
    logger.info("Unmatched images          : %d", len(result.unmatched_images))
    logger.info("Unmatched XML files       : %d", len(result.unmatched_xmls))
    if not result.images and not result.xmls:
        logger.error("No images or XML files found. Check that the download is extracted into %s", root)
        return 1
    summarize_directories(result, root)
    log_file_list("Images without XML (not used for training; e.g. the unlabeled test split)",
                  result.unmatched_images, root)
    log_file_list("XML files without an image", result.unmatched_xmls, root)

    label_counts, sizes, problems = collect_annotation_stats(result)
    df = build_distribution_table(label_counts, cfg)
    logger.info("Class distribution (all XML files):\n%s", df.to_string(index=False))
    log_size_stats(sizes)

    if problems:
        logger.warning("%d annotation problems found, first %d:", len(problems), MAX_LISTED_FILES)
        for msg in problems[:MAX_LISTED_FILES]:
            logger.warning("    %s", msg)

    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / "class_distribution.csv", index=False)
    plot_distribution(df, reports_dir / "class_distribution.png")
    if problems:
        (reports_dir / "inspection_problems.txt").write_text("\n".join(problems), encoding="utf-8")
    logger.info("Saved %s and %s", reports_dir / "class_distribution.csv", reports_dir / "class_distribution.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
