"""Split converted data into train/val/test and build the YOLO dataset + data.yaml.

Reads ``<interim_dir>/manifest.csv`` (from convert_voc_to_yolo.py) and writes::

    <yolo_dir>/images/{train,val,test}/<uid>.<ext>
    <yolo_dir>/labels/{train,val,test}/<uid>.txt
    <yolo_dir>/data.yaml
    <yolo_dir>/split_summary.json
    <yolo_dir>/splits/{train,val,test}.txt

Leakage control (config ``split.strategy``): images in the same group always
land in the same split. ``random`` treats every image as its own group, which
is the documented default because RDD2022 India filenames carry no reliable
sequence/location metadata. See config.yaml for ``block`` and ``regex``.

Extra images (manifest ``source == "extra"``, from dataset.extra_dirs) are
always put in the training split. Only RDD2022 images are split, so val/test
are pure RDD2022 India and identical with or without extra data.

Usage:
    python src/split_dataset.py [--config config.yaml] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src.common import get_class_names, load_config, resolve_path, set_seed, setup_logging
from src.convert_voc_to_yolo import LABELS_SUBDIR, MANIFEST_NAME, SOURCE_EXTRA

logger = logging.getLogger("split_dataset")

SPLITS = ("train", "val", "test")
RATIO_TOLERANCE = 1e-6
TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)(?!.*\d)")


def validate_ratios(split_cfg: dict[str, Any]) -> dict[str, float]:
    """Return split ratios, checking they are non-negative and sum to 1."""
    ratios = {s: float(split_cfg.get(s, 0.0)) for s in SPLITS}
    if any(r < 0 for r in ratios.values()) or ratios["train"] <= 0 or ratios["val"] <= 0:
        raise ValueError(f"Invalid split ratios {ratios}: train and val must be > 0, test >= 0.")
    if abs(sum(ratios.values()) - 1.0) > RATIO_TOLERANCE:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios.values()):.4f}: {ratios}")
    return ratios


def assign_groups(uids: pd.Series, split_cfg: dict[str, Any]) -> pd.Series:
    """Map every image uid to a group key according to split.strategy."""
    strategy = str(split_cfg.get("strategy", "random")).lower()
    if strategy == "random":
        return uids.copy()
    if strategy == "block":
        block_size = int(split_cfg.get("block_size", 1))

        def block_key(uid: str) -> str:
            m = TRAILING_NUMBER.match(uid)
            return f"{m.group(1)}#{int(m.group(2)) // block_size}" if m else uid

        return uids.map(block_key)
    if strategy == "regex":
        pattern = split_cfg.get("group_regex")
        if not pattern:
            raise ValueError("split.strategy is 'regex' but split.group_regex is not set.")
        compiled = re.compile(pattern)

        def regex_key(uid: str) -> str:
            m = compiled.search(uid)
            return m.group(1) if m and m.groups() else uid

        groups = uids.map(regex_key)
        unmatched = int((groups == uids).sum())
        if unmatched:
            logger.warning("%d images did not match group_regex and form singleton groups.", unmatched)
        return groups
    raise ValueError(f"Unknown split.strategy '{strategy}' (use random, block or regex).")


def _group_split(indices: np.ndarray, groups: np.ndarray, test_fraction: float, seed: int):
    """Split indices so that no group is shared; returns (keep, holdout)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    keep, holdout = next(splitter.split(indices, groups=groups))
    return indices[keep], indices[holdout]


def split_indices(groups: pd.Series, ratios: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Return {split: row indices}; groups never cross splits.

    Note: GroupShuffleSplit applies ratios to the number of *groups*, so with
    grouped strategies the per-image proportions are approximate.
    """
    all_idx = np.arange(len(groups))
    group_arr = groups.to_numpy()
    if groups.nunique() < len(SPLITS):
        raise ValueError(f"Only {groups.nunique()} groups; need at least {len(SPLITS)} to split.")
    holdout_fraction = ratios["val"] + ratios["test"]
    train_idx, rest_idx = _group_split(all_idx, group_arr, holdout_fraction, seed)
    if ratios["test"] == 0:
        return {"train": train_idx, "val": rest_idx, "test": np.array([], dtype=int)}
    test_share = ratios["test"] / holdout_fraction
    val_idx, test_idx = _group_split(rest_idx, group_arr[rest_idx], test_share, seed)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def prepare_output(yolo_dir: Path, force: bool) -> None:
    """Refuse to overwrite existing processed data unless --force is given."""
    targets = [yolo_dir / "images", yolo_dir / "labels", yolo_dir / "splits"]
    existing = [t for t in targets if t.exists() and any(t.rglob("*"))]
    if existing and not force:
        raise FileExistsError(f"{yolo_dir} already contains a processed dataset. Re-run with --force to rebuild it.")
    for t in targets:
        if t.exists():
            shutil.rmtree(t)
    for split in SPLITS:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (yolo_dir / "splits").mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str) -> str:
    """Copy, hardlink or symlink ``src`` to ``dst``; fall back to copying. Returns the mode used."""
    if mode in ("hardlink", "symlink"):
        try:
            if mode == "hardlink":
                os.link(src, dst)
            else:
                dst.symlink_to(src.resolve())
            return mode
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def write_split(rows: pd.DataFrame, split: str, labels_src: Path, yolo_dir: Path, mode: str) -> set[str]:
    """Place images and labels for one split; returns the set of link modes used."""
    used_modes: set[str] = set()
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"Writing {split}", unit="img"):
        image_src = Path(row.image_path)
        label_src = labels_src / f"{row.uid}.txt"
        if not image_src.is_file() or not label_src.is_file():
            raise FileNotFoundError(f"Missing source file for '{row.uid}': {image_src} / {label_src}. "
                                    "Re-run convert_voc_to_yolo.py --force.")
        used_modes.add(place_file(image_src, yolo_dir / "images" / split / f"{row.uid}{image_src.suffix.lower()}", mode))
        shutil.copy2(label_src, yolo_dir / "labels" / split / f"{row.uid}.txt")
    (yolo_dir / "splits" / f"{split}.txt").write_text("\n".join(rows["uid"]) + "\n", encoding="utf-8")
    return used_modes


def write_data_yaml(yolo_dir: Path, names: dict[int, str], has_test: bool) -> Path:
    """Write an Ultralytics data.yaml using forward-slash paths (valid on Windows and Linux)."""
    data = {"path": yolo_dir.resolve().as_posix(), "train": "images/train", "val": "images/val"}
    if has_test:
        data["test"] = "images/test"
    data["nc"] = len(names)
    data["names"] = {int(k): v for k, v in sorted(names.items())}
    out = yolo_dir / "data.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def box_counts(frame: pd.DataFrame, names: dict[int, str]) -> dict[str, int]:
    return {names[cid]: int(frame.get(f"n_class_{cid}", pd.Series(dtype=int)).sum()) for cid in names}


def build_summary(manifest: pd.DataFrame, parts: dict[str, np.ndarray], groups: pd.Series,
                  names: dict[int, str], cfg: dict[str, Any], ratios: dict[str, float],
                  extra: pd.DataFrame) -> dict[str, Any]:
    """Collect per-split image/box counts for split_summary.json (RDD2022 rows + extra training rows)."""
    summary: dict[str, Any] = {
        "strategy": cfg["split"].get("strategy", "random"), "seed": int(cfg["split"]["seed"]),
        "ratios": ratios, "num_groups": int(groups.nunique()), "splits": {},
        "extra_train": {"images": int(len(extra)), "boxes": box_counts(extra, names)},
    }
    for split, idx in parts.items():
        sub = manifest.iloc[idx]
        boxes = box_counts(sub, names)
        summary["splits"][split] = {"images": int(len(sub)), "images_without_boxes": int((sub["num_boxes"] == 0).sum()),
                                    "groups": int(groups.iloc[idx].nunique()), "boxes": boxes}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild an existing YOLO dataset")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    interim_dir = resolve_path(cfg["dataset"]["interim_dir"])
    yolo_dir = resolve_path(cfg["dataset"]["yolo_dir"])
    manifest_path = interim_dir / MANIFEST_NAME
    names = get_class_names(cfg)
    seed = int(cfg["split"]["seed"])
    set_seed(seed)

    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path} not found. Run src/convert_voc_to_yolo.py first.")
        full = pd.read_csv(manifest_path, dtype={"uid": str})
        is_extra = full["source"].eq(SOURCE_EXTRA) if "source" in full else pd.Series(False, index=full.index)
        manifest, extra = full[~is_extra].reset_index(drop=True), full[is_extra].reset_index(drop=True)
        if manifest.empty:
            raise ValueError(f"{manifest_path} has no RDD2022 images.")
        ratios = validate_ratios(cfg["split"])
        groups = assign_groups(manifest["uid"], cfg["split"])
        parts = split_indices(groups, ratios, seed)
        prepare_output(yolo_dir, args.force)
        modes: set[str] = set()
        for split, idx in parts.items():
            rows = manifest.iloc[idx]
            if split == "train" and not extra.empty:
                rows = pd.concat([rows, extra], ignore_index=True)
            modes |= write_split(rows, split, interim_dir / LABELS_SUBDIR, yolo_dir,
                                 str(cfg["split"].get("link_mode", "copy")))
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    data_yaml = write_data_yaml(yolo_dir, names, has_test=len(parts["test"]) > 0)
    summary = build_summary(manifest, parts, groups, names, cfg, ratios, extra)
    summary["file_placement"] = sorted(modes)
    (yolo_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Strategy: %s | seed: %d | groups: %d", summary["strategy"], seed, summary["num_groups"])
    if summary["strategy"] == "random":
        logger.info("Random per-image split: no grouping metadata used (see README, 'Data leakage').")
    for split, info in summary["splits"].items():
        logger.info("  %-5s images=%-6d empty=%-5d boxes=%s", split, info["images"],
                    info["images_without_boxes"], info["boxes"])
    if not extra.empty:
        logger.info("  + %d extra images added to train only, boxes=%s", len(extra), summary["extra_train"]["boxes"])
    logger.info("Wrote %s", data_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
