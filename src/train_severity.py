"""Train the Low / Medium / High severity classifier on human-labeled defect crops.

Inputs : data/severity_crops/metadata.csv  (src/crop_defects.py --save-crops)
         data/severity_crops/labels.csv    (streamlit run label_severity.py)
Outputs: models/severity_classifier.pt     (picked up automatically by the app)
         reports/severity/                 (metrics, confusion matrix, heuristic comparison)

The validation split is grouped by source image (crops of one photo never
appear in both splits) and stratified by severity level.

Usage:
    python src/train_severity.py [--epochs 30] [--arch mobilenet_v3_small] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.common import load_config, resolve_device, resolve_path, set_seed, setup_logging
from src.crop_defects import METADATA_NAME
from src.severity import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEVERITY_LEVELS,
    HeuristicSeverityEstimator,
    build_classifier,
    preprocess_crop,
    torch_device,
)

logger = logging.getLogger("train_severity")

BRIGHTNESS_JITTER = 0.15   # +-15% brightness augmentation
CONTRAST_JITTER = 0.15     # +-15% contrast augmentation
HFLIP_PROB = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--arch", default=None, help="mobilenet_v3_small | mobilenet_v3_large | resnet18")
    parser.add_argument("--device", default=None, help="auto | cpu | 0")
    return parser.parse_args()


def load_labeled_crops(cfg: dict[str, Any]) -> pd.DataFrame:
    """Join crop metadata with human labels; keep Low/Medium/High only."""
    metadata_path = resolve_path(cfg["severity"]["crops_dir"]) / METADATA_NAME
    labels_path = resolve_path(cfg["severity"]["training"]["labels_file"])
    for path, hint in ((metadata_path, "python src/crop_defects.py ... --save-crops"),
                       (labels_path, "streamlit run label_severity.py")):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Create it with: {hint}")
    labels = pd.read_csv(labels_path)[["crop_path", "severity"]]
    data = pd.read_csv(metadata_path).merge(labels, on="crop_path", how="inner")
    data = data[data["severity"].isin(SEVERITY_LEVELS)].reset_index(drop=True)
    data["crop_file"] = data["crop_path"].map(resolve_path)
    missing = ~data["crop_file"].map(lambda p: p.is_file())
    if missing.any():
        logger.warning("%d labeled crops are missing on disk and are skipped.", int(missing.sum()))
        data = data[~missing].reset_index(drop=True)
    data["target"] = data["severity"].map({lvl: i for i, lvl in enumerate(SEVERITY_LEVELS)})
    return data


def check_counts(data: pd.DataFrame, min_per_level: int) -> None:
    counts = data["severity"].value_counts().reindex(SEVERITY_LEVELS, fill_value=0)
    logger.info("Labeled crops per level: %s", counts.to_dict())
    too_few = counts[counts < min_per_level]
    if not too_few.empty:
        raise ValueError(f"Need at least {min_per_level} labeled crops per level; too few for: "
                         f"{too_few.to_dict()}. Label more with `streamlit run label_severity.py`.")


def split_train_val(data: pd.DataFrame, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified (by level), grouped (by source image) split."""
    n_splits = max(2, round(1 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(data, data["target"], groups=data["image_name"]))
    return train_idx, val_idx


class CropDataset:
    """Minimal torch-compatible dataset of labeled crops."""

    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool, seed: int) -> None:
        self.paths = frame["crop_file"].tolist()
        self.targets = frame["target"].tolist()
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        if self.rng.random() < HFLIP_PROB:
            crop = crop[:, ::-1]
        alpha = 1.0 + self.rng.uniform(-CONTRAST_JITTER, CONTRAST_JITTER)
        beta = 255.0 * self.rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER)
        return np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx: int):
        import torch

        crop = cv2.imread(str(self.paths[idx]))
        if self.augment:
            crop = self._augment(crop)
        x = torch.from_numpy(np.ascontiguousarray(preprocess_crop(crop, self.image_size)))
        return x, self.targets[idx]


def class_weights(targets: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so rare levels are not ignored."""
    counts = np.bincount(targets, minlength=len(SEVERITY_LEVELS)).astype(np.float32)
    return counts.sum() / (len(SEVERITY_LEVELS) * np.maximum(counts, 1.0))


def predict_all(model, loader, device: str) -> np.ndarray:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def train_model(train_ds: CropDataset, val_ds: CropDataset, val_targets: np.ndarray, tcfg: dict[str, Any],
                arch: str, device: str) -> tuple[Any, float, int]:
    """Fine-tune an ImageNet-pretrained classifier; return (best_state_dict, best_macro_f1, best_epoch)."""
    import torch
    from torch.utils.data import DataLoader

    try:
        model = build_classifier(arch, len(SEVERITY_LEVELS), pretrained=True)
    except Exception as exc:  # e.g. no internet to download ImageNet weights
        logger.warning("Could not load ImageNet weights (%s); training from random initialisation.", exc)
        model = build_classifier(arch, len(SEVERITY_LEVELS), pretrained=False)
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=int(tcfg["batch"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(tcfg["batch"]), shuffle=False, num_workers=0)
    weights = torch.tensor(class_weights(np.array(train_ds.targets)), device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]),
                                  weight_decay=float(tcfg["weight_decay"]))

    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), torch.as_tensor(y, device=device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        val_f1 = f1_score(val_targets, predict_all(model, val_loader, device), average="macro", zero_division=0)
        logger.info("epoch %3d  train_loss %.4f  val_macro_f1 %.3f", epoch, total_loss / len(train_ds), val_f1)
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(tcfg["patience"]):
                logger.info("Early stopping: no improvement for %d epochs.", stale)
                break
    return best_state, best_f1, best_epoch


def heuristic_predictions(frame: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """Heuristic estimate for each crop (to compare it against the human labels)."""
    estimator = HeuristicSeverityEstimator(cfg["severity"]["heuristic"])
    preds = []
    for row in frame.itertuples(index=False):
        det = {"class_name": row.class_name, "bbox": [row.x1, row.y1, row.x2, row.y2]}
        result = estimator.predict(cv2.imread(str(row.crop_file)), det, (row.image_height, row.image_width))
        preds.append(SEVERITY_LEVELS.index(result["severity"]))
    return np.array(preds)


def save_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(SEVERITY_LEVELS)), SEVERITY_LEVELS)
    ax.set_yticks(range(len(SEVERITY_LEVELS)), SEVERITY_LEVELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Human label")
    ax.set_title(title, loc="left")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> dict[str, Any]:
    labels = list(range(len(SEVERITY_LEVELS)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=list(SEVERITY_LEVELS),
                                   output_dict=True, zero_division=0)
    logger.info("%s on validation crops:\n%s", name, classification_report(
        y_true, y_pred, labels=labels, target_names=list(SEVERITY_LEVELS), zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    save_confusion_matrix(cm, f"{name} - validation", out_dir / f"confusion_matrix_{name.lower()}.png")
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
            "per_level": {lvl: report[lvl] for lvl in SEVERITY_LEVELS}, "confusion_matrix": cm.tolist()}


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tcfg = dict(cfg["severity"]["training"])
    tcfg["epochs"] = args.epochs or tcfg["epochs"]
    tcfg["batch"] = args.batch or tcfg["batch"]
    arch = args.arch or tcfg["arch"]
    seed = int(tcfg["seed"])
    set_seed(seed)

    try:
        data = load_labeled_crops(cfg)
        check_counts(data, int(tcfg["min_per_level"]))
        train_idx, val_idx = split_train_val(data, float(tcfg["val_fraction"]), seed)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    train_df, val_df = data.iloc[train_idx], data.iloc[val_idx]
    logger.info("Train crops: %d | validation crops: %d (grouped by source image)", len(train_df), len(val_df))

    device = torch_device(resolve_device(args.device or cfg["training"].get("device", "auto")))
    size = int(tcfg["image_size"])
    val_targets = val_df["target"].to_numpy()
    state, best_f1, best_epoch = train_model(CropDataset(train_df, size, True, seed),
                                             CropDataset(val_df, size, False, seed), val_targets, tcfg, arch, device)

    out_path = resolve_path(cfg["severity"]["model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save({"arch": arch, "state_dict": state, "labels": list(SEVERITY_LEVELS), "image_size": size,
                "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "best_epoch": best_epoch,
                "val_macro_f1": best_f1, "num_train": len(train_df), "num_val": len(val_df),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, out_path)
    logger.info("Saved severity model to %s (best epoch %d, val macro-F1 %.3f)", out_path, best_epoch, best_f1)

    from torch.utils.data import DataLoader

    model = build_classifier(arch, len(SEVERITY_LEVELS))
    model.load_state_dict(state)
    model.to(device)
    val_loader = DataLoader(CropDataset(val_df, size, False, seed), batch_size=int(tcfg["batch"]), num_workers=0)
    reports_dir = resolve_path(cfg["paths"]["reports_dir"]) / "severity"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": evaluate("Classifier", val_targets, predict_all(model, val_loader, device), reports_dir),
        "heuristic": evaluate("Heuristic", val_targets, heuristic_predictions(val_df, cfg), reports_dir),
        "num_train": len(train_df), "num_val": len(val_df), "arch": arch, "best_epoch": best_epoch,
        "label_counts": data["severity"].value_counts().to_dict(),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Validation macro-F1: classifier %.3f vs heuristic %.3f. Reports in %s",
                results["model"]["macro_f1"], results["heuristic"]["macro_f1"], reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
