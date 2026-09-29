"""Shared helpers: configuration, paths, logging, seeding and device selection."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

REQUIRED_SECTIONS = ("paths", "dataset", "classes", "model", "training", "split", "inference", "severity")

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing or invalid."""


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple console logger for command-line scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the YAML configuration file."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Config {config_path} is missing sections: {', '.join(missing)}")
    return cfg


def resolve_path(path: str | Path | None) -> Path | None:
    """Resolve a config path relative to the project root (absolute paths are kept)."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def to_project_relative(path: Path) -> str:
    """Forward-slash path relative to the project root if inside it, else absolute (for CSV files)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def get_class_names(cfg: dict[str, Any]) -> dict[int, str]:
    """Return {class_id: class_name} from the config."""
    return {int(k): str(v) for k, v in cfg["classes"]["names"].items()}


def get_source_code_map(cfg: dict[str, Any]) -> dict[str, int]:
    """Return {RDD2022 label code: class_id} from the config."""
    names = get_class_names(cfg)
    mapping = {str(k).strip().upper(): int(v) for k, v in cfg["classes"]["source_codes"].items()}
    unknown = [cid for cid in mapping.values() if cid not in names]
    if unknown:
        raise ConfigError(f"classes.source_codes references unknown class ids: {unknown}")
    return mapping


def get_image_extensions(cfg: dict[str, Any]) -> set[str]:
    """Return the lower-case image extensions (with dot) to discover."""
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in cfg["dataset"]["image_extensions"]}


def pretty_class_name(name: str) -> str:
    """'longitudinal_crack' -> 'Longitudinal crack'."""
    return name.replace("_", " ").strip().capitalize()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("PyTorch not installed; skipping torch seeding.")


def cuda_available() -> bool:
    """Return True if PyTorch is installed and can see a CUDA GPU."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_device(device: str | int | None) -> str:
    """Translate 'auto' into '0' (first GPU) or 'cpu'; pass explicit values through.

    Falls back to CPU with a warning if a GPU was requested but CUDA is unavailable.
    """
    value = "auto" if device is None else str(device).strip().lower()
    has_cuda = cuda_available()
    if value in ("", "auto"):
        return "0" if has_cuda else "cpu"
    if value != "cpu" and not has_cuda:
        logger.warning("Device '%s' requested but CUDA is not available; using CPU.", device)
        return "cpu"
    return value.replace("cuda:", "")


def find_latest_weights(runs_dir: Path) -> Path | None:
    """Return the most recently modified runs/detect/*/weights/best.pt, if any."""
    if not runs_dir.is_dir():
        return None
    candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_weights(cfg: dict[str, Any], weights: str | Path | None = None) -> Path:
    """Find detector weights: explicit arg > config inference.weights > latest training run.

    Raises FileNotFoundError with a helpful message if nothing is found.
    """
    if weights:
        path = resolve_path(weights)
        if path and path.is_file():
            return path
        raise FileNotFoundError(f"Weights file not found: {path}")
    configured = resolve_path(cfg["inference"].get("weights"))
    if configured and configured.is_file():
        return configured
    latest = find_latest_weights(resolve_path(cfg["paths"]["runs_dir"]))
    if latest:
        logger.info("Configured weights not found; using latest training run: %s", latest)
        return latest
    raise FileNotFoundError(
        "No trained detector found. Train one with `python src/train_detector.py` "
        f"(expected {configured} or runs/detect/*/weights/best.pt)."
    )
