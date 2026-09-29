"""Fine-tune a pretrained Ultralytics YOLO detector on the prepared RDD2022 data.

All defaults come from config.yaml; common settings can be overridden:

    python src/train_detector.py
    python src/train_detector.py --model yolo11s.pt --epochs 150 --imgsz 800 --batch 8
    python src/train_detector.py --device cpu --epochs 1     # quick CPU smoke run
    python src/train_detector.py --resume                     # continue the newest interrupted run
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config, resolve_device, resolve_path, set_seed, setup_logging

logger = logging.getLogger("train_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--model", default=None, help="Pretrained weights, e.g. yolo11n.pt or yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None, help="Batch size (-1 = Ultralytics auto-batch on GPU)")
    parser.add_argument("--imgsz", "--image-size", dest="imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | 0 | 0,1")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--name", default=None, help="Run name under runs/detect/")
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="Resume an interrupted run: path to its last.pt, or no value for the newest run")
    return parser.parse_args()


def build_train_params(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config values with command-line overrides."""
    t = cfg["training"]

    def pick(cli_value, cfg_value):
        return cfg_value if cli_value is None else cli_value

    return {
        "model": pick(args.model, cfg["model"]["name"]),
        "epochs": int(pick(args.epochs, t["epochs"])),
        "batch": int(pick(args.batch, t["batch"])),
        "imgsz": int(pick(args.imgsz, t["image_size"])),
        "device": resolve_device(pick(args.device, t.get("device", "auto"))),
        "workers": int(pick(args.workers, t["workers"])),
        "patience": int(pick(args.patience, t["patience"])),
        "seed": int(t["seed"]),
        "deterministic": bool(t.get("deterministic", True)),
        "name": pick(args.name, t.get("run_name", "train")),
    }


def find_resume_checkpoint(runs_dir: Path, resume: str) -> Path:
    """Return the last.pt to resume from: an explicit path, or the newest one under runs_dir."""
    if resume != "latest":
        path = resolve_path(resume)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume not found: {path}")
        return path
    candidates = sorted(runs_dir.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No runs/*/weights/last.pt found under {runs_dir}; nothing to resume.")
    return candidates[0]


def check_dataset(data_yaml: Path) -> dict[str, Any]:
    """Make sure the YOLO dataset exists and return the split summary if present."""
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"{data_yaml} not found. Prepare the dataset first:\n"
            "  python src/convert_voc_to_yolo.py\n  python src/split_dataset.py"
        )
    for split in ("train", "val"):
        folder = data_yaml.parent / "images" / split
        if not folder.is_dir() or not any(folder.iterdir()):
            raise FileNotFoundError(f"No images in {folder}. Re-run src/split_dataset.py.")
    summary_path = data_yaml.parent / "split_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}


def record_run(save_dir: Path, params: dict[str, Any], data_yaml: Path, split_summary: dict[str, Any]) -> None:
    """Write run_metadata.json next to the Ultralytics outputs."""
    import torch
    import ultralytics

    meta = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "data_yaml": data_yaml.as_posix(),
        "dataset_split": split_summary,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics.__version__},
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (save_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    data_yaml = resolve_path(cfg["dataset"]["yolo_dir"]) / "data.yaml"
    runs_dir = resolve_path(cfg["paths"]["runs_dir"])

    try:
        split_summary = check_dataset(data_yaml)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run: pip install -r requirements.txt")
        return 1

    if args.resume:
        try:
            checkpoint = find_resume_checkpoint(runs_dir, args.resume)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Resuming training from %s (original settings are kept)", checkpoint)
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        trainer_args = model.trainer.args
        params = {k: getattr(trainer_args, k, None)
                  for k in ("model", "epochs", "batch", "imgsz", "device", "workers", "patience", "seed", "name")}
        params["resumed_from"] = checkpoint.as_posix()
    else:
        params = build_train_params(cfg, args)
        set_seed(params["seed"])
        logger.info("Training configuration:")
        for key, value in params.items():
            logger.info("  %-13s %s", key, value)
        if params["device"] == "cpu":
            logger.warning("Training on CPU - this will be slow. Install a CUDA build of PyTorch to use a GPU.")
        model_name = params.pop("model")
        model = YOLO(model_name)  # pretrained checkpoint; Ultralytics downloads official weights if missing
        model.train(data=data_yaml.as_posix(), project=runs_dir.as_posix(), pretrained=True, **params)
        params["model"] = model_name

    save_dir = Path(model.trainer.save_dir)
    record_run(save_dir, params, data_yaml, split_summary)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        logger.error("Training finished but %s was not produced.", best)
        return 1

    export_to = resolve_path(cfg["training"].get("export_weights_to"))
    if export_to:
        export_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, export_to)
        logger.info("Copied best weights to %s", export_to)
    logger.info("Run directory: %s", save_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
