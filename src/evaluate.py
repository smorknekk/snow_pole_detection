"""
evaluate.py — Evaluate a trained YOLOv8 model on the validation set.

Usage (from the project root):
    python src/evaluate.py --weights outputs/snow_pole_run/weights/best.pt

The script runs YOLOv8 validation and prints the standard detection
metrics: Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
It also saves per-class results and confusion matrix plots to outputs/.
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a Python dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(weights: str, config_path: str = "configs/config.yaml") -> None:
    """
    Run validation on the validation split and print results.

    Args:
        weights:     Path to the trained model weights (.pt file).
        config_path: Path to the training config YAML (for dataset & settings).
    """
    # ------------------------------------------------------------------ #
    # 1. Load config so we know which dataset and image size to use
    # ------------------------------------------------------------------ #
    cfg = load_config(config_path)

    # ------------------------------------------------------------------ #
    # 2. Load the trained model from the given weights file
    # ------------------------------------------------------------------ #
    print(f"[INFO] Loading weights: {weights}")
    model = YOLO(weights)

    # ------------------------------------------------------------------ #
    # 3. Run validation
    # ------------------------------------------------------------------ #
    print("[INFO] Running evaluation on the validation set …")
    metrics = model.val(
        data=cfg["dataset"],
        imgsz=cfg.get("image_size", 640),
        batch=cfg.get("batch_size", 16),
        device=cfg.get("device", ""),
        project=cfg.get("project", "outputs"),
        name="evaluation",              # results saved under outputs/evaluation/
    )

    # ------------------------------------------------------------------ #
    # 4. Print a summary of the key metrics
    # ------------------------------------------------------------------ #
    print("\n========== Evaluation Results ==========")
    print(f"  Precision  (P) : {metrics.box.p.mean():.4f}")
    print(f"  Recall     (R) : {metrics.box.r.mean():.4f}")
    print(f"  mAP @ 0.50     : {metrics.box.map50:.4f}")
    print(f"  mAP @ 0.50:0.95: {metrics.box.map:.4f}")
    print("========================================")
    print(f"\n[INFO] Full results saved to: outputs/evaluation/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model.")
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs/snow_pole_run/weights/best.pt",
        help="Path to the trained model weights (.pt file).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the training config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        print("[HINT]  Train the model first with:  python src/train.py")
        raise SystemExit(1)

    evaluate(weights=str(weights_path), config_path=args.config)
