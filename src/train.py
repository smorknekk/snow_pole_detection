"""
train.py — Train a YOLOv8 model to detect snow poles.

Usage (from the project root):
    python src/train.py

The script reads all settings from configs/config.yaml and
configs/dataset.yaml, then starts training via the Ultralytics API.
Results (weights, plots, metrics) are saved under outputs/.
"""

import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a Python dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train(config_path: str = "configs/config.yaml") -> None:
    """
    Main training function.

    Args:
        config_path: Path to the training config YAML file.
    """
    # ------------------------------------------------------------------ #
    # 1. Load training config
    # ------------------------------------------------------------------ #
    print(f"[INFO] Loading config from: {config_path}")
    cfg = load_config(config_path)

    # ------------------------------------------------------------------ #
    # 2. Load the pretrained YOLOv8 model
    #    The first time this runs it downloads the weights automatically.
    # ------------------------------------------------------------------ #
    model_name = cfg.get("model", "yolov8n.pt")
    print(f"[INFO] Loading model: {model_name}")
    model = YOLO(model_name)

    # ------------------------------------------------------------------ #
    # 3. Start training
    #    We pass each hyperparameter individually so the call is clear.
    # ------------------------------------------------------------------ #
    print("[INFO] Starting training …")
    results = model.train(
        data=cfg["dataset"],            # path to dataset.yaml
        epochs=cfg.get("epochs", 50),
        batch=cfg.get("batch_size", 16),
        imgsz=cfg.get("image_size", 640),
        workers=cfg.get("workers", 4),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        augment=cfg.get("augment", True),
        project=cfg.get("project", "outputs"),
        name=cfg.get("name", "snow_pole_run"),
        save_period=cfg.get("save_period", 10),
        device=cfg.get("device", ""),   # auto-select GPU/CPU
    )

    # ------------------------------------------------------------------ #
    # 4. Report where results were saved
    # ------------------------------------------------------------------ #
    save_dir = Path(results.save_dir)
    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Results saved to: {save_dir}")
    print(f"[INFO] Best weights   : {save_dir / 'weights' / 'best.pt'}")
    print(f"[INFO] Last weights   : {save_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    train()
