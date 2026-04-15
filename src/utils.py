"""
utils.py — Small helper functions shared across the project.

Import anything you need with:
    from utils import count_images, verify_labels, plot_sample
"""

import random
from pathlib import Path

import yaml


# ------------------------------------------------------------------ #
# Config helpers
# ------------------------------------------------------------------ #

def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return its contents as a Python dict.

    Args:
        path: Path to the .yaml file.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
# Dataset helpers
# ------------------------------------------------------------------ #

def count_images(split_dir: str) -> int:
    """
    Count how many images are in a directory.

    Args:
        split_dir: Path to an image folder (e.g. 'data/images/train').

    Returns:
        Number of .jpg / .jpeg / .png files found.
    """
    image_extensions = {".jpg", ".jpeg", ".png"}
    folder = Path(split_dir)
    if not folder.exists():
        print(f"[WARN] Folder does not exist: {folder}")
        return 0
    count = sum(1 for f in folder.iterdir() if f.suffix.lower() in image_extensions)
    return count


def verify_labels(images_dir: str, labels_dir: str) -> None:
    """
    Check that every image has a corresponding YOLO label file (.txt).
    Prints a warning for each image that is missing a label.

    Args:
        images_dir: Path to the image folder  (e.g. 'data/images/train').
        labels_dir: Path to the labels folder (e.g. 'data/labels/train').
    """
    image_extensions = {".jpg", ".jpeg", ".png"}
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        print(f"[ERROR] Images folder not found: {images_path}")
        return

    missing = []
    for img_file in sorted(images_path.iterdir()):
        if img_file.suffix.lower() not in image_extensions:
            continue
        label_file = labels_path / (img_file.stem + ".txt")
        if not label_file.exists():
            missing.append(img_file.name)

    if missing:
        print(f"[WARN] {len(missing)} image(s) are missing label files:")
        for name in missing:
            print(f"       {name}")
    else:
        print(f"[OK] All images in '{images_dir}' have label files.")


def print_dataset_summary(dataset_yaml: str = "configs/dataset.yaml") -> None:
    """
    Print a quick summary of the dataset (image counts per split).

    Args:
        dataset_yaml: Path to the dataset YAML config.
    """
    cfg = load_yaml(dataset_yaml)
    root = Path(cfg.get("path", "data"))

    print("===== Dataset Summary =====")
    for split in ("train", "val", "test"):
        rel_path = cfg.get(split)
        if rel_path is None:
            continue
        full_path = root / rel_path
        n = count_images(str(full_path))
        print(f"  {split:5s}: {n} images  ({full_path})")
    print("===========================")


# ------------------------------------------------------------------ #
# Visualisation helpers
# ------------------------------------------------------------------ #

def get_random_image(split_dir: str) -> Path | None:
    """
    Return the path of a random image from a directory.

    Useful for quick visual spot-checks in a notebook.

    Args:
        split_dir: Path to an image folder.

    Returns:
        Path object, or None if folder is empty / missing.
    """
    image_extensions = {".jpg", ".jpeg", ".png"}
    folder = Path(split_dir)
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    if not images:
        print(f"[WARN] No images found in: {folder}")
        return None
    return random.choice(images)
