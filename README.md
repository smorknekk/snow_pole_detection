# Snow Pole Detection

A student computer-vision project that trains a **YOLOv8** object-detection model to locate snow poles in images.

---

## Project Structure

```
snow_pole_detection/
├── configs/
│   ├── dataset.yaml   # dataset paths and class names
│   └── config.yaml    # training hyperparameters
├── data/
│   ├── images/
│   │   ├── train/     # training images (.jpg / .png)
│   │   └── val/       # validation images
│   └── labels/
│       ├── train/     # YOLO-format label files (.txt)
│       └── val/
├── notebooks/         # Jupyter notebooks for exploration
├── outputs/           # training results, weights, plots (git-ignored)
├── src/
│   ├── train.py       # training script
│   ├── evaluate.py    # evaluation / metrics script
│   └── utils.py       # shared helper functions
├── venv/              # virtual environment (git-ignored)
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/snow_pole_detection.git
cd snow_pole_detection
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install ultralytics pyyaml
```

---

## Add Your Data

Place your images and YOLO-format label files into the correct folders:

```
data/images/train/   ← training images
data/images/val/     ← validation images
data/labels/train/   ← one .txt label file per training image
data/labels/val/     ← one .txt label file per validation image
```

Each `.txt` label file should contain one line per bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalised** (between 0 and 1). For this project `class_id` is always `0` (snow_pole).

---

## Train

```bash
python src/train.py
```

Results (weights, loss curves, sample predictions) are saved under `outputs/snow_pole_run/`.

You can adjust hyperparameters in [configs/config.yaml](configs/config.yaml).

---

## Evaluate

```bash
python src/evaluate.py --weights outputs/snow_pole_run/weights/best.pt
```

This prints Precision, Recall, mAP@0.5, and mAP@0.5:0.95 on the validation set.

---

## Dataset Utilities

A quick check from a Python shell or notebook:

```python
import sys
sys.path.insert(0, "src")
from utils import print_dataset_summary, verify_labels

print_dataset_summary()
verify_labels("data/images/train", "data/labels/train")
```

---

## Model

The project uses **YOLOv8 nano** (`yolov8n.pt`) by default — the smallest and fastest variant, ideal for experimentation on a laptop.  
Change the `model` field in [configs/config.yaml](configs/config.yaml) to switch to a larger variant (s / m / l / x).

---

## License

This project is for educational purposes.
