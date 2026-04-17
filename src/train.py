import sys
import yaml
from ultralytics import YOLO

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f) #config like dictonray

model_name = cfg.get("model", "yolov8n.pt")

model = YOLO(model_name) #from internet

results = model.train(
    data=cfg["dataset"],
    epochs=cfg.get("epochs", 50),
    batch=cfg.get("batch_size", 16),
    imgsz=cfg.get("image_size", 640), #resize
    project=cfg.get("project", "outputs"),
    name=cfg.get("name", "run1")
)

print("done")
print("saved")
