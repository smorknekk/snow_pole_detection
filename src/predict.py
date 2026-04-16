import yaml
from ultralytics import YOLO
from pathlib import Path

weights = "runs/detect/outputs/snow_pole_combined_s/weights/best.pt"  # best model (YOLOv8s combined)
test_images = "/datasets/tdt4265/Poles2025/Road_poles_iPhone/images/Test/test"

model = YOLO(weights)

model.predict(
    source=test_images,
    project="outputs",
    name="iphone_submission2",
    save_txt=True,   # saves predictions as .txt files in YOLO format
    save_conf=True   # adds confidence score as last column (required by leaderboard)
)

print("predictions saved to runs/detect/outputs/iphone_submission/labels/")
print("zip that folder and upload to the leaderboard")
