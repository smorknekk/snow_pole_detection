"""
predict_rtdetr.py — Generate predictions on the test set for leaderboard submission.

Usage (from the project root):
    python src/predict_rtdetr.py

Predictions are saved as .txt files (YOLO format with confidence) to:
    outputs/rtdetr_submission/labels/

Zip that folder and upload to the roadpoles_v1 leaderboard.
"""

from ultralytics import RTDETR

TEST_SET = "/datasets/tdt4265/Poles2025/Road_poles_iPhone/images/Test/test"
WEIGHTS  = "runs/detect/outputs/snow_pole_rtdetr10/weights/best.pt"

model = RTDETR(WEIGHTS)

model.predict(
    source=TEST_SET,
    project="outputs",
    name="iphone_submission",
    save_txt=True,
    save_conf=True,
    imgsz=640
)

print("\n[INFO] Done! Zip the folder: outputs/rtdetr_submission/labels/")
print("[INFO] Upload the .zip to the roadpoles_v1 leaderboard.")
