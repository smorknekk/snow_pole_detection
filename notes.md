## Progress Log

### Setup
- Project structure created with train.py, evaluate.py, utils.py, configs/
- Dataset: roadpoles_v1 (322 train, 92 val images, 1 class: pole)
- Dataset path: /datasets/tdt4265/Poles2025/roadpoles_v1/

### Run 1 — YOLOv8n baseline
- Model: yolov8n.pt (pretrained)
- Epochs: 50, batch: 16, imgsz: 640
- GPU: NVIDIA RTX 4090
- Training time: ~72 seconds
- Results (best.pt on val set):
    Precision: 0.677
    Recall:    0.850
    mAP@0.50:  0.806
    mAP@0.5:0.95: 0.466
- Weights: runs/detect/outputs/snow_pole_run3/weights/best.pt

### EDA — roadpoles_v1
- Train images: 322, Val images: 92
- Train boxes: 392, Val boxes: 113
- Avg poles per image: 1.22
- Box width  — min: 0.0022, max: 0.0759, avg: 0.0091
- Box height — min: 0.0524, max: 0.4358, avg: 0.1219
- Key insight: poles are extremely thin (avg width < 1% of image width)
- Plots saved: outputs/train_widths.png, outputs/train_heights.png

### Run 2 — YOLOv8n, 100 epochs
- Model: yolov8n.pt (3M params, 8.1 GFLOPs)
- Epochs: 100, batch: 16, imgsz: 640
- GPU: NVIDIA RTX 4090
- Training time: ~140 seconds
- Results (best.pt on val set):
    Precision: 0.764
    Recall:    0.746
    mAP@0.50:  0.813
    mAP@0.5:0.95: 0.470
- Weights: runs/detect/outputs/snow_pole_run_with_100epochs/weights/best.pt

### Run 3 — YOLOv8m, 150 epochs
- Model: yolov8m.pt (25.8M params, 78.7 GFLOPs)
- Epochs: 150, batch: 16, imgsz: 640
- GPU: NVIDIA RTX 4090
- Training time: ~6.7 min (0.112 hours)
- Results (best.pt on val set):
    Precision: 0.938
    Recall:    0.808
    mAP@0.50:  0.874
    mAP@0.5:0.95: 0.527
- Weights: runs/detect/outputs/snow_pole_run_with_150epochs/weights/best.pt

### Training comparison
| Run | Model    | Epochs | mAP50 | mAP50-95 |
|-----|----------|--------|-------|----------|
| 1   | YOLOv8n  | 50     | 0.806 | 0.466    |
| 2   | YOLOv8n  | 100    | 0.813 | 0.470    |
| 3   | YOLOv8m  | 150    | 0.874 | 0.527    |
Key takeaway: switching to YOLOv8m had more impact than doubling epochs.

### Run 4 — YOLOv8l, 267 epochs
- Model: yolov8l.pt (43.6M params, 164.8 GFLOPs)
- Epochs: 267, batch: 16, imgsz: 640
- GPU: NVIDIA RTX 4090
- Training time: ~17 min (0.288 hours)
- Results (best.pt on val set):
    Precision: 0.958
    Recall:    0.809
    mAP@0.50:  0.873
    mAP@0.5:0.95: 0.535
- Weights: runs/detect/outputs/snow_pole_run_with_267epochs_l/weights/best.pt
- Note: no improvement over YOLOv8m on mAP50 — bigger model not worth it on this small dataset

### Run 5 — YOLOv8s, 167 epochs
- Model: yolov8s.pt (11.1M params, 28.4 GFLOPs)
- Epochs: 167, batch: 16, imgsz: 640
- GPU: NVIDIA RTX 4090
- Training time: ~4 min (0.066 hours)
- Results (best.pt on val set):
    Precision: 0.857
    Recall:    0.867
    mAP@0.50:  0.876
    mAP@0.5:0.95: 0.536
- Weights: runs/detect/outputs/snow_pole_run_with_267epochs_l3/weights/best.pt
- Note: best mAP50 so far, fastest of the bigger models — YOLOv8s is the sweet spot for this dataset

### Leaderboard — First submission (roadpoles_v1 only training)
- roadpoles_v1 leaderboard: 63%
- iPhone leaderboard: 29%
- iPhone score was poor because the model was only trained on roadpoles_v1 dashcam footage.
  iPhone images have a completely different look (different camera, angle, lighting).
  The model had never seen that domain → poor generalization.
- Fix: create combined dataset (roadpoles_v1 + Road_poles_iPhone) and retrain.

### Combined dataset training (Run 6)
- Dataset: configs/dataset_combined.yaml (roadpoles_v1 + Road_poles_iPhone train/val)
- Model: yolov8s.pt, 167 epochs
- mAP50: 0.917, mAP50-95: 0.654
- Training time: 821s

### Leaderboard — Second submission (combined dataset model)
- roadpoles_v1 leaderboard: 63.07% (no change — v1 training data unchanged)
- iPhone leaderboard: 65.3% (was 29% — huge improvement from combined training)

### Sustainability
- Total training time: ~3300s (~55 min) on RTX 4090 (450W TDP)
- Energy: ~0.41 kWh
- Tesla Model Y (~16 kWh/100km): ~2.6 km equivalent

### TODO
- [x] EDA (exploratory data analysis)
- [x] Train YOLOv8n for 100 epochs
- [x] Train YOLOv8m for 150 epochs
- [x] First leaderboard submission (roadpoles_v1 only)
- [x] Train on combined dataset and resubmit to both leaderboards
- [ ] Partner: implement RT-DETR
- [ ] Make slides + record presentation video
- [ ] Submit on Blackboard