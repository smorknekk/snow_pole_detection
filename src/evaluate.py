import yaml
from ultralytics import YOLO
from pathlib import Path

weights = "runs/detect/outputs/snow_pole_run_with_100epochs/weights/best.pt"  # best checkpoint from training
config_path = "configs/config.yaml"

if not Path(weights).exists():  #stop early if weights are missing
    print("weights not found:", weights)
    print("run training first")
    exit()

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)  #load dataset path, image size, batch size

model = YOLO(weights)  #load the trained model

metrics = model.val(
    data=cfg["dataset"],                  
    imgsz=cfg.get("image_size", 640),     
    batch=cfg.get("batch_size", 16),       
    device=cfg.get("device", ""),          
    project=cfg.get("project", "outputs"), 
    name="evaluation"                      
)

print("precision:", metrics.box.p.mean())  # fraction of detections that were correct
print("recall:", metrics.box.r.mean())     # fraction of real poles that were found
print("map50:", metrics.box.map50)         #iou
print("map50-95:", metrics.box.map)        #more strict iou