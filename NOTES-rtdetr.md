# TDT4265 Mini-Project — Personal Notes
# Snow Pole Detection — RT-DETR Implementation

---

## Project Overview

- **Course:** TDT4265
- **Task:** Object detection of snow poles for autonomous driving in winter conditions
- **Dataset:** roadpoles_v1 — 322 train / 92 val images, 1 class (`pole`), YOLO format
- **Dataset location:** `/datasets/tdt4265/Poles2025/roadpoles_v1`
- **Group size:** 2 → must implement at least 2 architectures, max 1 YOLO variant
- **My role:** RT-DETR (non-YOLO architecture)
- **Partner's role:** YOLOv8n (YOLO variant)
- **Framework:** Ultralytics (Python), NVIDIA RTX 4090, CUDA 12.4

---

## Architecture: RT-DETR vs YOLOv8

### YOLOv8 (partner)
- CNN-based, anchor-free
- Divides image into a grid; predicts boxes per grid cell
- Uses Feature Pyramid Network (FPN) for multi-scale detection
- Requires Non-Maximum Suppression (NMS) as post-processing
- Very fast, lightweight (nano variant ~3M params)

### RT-DETR (mine)
- Transformer-based (encoder-decoder)
- Uses a fixed set of **learnable object queries** that attend to the entire image via cross-attention
- No grid, no anchors, **no NMS needed** — queries compete directly
- Backbone: HGNetv2 (in Ultralytics implementation)
- "RT" = Real-Time; first transformer detector fast enough for real-time use (Baidu, 2023)
- ~32M parameters, 103.4 GFLOPs

### Key difference for report:
YOLO asks "is there an object in each grid cell?" — RT-DETR asks "what objects are in this image?" via attention over the whole image. This makes RT-DETR better at global context but more computationally expensive.

---

## Hyperparameter Decisions (config_rtdetr.yaml)

| Parameter | YOLOv8n (partner) | RT-DETR-l (mine) | Reason for change |
|---|---|---|---|
| model | yolov8n.pt | rtdetr-l.pt | Different architecture |
| batch_size | 16 | 8 | RT-DETR is heavier |
| lr0 | 0.01 | 0.0001 | Transformers need much lower LR |
| lrf | 0.01 | 0.0001 | Consistent with lr0 |
| weight_decay | 0.0005 | 0.0001 | Slightly lower works better for transformers |
| optimizer | SGD (auto) | AdamW (auto) | Ultralytics selects AdamW for RT-DETR automatically |
| epochs | 50 | 50 | Same for fair comparison |
| image_size | 640 | 640 | Same |

---

## Files Created / Modified

| File | Action | Notes |
|---|---|---|
| `configs/config_rtdetr.yaml` | Created | RT-DETR specific hyperparameters |
| `configs/dataset.yaml` | Already existed | Shared dataset config, unchanged |
| `src/train_rtdetr.py` | Created | Based on train.py, uses `RTDETR` class |

---

## Steps Completed

- [x] Understood project requirements and architecture differences
- [x] Created `configs/config_rtdetr.yaml` with RT-DETR hyperparameters
- [x] Verified dataset path (`/datasets/tdt4265/Poles2025/roadpoles_v1`) — shared, no copy needed
- [x] Wrote `src/train_rtdetr.py`
- [x] Ran training (50 epochs, ~9 minutes on RTX 4090)
- [ ] Submit to leaderboard (roadpoles_v1 submission page)
- [ ] Compare full metrics with partner (get partner's Precision/Recall/mAP50-95)

---

## Results Comparison

| Metric | YOLOv8n (partner) | RT-DETR-l (mine) |
|---|---|---|
| Precision | — | 0.898 |
| Recall | — | 0.940 |
| **mAP@50** | **0.806** | **0.974** |
| mAP@0.5:0.95 | — | 0.600 |
| Training time | — | ~9 min (RTX 4090) |
| Parameters | ~3M | ~32M |

**Best weights:** `runs/detect/outputs/snow_pole_rtdetr4/weights/best.pt`

### Leaderboard Results (hidden test set, 46 images)

| Metric | Score |
|---|---|
| **mAP@0.5:0.95** (leaderboard metric) | **64.38%** |
| mAP@50 | 94.12% |
| AR@10 (Average Recall, max 10 preds) | 70.17% |

---

## Edge Deployment Note (for report/presentation)

The project brief mentions models should be suited for edge/real-time use. RT-DETR-l is larger than YOLOv8n. Address in report:
- RT-DETR was chosen to maximize detection accuracy and compare transformer vs CNN architectures
- For actual edge deployment, model compression (quantization, pruning) or a smaller variant would be needed
- The "RT" in RT-DETR confirms real-time inference was a design goal, unlike earlier DETR variants

---

## Presentation Structure

*(Add details from Section 1.1 guidelines once reviewed)*

- [ ] Introduction — task, dataset, motivation
- [ ] Architecture explanation — YOLOv8 vs RT-DETR (with diagrams if possible)
- [ ] Implementation details — configs, hyperparameter choices, why
- [ ] Results — metrics table, visual predictions
- [ ] Comparison between architectures
- [ ] Sustainability section *(see Section 1.2 guidelines)*
- [ ] Conclusion

**Time limit: 14 minutes (group of 2)**

---

## Sustainability Notes

*(Add content per Section 1.2 guidelines)*
- Training took only ~9 minutes on RTX 4090 — low energy footprint
- Consider: energy cost of training (GPU hours), dataset collection impact, societal benefit of safer autonomous driving in winter

---

## Useful Commands

```bash
# Activate venv
cd /work/avivan/snow_pole_detection
source venv/bin/activate

# Run RT-DETR training
python src/train_rtdetr.py

# Check GPU
nvidia-smi
```

---

## Resources

- [RT-DETR paper (Baidu, 2023)](https://arxiv.org/abs/2304.08069)
- [Ultralytics RT-DETR docs](https://docs.ultralytics.com/models/rtdetr/)
- [Project repo](https://github.com/smorknekk/snow_pole_detection)
