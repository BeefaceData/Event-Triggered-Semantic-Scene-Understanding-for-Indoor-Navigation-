# Event-Triggered Semantic Scene Understanding for Indoor Navigation

> Midterm Project — Advanced Computer Vision | Spring 2025

---

## Overview

This project proposes a selective, event-triggered indoor navigation pipeline that invokes a vision-language model (BLIP) only at geometrically ambiguous decision points — reducing average inference latency by 35–63% compared to always-on semantic pipelines while maintaining semantic decision capability.

---

## Project Structure

```
ACV_Project_Indoor_Navigation/
│
├── baseline1_geometry.py         # Baseline 1 — YOLOv8 + MiDaS geometry-only pipeline
├── baseline2_ocr.py              # Baseline 2 — Always-on EasyOCR semantic pipeline
├── baseline3_blip.py             # Baseline 3 — Always-on BLIP VQA semantic pipeline
├── proposed_method.py            # Proposed — Event-triggered selective semantic pipeline
│
├── dataset_evaluation.py         # Evaluation on NYU Depth V2 (.mat format)
├── dataset_evaluation_h5.py      # Evaluation on NYU Depth V2 (Kaggle h5 format)
│
├── nyu_depth_v2_labeled.mat      # NYU Depth V2 labeled dataset (download separately)
├── NYU_Depth_Dataset_V2/         # NYU Depth V2 h5 dataset (download separately)
│
├── results.json                  # Results from .mat evaluation
├── results_h5.json               # Results from h5 evaluation
│
└── README.md                     # This file
```

---

## Installation

### 1. Create and activate conda environment

```bash
conda create -n acv_project python=3.11 -y
conda activate acv_project
```

### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install all dependencies

```bash
pip install ultralytics
pip install opencv-contrib-python
pip install timm
pip install matplotlib
pip install numpy
pip install easyocr
pip install transformers
pip install scipy
pip install h5py
pip install datasets
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### NYU Depth V2 (Labeled .mat — 2.8GB)

Download directly:
```
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```
Place `nyu_depth_v2_labeled.mat` in the project root directory.

### NYU Depth V2 (Kaggle h5 — full dataset)

Download from:
```
https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2
```
Extract and place the `NYU_Depth_Dataset_V2/` folder in the project root directory.

---

## Running the Pipelines

### Baseline 1 — Geometry Only (YOLOv8 + MiDaS)

```bash
python baseline1_geometry.py
```
- Input: webcam (default) or video file
- Output: live window with depth map, obstacle boxes, navigation decision
- Press **Q** to quit and print evaluation summary

### Baseline 2 — Always-On OCR

```bash
python baseline2_ocr.py
```
- Input: webcam (default) or video file
- Output: live window with detected text boxes and navigation decision
- Press **Q** to quit and print evaluation summary

### Baseline 3 — Always-On BLIP

```bash
python baseline3_blip.py
```
- Input: webcam (default) or video file
- Output: live window with BLIP answer and navigation decision
- Press **Q** to quit and print evaluation summary

### Proposed Method — Event-Triggered

```bash
python proposed_method.py
```
- Input: webcam (default) or video file
- Output: live window showing trigger state (ON/OFF), decision, latency
- Press **Q** to quit and print full evaluation summary

### To run on a video file instead of webcam

Change the last line of any script from:
```python
run_baseline(0)   # 0 = webcam
```
To:
```python
run_baseline("path/to/your/video.mp4")
```

---

## Dataset Evaluation

### Run evaluation on NYU Depth V2 (.mat format)

```bash
python dataset_evaluation.py
```

### Run evaluation on NYU Depth V2 (h5 format)

```bash
python dataset_evaluation_h5.py
```

Both scripts evaluate all four pipelines on 100 frames and produce:
- A printed results table comparing all methods
- A saved `results.json` file with latency statistics

To evaluate more frames, change `NUM_FRAMES = 100` at the top of the script.

---

## Results Summary

### Webcam Evaluation

| Method | Avg Latency | Semantic Calls |
|---|---|---|
| Geometry Only | ~46ms | 0% |
| Semantic Always (OCR) | ~140ms | 100% |
| Semantic Always (BLIP) | ~200ms | 100% |
| **Proposed Method** | **~74ms avg** | **~17%** |

### NYU Depth V2 Dataset Evaluation (100 frames)

| Method | Avg Latency | Max Latency | Decisions |
|---|---|---|---|
| Geometry Only | ~46ms* | 7.5ms | left/center/right |
| Semantic Always (OCR) | 126ms | 178ms | None detected |
| Semantic Always (BLIP) | 121ms | 409ms | left/right |
| **Proposed Method** | **79ms** | **138ms** | **left/center/right** |

*Decision logic only — full pipeline ~46ms from webcam evaluation.

### Trigger Efficiency

| Condition | Trigger Rate | Avg Latency |
|---|---|---|
| Webcam (controlled) | 17% | ~74ms |
| NYU .mat (diverse scenes) | 68% | 79ms |
| NYU h5 (basement) | 59% | 76ms |

---

## Models Used

| Model | Role | Source |
|---|---|---|
| YOLOv8n | Obstacle detection | Ultralytics |
| MiDaS Small | Depth estimation | Intel ISL via PyTorch Hub |
| EasyOCR | Text detection | JaidedAI |
| BLIP-vqa-base | Visual question answering | Salesforce via HuggingFace |

---

## Key Findings

- Proposed method reduces latency by **35–63%** vs always-on BLIP
- OCR produces **zero valid decisions** on NYU Depth V2 — confirms VLM is necessary for datasets without signage
- Trigger rate varies by scene type (17%–68%) — adaptive threshold calibration is the primary next step

---

## Known Limitations

- No ground truth navigation labels — decision accuracy not yet measured
- Fixed trigger threshold (50.0) requires manual calibration per scene type
- Evaluation on static frames — closed-loop navigation not tested
- HM3DSem evaluation via Habitat-sim pending (requires Linux + Habitat-sim)

---

## Future Work

- Extend evaluation to HM3DSem via Habitat-sim on Linux
- Evaluate on ScanNet for semantic robustness across cluttered environments
- Ablation study: fixed vs adaptive vs random trigger thresholds
- Add ground truth labels for decision accuracy measurement
- Entropy-based ambiguity metric as alternative trigger condition

---

## References

1. Saranya M, Arulselvarani S. *Real-Time Obstacle Detection using YOLOv8 for Assistive Navigation.* INDJST, 2025.
2. Medjaldi A et al. *Cost-Effective Real-Time Obstacle Detection for AGVs using YOLOv8 and RGB-D.* ETASR, 2025.
3. Birkl R et al. *MiDaS v3.1 — A Model Zoo for Robust Monocular Relative Depth Estimation.* arXiv:2307.14460, 2023.
4. Li J et al. *BLIP-2: Bootstrapping Language-Image Pre-training.* arXiv:2301.12597, 2023.
5. Liu H et al. *Visual Instruction Tuning (LLaVA).* arXiv:2304.08485, 2023.
6. Silberman N et al. *Indoor Segmentation and Support Inference from RGBD Images.* ECCV, 2012.
7. Yadav K et al. *Habitat-Matterport 3D Semantics Dataset.* arXiv:2210.05633, 2022.

---

## License

Academic use only. Dataset usage subject to NYU Depth V2 MIT License and HM3DSem Matterport Terms of Use.
