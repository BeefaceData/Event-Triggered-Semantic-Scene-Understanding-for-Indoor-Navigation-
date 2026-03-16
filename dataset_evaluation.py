"""
Dataset Evaluation — NYU Depth V2
Project: Event-Triggered Semantic Scene Understanding for Indoor Navigation
Runs all four pipelines on real indoor scenes and produces a comparison table.

Dataset: NYU Depth V2
- 1449 densely labeled RGB-D images
- 26 indoor scene classes: bedroom, kitchen, office, corridor, etc.
- No license required — loads directly from Hugging Face

Usage:
    python dataset_evaluation.py
"""

import cv2
import torch
import numpy as np
import time
import json
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForQuestionAnswering
import easyocr
from datasets import load_dataset

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES          = 100        # How many frames to evaluate (increase for final paper)
SAVE_RESULTS_JSON   = "results.json"
CONFIDENCE_THRESHOLD = 0.4

OBSTACLE_CLASSES = [
    "person", "chair", "couch", "bed",
    "dining table", "toilet", "door"
]

DIRECTION_KEYWORDS = {
    "left":    ["left"],
    "right":   ["right"],
    "forward": ["forward", "straight", "ahead", "center"],
    "stop":    ["stop", "blocked", "back"],
}

FREE_PATH_DIFF_THRESHOLD = 50.0

print(f"[INFO] Running on: {DEVICE}")

# ─────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────
def load_nyu_dataset(num_frames=NUM_FRAMES):
    print(f"[INFO] Loading NYU Depth V2 from Hugging Face...")
    print(f"[INFO] This may take a few minutes on first download (~2.8GB)")
    dataset = load_dataset(
        "sayakpaul/nyu_depth_v2",
        trust_remote_code=True,
        split="train",
        storage_options={'client_kwargs': {'timeout': __import__('aiohttp').ClientTimeout(total=3600)}}
    )
    print(f"[INFO] Dataset loaded — {len(dataset)} total frames")
    print(f"[INFO] Using first {num_frames} frames for evaluation")
    return dataset

# ─────────────────────────────────────────────
# 3. LOAD ALL MODELS
# ─────────────────────────────────────────────
def load_models():
    print("[INFO] Loading YOLOv8...")
    yolo = YOLO("yolov8n.pt")
    yolo.to(DEVICE)

    print("[INFO] Loading MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(DEVICE)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print("[INFO] Loading EasyOCR...")
    ocr = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))

    print("[INFO] Loading BLIP-vqa-base...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        torch_dtype=torch.float16
    ).to(DEVICE)
    blip.eval()

    print("[INFO] All models loaded")
    return yolo, midas, transform, ocr, processor, blip

# ─────────────────────────────────────────────
# 4. SHARED GEOMETRY FUNCTIONS
# ─────────────────────────────────────────────
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def estimate_depth(frame, midas, transform):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(DEVICE)
    with torch.no_grad():
        depth = midas(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False
        ).squeeze()
    return depth.cpu().numpy()

def detect_obstacles(frame, yolo):
    results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    obstacles = []
    for result in results:
        for box in result.boxes:
            cls_name = yolo.names[int(box.cls)]
            if cls_name in OBSTACLE_CLASSES:
                obstacles.append({
                    "class": cls_name,
                    "bbox": box.xyxy[0].cpu().numpy()
                })
    return obstacles

def geometry_decision(depth_map, obstacles):
    h, w = depth_map.shape
    regions = {
        "left":   depth_map[:, :w//3],
        "center": depth_map[:, w//3:2*w//3],
        "right":  depth_map[:, 2*w//3:]
    }
    scores = {k: np.mean(v) for k, v in regions.items()}
    center_blocked = any(
        (box["bbox"][0] + box["bbox"][2]) / 2 > w // 3 and
        (box["bbox"][0] + box["bbox"][2]) / 2 < 2 * w // 3
        for box in obstacles
    )
    direction = min(scores, key=scores.get)
    if center_blocked and direction == "center":
        direction = "left" if scores["left"] < scores["right"] else "right"
    return direction, scores

def is_ambiguous(scores, obstacles):
    sorted_scores = sorted(scores.values())
    if len(sorted_scores) >= 2:
        diff = abs(sorted_scores[0] - sorted_scores[1])
        if diff < FREE_PATH_DIFF_THRESHOLD:
            return True, "multiple open paths"
    center_blocked = any(
        (box["bbox"][0] + box["bbox"][2]) / 2 > 0
        for box in obstacles
    )
    if center_blocked:
        return True, "center blocked"
    return False, ""

def blip_decision(frame, processor, blip):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    question  = "Which direction should I go to navigate forward, left, or right?"
    inputs    = processor(pil_image, question, return_tensors="pt")
    inputs    = {
        k: v.to(DEVICE).half() if v.dtype == torch.float32 else v.to(DEVICE)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        output = blip.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(output[0], skip_special_tokens=True).lower().strip()
    for direction, keywords in DIRECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in answer:
                return direction, answer
    return None, answer

def ocr_decision(frame, reader):
    results = reader.readtext(frame)
    for (bbox, text, conf) in results:
        if conf < 0.4:
            continue
        text = text.lower().strip()
        for direction, keywords in DIRECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return direction, text
    return None, None

# ─────────────────────────────────────────────
# 5. RUN ALL FOUR PIPELINES ON ONE FRAME
# ─────────────────────────────────────────────
def evaluate_frame(frame, yolo, midas, transform, ocr, processor, blip):
    results = {}

    # Shared geometry — compute once
    obstacles  = detect_obstacles(frame, yolo)
    depth_map  = estimate_depth(frame, midas, transform)

    # ── Baseline 1: Geometry Only ──
    t = time.time()
    geo_dir, scores = geometry_decision(depth_map, obstacles)
    results["baseline1_geometry"] = {
        "decision": geo_dir,
        "latency":  (time.time() - t) * 1000
    }

    # ── Baseline 2: Always-On OCR ──
    t = time.time()
    ocr_dir, ocr_text = ocr_decision(frame, ocr)
    results["baseline2_ocr"] = {
        "decision": ocr_dir,
        "latency":  (time.time() - t) * 1000
    }

    # ── Baseline 3: Always-On BLIP ──
    t = time.time()
    blip_dir, blip_ans = blip_decision(frame, processor, blip)
    results["baseline3_blip"] = {
        "decision":    blip_dir,
        "blip_answer": blip_ans,
        "latency":     (time.time() - t) * 1000
    }

    # ── Proposed Method: Event-Triggered ──
    t = time.time()
    triggered, reason = is_ambiguous(scores, obstacles)
    if triggered:
        sem_dir, sem_ans = blip_decision(frame, processor, blip)
        final_dir = sem_dir if sem_dir else geo_dir
    else:
        final_dir = geo_dir
        sem_ans   = None
    results["proposed_method"] = {
        "decision":    final_dir,
        "triggered":   triggered,
        "reason":      reason,
        "latency":     (time.time() - t) * 1000
    }

    return results

# ─────────────────────────────────────────────
# 6. AGGREGATE AND PRINT RESULTS TABLE
# ─────────────────────────────────────────────
def print_results_table(all_results, scene_types):
    methods = [
        "baseline1_geometry",
        "baseline2_ocr",
        "baseline3_blip",
        "proposed_method"
    ]
    labels = [
        "Geometry Only",
        "Semantic Always (OCR)",
        "Semantic Always (BLIP)",
        "Proposed Method"
    ]

    print("\n" + "="*70)
    print("DATASET EVALUATION — NYU DEPTH V2")
    print("="*70)
    print(f"{'Method':<25} {'Avg Latency':>12} {'Min':>8} {'Max':>8} {'Decisions':>10}")
    print("-"*70)

    summary = {}
    for method, label in zip(methods, labels):
        latencies = [r[method]["latency"] for r in all_results]
        decisions = [r[method]["decision"] for r in all_results]
        dec_counts = {d: decisions.count(d) for d in set(decisions)}

        summary[method] = {
            "avg_latency": np.mean(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "decisions":   dec_counts
        }

        print(f"{label:<25} {np.mean(latencies):>10.1f}ms "
              f"{np.min(latencies):>6.1f}ms "
              f"{np.max(latencies):>6.1f}ms "
              f"  {dec_counts}")

    # Trigger efficiency for proposed method
    triggered = sum(1 for r in all_results if r["proposed_method"]["triggered"])
    total     = len(all_results)
    print("-"*70)
    print(f"\nProposed Method Trigger Efficiency:")
    print(f"  Triggered frames : {triggered} / {total} ({triggered/total*100:.1f}%)")
    print(f"  Geometry frames  : {total-triggered} / {total} ({(total-triggered)/total*100:.1f}%)")

    # Scene type breakdown
    print(f"\nScene Types Evaluated:")
    scene_counts = {}
    for s in scene_types:
        scene_counts[s] = scene_counts.get(s, 0) + 1
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {scene:<30}: {count} frames")

    print("="*70)
    return summary

# ─────────────────────────────────────────────
# 7. MAIN EVALUATION LOOP
# ─────────────────────────────────────────────
def run_evaluation():
    # Load dataset
    dataset = load_nyu_dataset(NUM_FRAMES)

    # Load models
    yolo, midas, transform, ocr, processor, blip = load_models()

    all_results  = []
    scene_types  = []

    print(f"\n[INFO] Starting evaluation on {NUM_FRAMES} frames...")
    print("[INFO] Progress will update every 10 frames\n")

    for i in range(NUM_FRAMES):
        sample = dataset[i]

        # Convert PIL image to cv2
        frame = pil_to_cv2(sample["image"])

        # Get scene type if available
        scene = sample.get("scene", f"scene_{i}")
        scene_types.append(str(scene))

        # Evaluate all four pipelines
        results = evaluate_frame(
            frame, yolo, midas, transform, ocr, processor, blip
        )
        all_results.append(results)

        # Progress update
        if (i + 1) % 10 == 0:
            geo_lat  = np.mean([r["baseline1_geometry"]["latency"] for r in all_results])
            blip_lat = np.mean([r["baseline3_blip"]["latency"] for r in all_results])
            prop_lat = np.mean([r["proposed_method"]["latency"] for r in all_results])
            trig_pct = sum(1 for r in all_results if r["proposed_method"]["triggered"]) / len(all_results) * 100
            print(f"[Frame {i+1:>3}/{NUM_FRAMES}] "
                  f"Geo: {geo_lat:.0f}ms | "
                  f"BLIP: {blip_lat:.0f}ms | "
                  f"Proposed: {prop_lat:.0f}ms | "
                  f"Trigger rate: {trig_pct:.1f}%")

    # Print final results table
    summary = print_results_table(all_results, scene_types)

    # Save raw results to JSON for your report
    with open(SAVE_RESULTS_JSON, "w") as f:
        json.dump({
            "num_frames": NUM_FRAMES,
            "summary":    {k: {
                "avg_latency": v["avg_latency"],
                "min_latency": v["min_latency"],
                "max_latency": v["max_latency"],
            } for k, v in summary.items()},
            "trigger_rate": sum(
                1 for r in all_results if r["proposed_method"]["triggered"]
            ) / NUM_FRAMES * 100
        }, f, indent=2)

    print(f"\n[INFO] Raw results saved to {SAVE_RESULTS_JSON}")
    print("[INFO] Evaluation complete")


if __name__ == "__main__":
    run_evaluation()
