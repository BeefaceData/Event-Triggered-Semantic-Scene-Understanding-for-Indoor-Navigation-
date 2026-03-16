"""
Proposed Method — Event-Triggered Semantic Scene Understanding
Project: Event-Triggered Semantic Scene Understanding for Indoor Navigation
Pipeline:
    Every frame  → YOLOv8 + MiDaS (geometry)
    Trigger fire → BLIP VQA (semantics) — only when ambiguous
Decision Rule:
    Geometry handles clear frames
    BLIP handles ambiguous frames (multiple open paths OR obstacle blocking center)
"""

import cv2
import torch
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForQuestionAnswering

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on: {DEVICE}")

# YOLOv8
CONFIDENCE_THRESHOLD = 0.4
OBSTACLE_CLASSES = [
    "person", "chair", "couch", "bed",
    "dining table", "toilet", "door"
]

# MiDaS
DEPTH_REGIONS = 3

# Ambiguity Trigger thresholds
FREE_PATH_DIFF_THRESHOLD = 50.0   # If score difference between regions < this → ambiguous
MIN_FREE_PATHS           = 2       # If >= 2 paths are open → ambiguous

# BLIP
MODEL_NAME   = "Salesforce/blip-vqa-base"
VQA_QUESTION = "Which direction should I go to navigate forward, left, or right?"

DIRECTION_KEYWORDS = {
    "left":    ["left"],
    "right":   ["right"],
    "forward": ["forward", "straight", "ahead", "center"],
    "stop":    ["stop", "blocked", "back"],
}

print(f"[INFO] Running on: {DEVICE}")

# ─────────────────────────────────────────────
# 2. LOAD ALL MODELS
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

    print("[INFO] Loading BLIP-vqa-base...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    print("[INFO] All models ready")
    return yolo, midas, transform, processor, model

# ─────────────────────────────────────────────
# 3. GEOMETRY PIPELINE (runs every frame)
# ─────────────────────────────────────────────
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
    depth_map = depth.cpu().numpy()
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return depth_map, depth_norm

def detect_obstacles(frame, yolo):
    results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    obstacles = []
    for result in results:
        for box in result.boxes:
            cls_name = yolo.names[int(box.cls)]
            if cls_name in OBSTACLE_CLASSES:
                obstacles.append({
                    "class": cls_name,
                    "confidence": float(box.conf),
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

# ─────────────────────────────────────────────
# 4. AMBIGUITY TRIGGER (the novel component)
# ─────────────────────────────────────────────
def is_ambiguous(scores, obstacles):
    """
    Trigger fires when:
    - Condition A: Two or more regions have similar free space scores
                   (difference < FREE_PATH_DIFF_THRESHOLD)
    - Condition B: Center is blocked by an obstacle AND left/right are both open
    Returns: (trigger_fired: bool, reason: str)
    """
    sorted_scores = sorted(scores.values())

    # Condition A — multiple paths look equally free
    if len(sorted_scores) >= 2:
        diff = abs(sorted_scores[0] - sorted_scores[1])
        if diff < FREE_PATH_DIFF_THRESHOLD:
            return True, "multiple open paths"

    # Condition B — center blocked, both sides open
    center_blocked = any(
        (box["bbox"][0] + box["bbox"][2]) / 2 > scores.get("center", 0)
        for box in obstacles
    )
    if center_blocked:
        return True, "center blocked — side selection needed"

    return False, ""

# ─────────────────────────────────────────────
# 5. SEMANTIC PIPELINE (runs only on trigger)
# ─────────────────────────────────────────────
def semantic_decision(frame, processor, model):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, VQA_QUESTION, return_tensors="pt")
    inputs = {
        k: v.to(DEVICE).half() if v.dtype == torch.float32 else v.to(DEVICE)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(output[0], skip_special_tokens=True).lower().strip()

    for direction, keywords in DIRECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in answer:
                return direction, answer

    return None, answer

# ─────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────
def visualize(frame, depth_norm, obstacles, decision,
              scores, latency, triggered, trigger_reason, blip_answer):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Region dividers
    cv2.line(vis, (w//3, 0), (w//3, h), (255, 255, 0), 2)
    cv2.line(vis, (2*w//3, 0), (2*w//3, h), (255, 255, 0), 2)

    # Obstacle boxes
    for obs in obstacles:
        x1, y1, x2, y2 = map(int, obs["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, obs["class"], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Decision
    cv2.putText(vis, f"GO: {decision.upper() if decision else 'NONE'}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(vis, f"Latency: {latency:.1f}ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Trigger status
    if triggered:
        trigger_label = f"TRIGGER: ON — {trigger_reason}"
        trigger_color = (0, 165, 255)   # orange
        if blip_answer:
            cv2.putText(vis, f"BLIP: '{blip_answer}'", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        trigger_label = "TRIGGER: OFF — geometry only"
        trigger_color = (200, 200, 200)

    cv2.putText(vis, trigger_label, (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, trigger_color, 2)

    # Region scores
    for i, (region, score) in enumerate(scores.items()):
        cv2.putText(vis, f"{region}: {score:.1f}", (20 + i * 160, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return vis

# ─────────────────────────────────────────────
# 7. EVALUATION LOGGER
# ─────────────────────────────────────────────
class EvaluationLogger:
    def __init__(self):
        self.latencies         = []
        self.decisions         = []
        self.triggered_frames  = 0
        self.total_frames      = 0
        self.geo_latencies     = []   # latency when trigger OFF
        self.sem_latencies     = []   # latency when trigger ON

    def log(self, latency, decision, triggered):
        self.latencies.append(latency)
        self.decisions.append(decision)
        self.total_frames += 1
        if triggered:
            self.triggered_frames += 1
            self.sem_latencies.append(latency)
        else:
            self.geo_latencies.append(latency)

    def summary(self):
        total     = self.total_frames
        triggered = self.triggered_frames
        pct       = triggered / total * 100 if total else 0

        print("\n" + "="*40)
        print("PROPOSED METHOD — EVALUATION SUMMARY")
        print("="*40)
        print(f"Total frames              : {total}")
        print(f"Average latency           : {np.mean(self.latencies):.2f} ms")
        print(f"Geometry-only avg latency : {np.mean(self.geo_latencies):.2f} ms" if self.geo_latencies else "")
        print(f"Semantic-trigger avg lat  : {np.mean(self.sem_latencies):.2f} ms" if self.sem_latencies else "")
        print(f"Triggered frames          : {triggered} / {total} ({pct:.1f}%)")
        print(f"Geometry-only frames      : {total - triggered} / {total} ({100-pct:.1f}%)")
        print(f"Decision distribution     :")
        for d in list(DIRECTION_KEYWORDS.keys()) + [None]:
            count = self.decisions.count(d)
            label = d if d else "none"
            pct_d = count / total * 100 if total else 0
            print(f"  {label:<10}: {count} ({pct_d:.1f}%)")
        print("="*40)

# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_proposed(source=0):
    """
    source: 0 = webcam, or path to video file
    """
    yolo, midas, transform, processor, blip = load_models()
    logger = EvaluationLogger()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    print("[INFO] Running Proposed Method — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # ── Stage 1: Geometry (always runs) ──
        obstacles           = detect_obstacles(frame, yolo)
        depth_map, depth_norm = estimate_depth(frame, midas, transform)
        geo_dir, scores     = geometry_decision(depth_map, obstacles)

        # ── Stage 2: Ambiguity Trigger ──
        triggered, reason   = is_ambiguous(scores, obstacles)

        # ── Stage 3: Semantic (only if triggered) ──
        if triggered:
            sem_dir, blip_answer = semantic_decision(frame, processor, blip)
            final_decision       = sem_dir if sem_dir else geo_dir
        else:
            final_decision = geo_dir
            blip_answer    = None

        latency = (time.time() - start) * 1000
        logger.log(latency, final_decision, triggered)

        # ── Visualization ──
        vis = visualize(frame, depth_norm, obstacles, final_decision,
                        scores, latency, triggered, reason, blip_answer)

        cv2.imshow("Proposed Method — Event-Triggered", vis)

        tag = f"[TRIGGER]" if triggered else "[GEOMETRY]"
        print(f"{tag} Decision: {final_decision:<10} | "
              f"Latency: {latency:.1f}ms | Trigger: {triggered}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.summary()


if __name__ == "__main__":
    # run_proposed("path/to/video.mp4")
    run_proposed(0)   # 0 = webcam
