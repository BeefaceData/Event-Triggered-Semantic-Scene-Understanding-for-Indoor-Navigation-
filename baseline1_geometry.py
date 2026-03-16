"""
Baseline 1 — Geometry-Only Indoor Navigation
Project: Event-Triggered Semantic Scene Understanding for Indoor Navigation
Pipeline: YOLOv8 (obstacle detection) + MiDaS (depth estimation)
Decision Rule: Choose direction with largest free space
"""

import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on: {DEVICE}")

CONFIDENCE_THRESHOLD = 0.4      # YOLOv8 detection confidence
DEPTH_REGIONS = 3               # Split frame into left / center / right
OBSTACLE_CLASSES = [            # COCO classes treated as obstacles
    "person", "chair", "couch", "bed",
    "dining table", "toilet", "door", "wall"
]

# ─────────────────────────────────────────────
# 2. LOAD MODELS
# ─────────────────────────────────────────────
def load_models():
    print("[INFO] Loading YOLOv8...")
    yolo = YOLO("yolov8n.pt")   # nano — fastest, swap to yolov8s.pt for better accuracy
    yolo.to(DEVICE)

    print("[INFO] Loading MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # small = fastest
    midas.to(DEVICE)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return yolo, midas, transform

# ─────────────────────────────────────────────
# 3. DEPTH ESTIMATION
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
    # Normalize to 0-255 for visualization
    depth_normalized = cv2.normalize(
        depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    return depth_map, depth_normalized

# ─────────────────────────────────────────────
# 4. FREE SPACE ESTIMATION
# ─────────────────────────────────────────────
def estimate_free_space(depth_map):
    """
    Split frame into 3 horizontal regions: left / center / right
    Higher mean depth value = closer obstacle (MiDaS inverse depth)
    Lower mean depth value = more free space
    Returns: region scores and recommended direction
    """
    h, w = depth_map.shape
    regions = {
        "left":   depth_map[:, :w//3],
        "center": depth_map[:, w//3:2*w//3],
        "right":  depth_map[:, 2*w//3:]
    }
    # Lower score = more free space
    scores = {k: np.mean(v) for k, v in regions.items()}
    decision = min(scores, key=scores.get)
    return scores, decision

# ─────────────────────────────────────────────
# 5. OBSTACLE DETECTION
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 6. NAVIGATION DECISION
# ─────────────────────────────────────────────
def navigation_decision(depth_map, obstacles):
    scores, direction = estimate_free_space(depth_map)

    # If obstacle directly ahead, override with least-blocked side
    center_blocked = any(
        (box["bbox"][0] + box["bbox"][2]) / 2 > depth_map.shape[1] // 3 and
        (box["bbox"][0] + box["bbox"][2]) / 2 < 2 * depth_map.shape[1] // 3
        for box in obstacles
    )

    if center_blocked and direction == "center":
        # Choose between left and right
        direction = "left" if scores["left"] < scores["right"] else "right"

    return direction, scores

# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────
def visualize(frame, depth_normalized, obstacles, direction, scores, latency):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw region boundaries
    cv2.line(vis, (w//3, 0), (w//3, h), (255, 255, 0), 2)
    cv2.line(vis, (2*w//3, 0), (2*w//3, h), (255, 255, 0), 2)

    # Draw obstacle boxes
    for obs in obstacles:
        x1, y1, x2, y2 = map(int, obs["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, obs["class"], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw navigation decision
    cv2.putText(vis, f"GO: {direction.upper()}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(vis, f"Latency: {latency:.1f}ms", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Score overlay
    for i, (region, score) in enumerate(scores.items()):
        cv2.putText(vis, f"{region}: {score:.1f}", (20 + i*150, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return vis, depth_normalized

# ─────────────────────────────────────────────
# 8. EVALUATION LOGGER
# ─────────────────────────────────────────────
class EvaluationLogger:
    def __init__(self):
        self.latencies = []
        self.decisions = []
        self.obstacle_counts = []

    def log(self, latency, decision, obstacle_count):
        self.latencies.append(latency)
        self.decisions.append(decision)
        self.obstacle_counts.append(obstacle_count)

    def summary(self):
        print("\n" + "="*40)
        print("BASELINE 1 — EVALUATION SUMMARY")
        print("="*40)
        print(f"Total frames processed : {len(self.latencies)}")
        print(f"Average latency        : {np.mean(self.latencies):.2f} ms")
        print(f"Min latency            : {np.min(self.latencies):.2f} ms")
        print(f"Max latency            : {np.max(self.latencies):.2f} ms")
        print(f"Decision distribution  :")
        for d in ["left", "center", "right"]:
            count = self.decisions.count(d)
            pct = count / len(self.decisions) * 100 if self.decisions else 0
            print(f"  {d:<10}: {count} ({pct:.1f}%)")
        print("="*40)

# ─────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_baseline(source=0):
    """
    source: 0 = webcam, or pass a video/image path
    Example: run_baseline("path/to/indoor_video.mp4")
    """
    yolo, midas, transform = load_models()
    logger = EvaluationLogger()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    print("[INFO] Running Baseline 1 — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # --- Core pipeline ---
        obstacles               = detect_obstacles(frame, yolo)
        depth_map, depth_norm   = estimate_depth(frame, midas, transform)
        direction, scores       = navigation_decision(depth_map, obstacles)

        latency = (time.time() - start) * 1000   # ms
        logger.log(latency, direction, len(obstacles))

        # --- Visualization ---
        vis, depth_vis = visualize(frame, depth_norm, obstacles,
                                   direction, scores, latency)

        cv2.imshow("Baseline 1 — RGB + Detection", vis)
        cv2.imshow("Baseline 1 — Depth Map", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.summary()


if __name__ == "__main__":
    # Replace with your video path or image folder
    # run_baseline("path/to/video.mp4")
    run_baseline(0)   # 0 = webcam for quick test
