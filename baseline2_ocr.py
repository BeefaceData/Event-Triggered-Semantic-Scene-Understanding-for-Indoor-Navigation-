"""
Baseline 2 — Always-On Semantic Navigation (OCR)
Project: Event-Triggered Semantic Scene Understanding for Indoor Navigation
Pipeline: EasyOCR on every frame → read sign → choose direction
Decision Rule: If directional keyword found → follow it, else → no decision
"""

import cv2
import torch
import numpy as np
import time
import easyocr

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OCR_LANGUAGE = ["en"]
MIN_CONFIDENCE = 0.4          # Minimum OCR confidence to accept text

# Keywords that map to navigation decisions
DIRECTION_KEYWORDS = {
    "left":     ["left", "←", "<", "lft"],
    "right":    ["right", "→", ">", "rgt", "=>", "->"],
    "forward":  ["forward", "ahead", "straight", "↑", "up"],
    "exit":     ["exit", "way out", "out"],
    "stop":     ["stop", "halt", "danger", "warning"],
}

print(f"[INFO] Running on: {DEVICE}")

# ─────────────────────────────────────────────
# 2. LOAD OCR MODEL
# ─────────────────────────────────────────────
def load_ocr():
    print("[INFO] Loading EasyOCR...")
    reader = easyocr.Reader(OCR_LANGUAGE, gpu=(DEVICE == "cuda"))
    print("[INFO] EasyOCR ready")
    return reader

# ─────────────────────────────────────────────
# 3. RUN OCR ON FRAME
# ─────────────────────────────────────────────
def run_ocr(frame, reader):
    """
    Returns list of detected text results:
    Each result: (bbox, text, confidence)
    """
    results = reader.readtext(frame)
    filtered = [
        (bbox, text.lower().strip(), conf)
        for (bbox, text, conf) in results
        if conf >= MIN_CONFIDENCE
    ]
    return filtered

# ─────────────────────────────────────────────
# 4. PARSE NAVIGATION DECISION FROM TEXT
# ─────────────────────────────────────────────
def parse_direction(ocr_results):
    """
    Scan all detected text for directional keywords.
    Returns: (decision, matched_text) or (None, None)
    """
    for (bbox, text, conf) in ocr_results:
        for direction, keywords in DIRECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return direction, text, conf
    return None, None, None

# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────
def visualize(frame, ocr_results, decision, matched_text, latency):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw all OCR detections
    for (bbox, text, conf) in ocr_results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        x, y = pts[0]
        cv2.putText(vis, f"{text} ({conf:.2f})", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw navigation decision
    if decision:
        label = f"GO: {decision.upper()} ('{matched_text}')"
        color = (0, 255, 0)
    else:
        label = "NO SIGN DETECTED"
        color = (0, 0, 255)

    cv2.putText(vis, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(vis, f"Latency: {latency:.1f}ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Texts found: {len(ocr_results)}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    return vis

# ─────────────────────────────────────────────
# 6. EVALUATION LOGGER
# ─────────────────────────────────────────────
class EvaluationLogger:
    def __init__(self):
        self.latencies      = []
        self.decisions      = []
        self.text_counts    = []
        self.no_sign_frames = 0

    def log(self, latency, decision, text_count):
        self.latencies.append(latency)
        self.decisions.append(decision)
        self.text_counts.append(text_count)
        if decision is None:
            self.no_sign_frames += 1

    def summary(self):
        total = len(self.latencies)
        print("\n" + "="*40)
        print("BASELINE 2 — EVALUATION SUMMARY")
        print("="*40)
        print(f"Total frames processed   : {total}")
        print(f"Average latency          : {np.mean(self.latencies):.2f} ms")
        print(f"Min latency              : {np.min(self.latencies):.2f} ms")
        print(f"Max latency              : {np.max(self.latencies):.2f} ms")
        print(f"Frames with no sign      : {self.no_sign_frames} ({self.no_sign_frames/total*100:.1f}%)")
        print(f"Frames with decision     : {total - self.no_sign_frames} ({(total-self.no_sign_frames)/total*100:.1f}%)")
        print(f"Decision distribution    :")
        for d in list(DIRECTION_KEYWORDS.keys()) + [None]:
            count = self.decisions.count(d)
            label = d if d else "none"
            pct   = count / total * 100 if total else 0
            print(f"  {label:<10}: {count} ({pct:.1f}%)")
        print("="*40)

# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_baseline2(source=0):
    """
    source: 0 = webcam, or pass a video/image path
    Example: run_baseline2("path/to/indoor_video.mp4")

    NOTE: OCR runs on EVERY frame — this is the always-on semantic baseline.
    High latency is expected and intentional — that is the point of comparison.
    """
    reader = load_ocr()
    logger = EvaluationLogger()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    print("[INFO] Running Baseline 2 (Always-On OCR) — press Q to quit")
    print("[INFO] Point camera at indoor signs, text, room numbers, arrows")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # --- Core pipeline ---
        ocr_results              = run_ocr(frame, reader)
        decision, matched, conf  = parse_direction(ocr_results)

        latency = (time.time() - start) * 1000
        logger.log(latency, decision, len(ocr_results))

        # --- Visualization ---
        vis = visualize(frame, ocr_results, decision, matched, latency)
        cv2.imshow("Baseline 2 — Always-On OCR", vis)

        # Print to terminal for logging
        if decision:
            print(f"[FRAME] Decision: {decision:<10} | Text: '{matched}' "
                  f"| Conf: {conf:.2f} | Latency: {latency:.1f}ms")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.summary()


if __name__ == "__main__":
    # Replace with your video path to test on indoor scenes with signs
    # run_baseline2("path/to/video.mp4")
    run_baseline2(0)   # 0 = webcam
