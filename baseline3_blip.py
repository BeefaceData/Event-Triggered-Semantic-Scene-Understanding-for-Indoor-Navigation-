"""
Baseline 3 — Always-On Semantic Navigation (BLIP VQA)
Project: Event-Triggered Semantic Scene Understanding for Indoor Navigation
Pipeline: BLIP-vqa-base on every frame → answer directional question → decide
Decision Rule: Ask "which direction should I go?" → parse answer
Hardware: Designed for 4GB VRAM — uses blip-vqa-base (~400MB)
"""

import cv2
import torch
import numpy as np
import time
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip-vqa-base"

# Question asked to BLIP on every frame
VQA_QUESTION = "Which direction should I go to navigate forward, left, or right?"

# Keywords to parse BLIP answer into a navigation decision
DIRECTION_KEYWORDS = {
    "left":    ["left"],
    "right":   ["right"],
    "forward": ["forward", "straight", "ahead", "center"],
    "stop":    ["stop", "blocked", "back"],
}

print(f"[INFO] Running on: {DEVICE}")

# ─────────────────────────────────────────────
# 2. LOAD BLIP MODEL
# ─────────────────────────────────────────────
def load_blip():
    print("[INFO] Loading BLIP-vqa-base (~400MB, fits in 4GB VRAM)...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForQuestionAnswering.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16   # float16 halves memory usage
    ).to(DEVICE)
    model.eval()
    print("[INFO] BLIP ready")
    return processor, model

# ─────────────────────────────────────────────
# 3. RUN BLIP VQA ON FRAME
# ─────────────────────────────────────────────
def run_blip(frame, processor, model, question=VQA_QUESTION):
    """
    Convert frame: PIL Image: run BLIP VQA
    Returns: raw answer string from BLIP
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs    = processor(pil_image, question, return_tensors="pt")

    # Move inputs to device with float16
    inputs = {
        k: v.to(DEVICE).half() if v.dtype == torch.float32 else v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.lower().strip()

# ─────────────────────────────────────────────
# 4. PARSE NAVIGATION DECISION FROM BLIP ANSWER
# ─────────────────────────────────────────────
def parse_decision(answer):
    """
    Map BLIP's free-text answer to a navigation direction.
    Returns: decision string or None
    """
    for direction, keywords in DIRECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in answer:
                return direction
    return None

# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────
def visualize(frame, answer, decision, latency):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Navigation decision
    if decision:
        label = f"GO: {decision.upper()}"
        color = (0, 255, 0)
    else:
        label = "NO DECISION"
        color = (0, 0, 255)

    cv2.putText(vis, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(vis, f"Latency: {latency:.1f}ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # BLIP raw answer
    cv2.putText(vis, f"BLIP: '{answer}'", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Always-on label
    cv2.putText(vis, "SEMANTIC: ALWAYS ON", (w - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    return vis

# ─────────────────────────────────────────────
# 6. EVALUATION LOGGER
# ─────────────────────────────────────────────
class EvaluationLogger:
    def __init__(self):
        self.latencies       = []
        self.decisions       = []
        self.raw_answers     = []
        self.no_dec_frames   = 0

    def log(self, latency, decision, answer):
        self.latencies.append(latency)
        self.decisions.append(decision)
        self.raw_answers.append(answer)
        if decision is None:
            self.no_dec_frames += 1

    def summary(self):
        total = len(self.latencies)
        print("\n" + "="*40)
        print("BASELINE 3 — EVALUATION SUMMARY")
        print("="*40)
        print(f"Total frames processed   : {total}")
        print(f"Average latency          : {np.mean(self.latencies):.2f} ms")
        print(f"Min latency              : {np.min(self.latencies):.2f} ms")
        print(f"Max latency              : {np.max(self.latencies):.2f} ms")
        print(f"Frames with no decision  : {self.no_dec_frames} ({self.no_dec_frames/total*100:.1f}%)")
        print(f"Semantic calls           : {total} (100% — always on)")
        print(f"Decision distribution    :")
        for d in list(DIRECTION_KEYWORDS.keys()) + [None]:
            count = self.decisions.count(d)
            label = d if d else "none"
            pct   = count / total * 100 if total else 0
            print(f"  {label:<10}: {count} ({pct:.1f}%)")
        print("\nSample BLIP answers:")
        for ans in list(set(self.raw_answers))[:5]:
            print(f"  → '{ans}'")
        print("="*40)

# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_baseline3(source=0):
    """
    source: 0 = webcam, or pass a video/image path
    Example: run_baseline3("path/to/indoor_video.mp4")

    NOTE: BLIP runs on EVERY frame — always-on semantic baseline.
    High latency is expected and intentional.
    """
    processor, model = load_blip()
    logger           = EvaluationLogger()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    print("[INFO] Running Baseline 3 (Always-On BLIP) — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # --- Core pipeline ---
        answer   = run_blip(frame, processor, model)
        decision = parse_decision(answer)

        latency = (time.time() - start) * 1000
        logger.log(latency, decision, answer)

        # --- Visualization ---
        vis = visualize(frame, answer, decision, latency)
        cv2.imshow("Baseline 3 — Always-On BLIP", vis)

        print(f"[FRAME] BLIP: '{answer}' → {decision} | {latency:.1f}ms")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.summary()


if __name__ == "__main__":
    # run_baseline3("path/to/video.mp4")
    run_baseline3(0)   # 0 = webcam
