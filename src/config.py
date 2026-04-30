"""Central configuration for the event-triggered navigation project."""

from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = PROJECT_ROOT
OUTPUT_ROOT = MODULE_ROOT / "outputs"
HM3D_REVIEW_DIR = OUTPUT_ROOT / "hm3d_review"
HM3D_RESULTS_JSON = OUTPUT_ROOT / "results_hm3d.json"
HM3D_MANUAL_REVIEW_CSV = OUTPUT_ROOT / "hm3d_manual_review.csv"
TRIGGER_COMPARISON_JSON = OUTPUT_ROOT / "trigger_comparison_hm3d.json"
DEMO_OUTPUT_DIR = OUTPUT_ROOT / "demo"
PLOT_OUTPUT_DIR = OUTPUT_ROOT / "plots"

DATASET_ROOT = PROJECT_ROOT / "datasets" / "hm3d" / "example"
HM3D_ANNOTATED_CONFIG = DATASET_ROOT / "hm3d_annotated_example_basis.scene_dataset_config.json"

YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_WIDTH = 640
IMG_HEIGHT = 480
FRAMES_PER_SCENE = 30
SAMPLES_PER_SCENE = 5

CONFIDENCE_THRESHOLD = 0.4
OCR_CONFIDENCE_THRESHOLD = 0.4
USE_OBSTACLE_FUSION = True

# Main uncertainty-trigger thresholds. These are intentionally explicit so that
# threshold sweeps can be run later without changing the rest of the implementation.
SEPARABILITY_THRESHOLD = 0.08
ENTROPY_THRESHOLD = 1.03

# Legacy trigger baseline only. This is not used by the main uncertainty trigger
# and is kept solely for comparison experiments against the older heuristic rule.
LEGACY_FREE_PATH_DIFF_THRESHOLD = 50.0

OBSTACLE_CLASSES = [
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "door",
]

DIRECTION_KEYWORDS = {
    "left": ["left"],
    "right": ["right"],
    "center": ["center", "straight", "ahead", "forward"],
    "stop": ["stop", "blocked", "back"],
}

TEXT_TRIGGER_KEYWORDS = [
    "left",
    "right",
    "straight",
    "ahead",
    "forward",
    "exit",
    "entrance",
    "room",
]

SIGNAGE_KEYWORDS = [
    "stairs",
    "stair",
    "exit",
    "elevator",
    "lift",
    "room",
    "entrance",
    "floor",
]

BLIP_MODEL_NAME = "Salesforce/blip-vqa-base"
BLIP_QUESTION = (
    "You are helping with indoor navigation. "
    "Answer with one direction: left, center, or right. "
    "Which direction is best to continue safely?"
)

SEMANTIC_BACKEND = "smolvlm"
SEMANTIC_BACKEND_OPTIONS = ["blip", "smolvlm"]
SMOLVLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Instruct"
SMOLVLM_QUESTION = (
    "You are helping with indoor navigation. "
    "Answer with one word only: left, center, or right. "
    "Which direction is best to continue safely?"
)
SMOLVLM_MAX_IMAGE_EDGE = 336
