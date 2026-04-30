"""Run the event-triggered pipeline on webcam or video."""

import argparse
import cv2

from config import DEMO_OUTPUT_DIR
from models import load_models
from pipeline import process_frame
from visualization import annotate_navigation_frame, stack_rgb_and_depth
from config import SEMANTIC_BACKEND_OPTIONS


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam or a path to a video")
    parser.add_argument("--save", action="store_true", help="Save the demo output video")
    parser.add_argument(
        "--trigger-mode",
        default="uncertainty",
        choices=["uncertainty", "legacy"],
        help="Choose which trigger formulation to use",
    )
    parser.add_argument(
        "--semantic-backend",
        default=None,
        choices=SEMANTIC_BACKEND_OPTIONS,
        help="Semantic backend to use in the pipeline.",
    )
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    models = load_models(semantic_backend=args.semantic_backend)
    print("[INFO] Running live pipeline. Press q to quit.")
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        record = process_frame(frame, models, trigger_mode=args.trigger_mode)
        vis = annotate_navigation_frame(frame, record)
        combo = stack_rgb_and_depth(vis, record["depth_visualization"])

        if args.save and writer is None:
            ensure_dir(DEMO_OUTPUT_DIR)
            height, width = combo.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(DEMO_OUTPUT_DIR / f"demo_{args.trigger_mode}.mp4"),
                fourcc,
                10,
                (width, height),
            )

        if writer is not None:
            writer.write(combo)

        cv2.imshow("Event-Triggered Indoor Navigation", combo)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
