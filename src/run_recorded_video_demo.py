"""Process a recorded video into an annotated event-triggered demo artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import cv2

from audio_narration import generate_spoken_audio
from config import DEMO_OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH, SEMANTIC_BACKEND_OPTIONS
from evaluate_hm3d import ensure_dir
from models import load_models
from pipeline import process_frame
from visualization import annotate_navigation_frame, stack_rgb_and_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Run the event-triggered pipeline on a recorded video.")
    parser.add_argument("--video", required=True, help="Path to the recorded input video.")
    parser.add_argument(
        "--trigger-mode",
        default="uncertainty",
        choices=["uncertainty", "legacy"],
        help="Trigger formulation to use.",
    )
    parser.add_argument(
        "--semantic-policy",
        default="event_triggered",
        choices=["geometry_only", "always_semantic", "event_triggered"],
        help="Semantic invocation policy for the recorded demo.",
    )
    parser.add_argument(
        "--semantic-backend",
        default=None,
        choices=SEMANTIC_BACKEND_OPTIONS,
        help="Semantic backend to use when the policy invokes semantics.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional custom stem for output files.",
    )
    parser.add_argument(
        "--add-audio",
        action="store_true",
        help="Add spoken narration to the saved output video.",
    )
    parser.add_argument(
        "--audio-mode",
        default="direction",
        choices=["direction", "action"],
        help="Narration mode passed to audio generation. 'direction' is silent on routine clear-path motion.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the processed video while saving it.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Process every Nth frame from the recorded video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of processed frames.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N processed frames.",
    )
    return parser.parse_args()


def make_writer(path: Path, width: int, height: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def safe_depth_vis(record: dict, height: int, width: int):
    """Return a usable depth visualization, even if the record is missing one."""
    vis = record.get("depth_visualization")
    if vis is not None:
        return vis

    blank = 64 * np.ones((height, width, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        "depth N/A",
        (10, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    return blank


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise RuntimeError(f"Input video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open recorded video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0
    input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_fps = max(fps / max(args.sample_every, 1), 1.0)

    ensure_dir(DEMO_OUTPUT_DIR)
    if args.output_stem is not None:
        output_stem = args.output_stem
    else:
        output_stem = f"{video_path.stem}_{args.semantic_policy}"
        if args.semantic_policy in {"event_triggered", "always_semantic"}:
            backend_suffix = args.semantic_backend or "default"
            output_stem = f"{output_stem}_{backend_suffix}"
    output_video_path = DEMO_OUTPUT_DIR / f"{output_stem}.mp4"
    silent_video_path = DEMO_OUTPUT_DIR / f"{output_stem}_silent.mp4" if args.add_audio else output_video_path
    output_json_path = DEMO_OUTPUT_DIR / f"{output_stem}.json"

    models = load_models(semantic_backend=args.semantic_backend)
    writer = None
    records = []
    frame_index = 0
    processed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % max(args.sample_every, 1) != 0:
            frame_index += 1
            continue

        if args.max_frames is not None and processed_frames >= args.max_frames:
            break

        record = process_frame(
            frame,
            models,
            trigger_mode=args.trigger_mode,
            semantic_policy=args.semantic_policy,
        )
        annotated = annotate_navigation_frame(frame, record)
        combo = stack_rgb_and_depth(
            annotated,
            safe_depth_vis(record, IMG_HEIGHT, IMG_WIDTH),
        )

        if writer is None:
            height, width = combo.shape[:2]
            writer = make_writer(silent_video_path, width, height, output_fps)

        writer.write(combo)

        if args.display:
            cv2.imshow("Recorded Event-Triggered Demo", combo)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        records.append(
            {
                "frame_index": frame_index,
                "processed_index": processed_frames,
                "time_s": frame_index / fps,
                "triggered": bool(record["trigger"]["triggered"]),
                "trigger_reasons": list(record["trigger"]["reasons"]),
                "semantic_policy": args.semantic_policy,
                "semantic_invoked": bool(record.get("semantic_invoked", False)),
                "semantic_backend": record.get("semantic_backend"),
                "final_direction": record["proposed"]["decision"],
                "semantic_direction": record["proposed"]["semantic_decision"],
                "geometry_direction": record["baseline_geometry"]["decision"],
                "ocr_direction": record["baseline_ocr"]["decision"],
                "ocr_text": record["baseline_ocr"]["ocr_text"],
                "signage_hits": record["baseline_ocr"]["signage_hits"],
                "guidance": record.get("guidance", {}),
                "scores": record["scores"],
                "center_blocked": bool(record["center_blocked"]),
                "entropy": float(record["trigger"]["entropy"]),
                "relative_separability": float(record["trigger"]["relative_separability"]),
                "backbone_latency_ms": float(record["backbone_latency_ms"]),
                "proposed_latency_ms": float(record["proposed"]["latency_ms"]),
            }
        )
        processed_frames += 1
        if processed_frames % max(args.progress_every, 1) == 0:
            if input_frame_count > 0:
                print(
                    f"[INFO] Processed {processed_frames} sampled frames "
                    f"(source frame {frame_index}/{input_frame_count})"
                )
            else:
                print(f"[INFO] Processed {processed_frames} sampled frames (source frame {frame_index})")
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    narrated_video_path = None
    if args.add_audio:
        narrated_video_path = generate_spoken_audio(
            video_path=silent_video_path,
            records=records,
            fps=max(int(round(fps)), 1),
            mode=args.audio_mode,
            output_video_path=output_video_path,
        )

    output = {
        "input_video_path": str(video_path),
        "semantic_policy": args.semantic_policy,
        "semantic_backend": models["semantic_backend"],
        "trigger_mode": args.trigger_mode,
        "fps": fps,
        "output_fps": output_fps,
        "sample_every": args.sample_every,
        "input_num_frames": input_frame_count,
        "num_frames": len(records),
        "output_video_path": str(output_video_path),
        "silent_video_path": str(silent_video_path) if args.add_audio else None,
        "video_with_audio_path": str(narrated_video_path) if narrated_video_path is not None else None,
        "num_triggered_frames": sum(1 for record in records if record["triggered"]),
        "num_semantic_calls": sum(1 for record in records if record["semantic_invoked"]),
        "records": records,
    }
    output_json_path.write_text(json.dumps(output, indent=2))

    trigger_rate = (output["num_triggered_frames"] / max(output["num_frames"], 1)) * 100.0
    semantic_rate = (output["num_semantic_calls"] / max(output["num_frames"], 1)) * 100.0
    guidance_counter = Counter(
        record.get("guidance", {}).get("text")
        for record in records
        if record.get("guidance", {}).get("text")
    )

    print(f"[INFO] Recorded demo video saved to {output_video_path}")
    print(f"[INFO] Recorded demo JSON saved to {output_json_path}")
    print(f"[INFO] Trigger rate: {trigger_rate:.2f}%")
    print(f"[INFO] Semantic call rate: {semantic_rate:.2f}%")
    if guidance_counter:
        top_guidance, top_count = guidance_counter.most_common(1)[0]
        print(f"[INFO] Most common guidance: {top_guidance} ({top_count} frames)")


if __name__ == "__main__":
    main()
