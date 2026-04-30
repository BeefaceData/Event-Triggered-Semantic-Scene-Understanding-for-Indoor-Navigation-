"""Compare legacy and uncertainty-based trigger behavior on HM3D scenes."""

import argparse
import json
import numpy as np

from hm3d_dataset import (
    resolve_annotated_config,
    resolve_dataset_root,
    split_trigger_comparison_json,
)
from evaluate_hm3d import find_scenes, render_frames_from_scene
from models import load_models
from pipeline import process_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trigger modes on an HM3D split.")
    parser.add_argument(
        "--split",
        default="example",
        choices=["example", "minival", "val", "train"],
        help="HM3D split to evaluate.",
    )
    parser.add_argument("--dataset-root", default=None, help="Optional explicit path to the HM3D split root.")
    parser.add_argument(
        "--annotated-config",
        default=None,
        help="Optional explicit path to a semantic scene dataset config JSON.",
    )
    parser.add_argument(
        "--frames-per-scene",
        type=int,
        default=30,
        help="Number of rendered frames per scene.",
    )
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap on the number of scenes evaluated.")
    return parser.parse_args()


def summarize(records):
    triggered = sum(1 for record in records if record["trigger"]["triggered"])
    latencies = [record["proposed"]["latency_ms"] for record in records]
    decisions = [record["proposed"]["decision"] for record in records]
    return {
        "num_frames": len(records),
        "trigger_rate": triggered / max(len(records), 1) * 100.0,
        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
        "max_latency": float(np.max(latencies)) if latencies else 0.0,
        "decisions": {str(decision): decisions.count(decision) for decision in set(decisions)},
    }


def run_mode(models, mode, dataset_root, annotated_config, frames_per_scene, max_scenes):
    records = []
    for scene_id, scene_path in find_scenes(dataset_root, max_scenes=max_scenes):
        frames, _ = render_frames_from_scene(
            scene_path,
            scene_id,
            dataset_root,
            annotated_config,
            frames_per_scene,
        )
        for frame in frames:
            records.append(process_frame(frame, models, trigger_mode=mode))
    return records


def main():
    args = parse_args()
    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(dataset_root, args.split, args.annotated_config)
    output_path = split_trigger_comparison_json(args.split)
    models = load_models()
    legacy_records = run_mode(
        models,
        "legacy",
        dataset_root,
        annotated_config,
        args.frames_per_scene,
        args.max_scenes,
    )
    uncertainty_records = run_mode(
        models,
        "uncertainty",
        dataset_root,
        annotated_config,
        args.frames_per_scene,
        args.max_scenes,
    )

    output = {
        "dataset_root": str(dataset_root),
        "legacy": summarize(legacy_records),
        "uncertainty": summarize(uncertainty_records),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[INFO] Trigger comparison saved to {output_path}")


if __name__ == "__main__":
    main()
