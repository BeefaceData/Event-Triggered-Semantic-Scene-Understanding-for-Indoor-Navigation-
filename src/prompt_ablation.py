"""Ablate BLIP prompt phrasing on triggered HM3D frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from config import (
    BLIP_MODEL_NAME,
    BLIP_QUESTION,
    DEVICE,
    DIRECTION_KEYWORDS,
    FRAMES_PER_SCENE,
)
from evaluate_hm3d import ensure_dir, find_scenes, render_frames_from_scene
from geometry import (
    compute_region_scores,
    detect_obstacles,
    estimate_depth,
    fuse_depth_with_obstacles,
    is_center_blocked,
    normalize_depth,
)
from hm3d_dataset import resolve_annotated_config, resolve_dataset_root, split_output_dir
from models import load_models
from trigger import evaluate_trigger, relative_separability, uncertainty_entropy

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency at runtime
    plt = None


PROMPT_VARIANTS = {
    "V1_current": BLIP_QUESTION,
    "V2_exact_word": (
        "Answer with exactly one word: left, center, or right. "
        "Which direction should the agent take?"
    ),
    "V3_exact_navigable": (
        "Choose exactly one option: left, center, or right. "
        "Which region is most navigable?"
    ),
    "V4_exact_safe": (
        "Respond with only one word: left, center, or right. "
        "Which direction is safest to continue?"
    ),
}

CURRENT_PROMPT_KEY = "V1_current"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test BLIP prompt sensitivity on triggered HM3D frames."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["example", "minival", "val", "train"],
        help="HM3D split to use.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional explicit HM3D split root.",
    )
    parser.add_argument(
        "--annotated-config",
        default=None,
        help="Optional explicit semantic scene dataset config path.",
    )
    parser.add_argument(
        "--frames-per-scene",
        type=int,
        default=FRAMES_PER_SCENE,
        help="Number of rendered frames per scene.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=5,
        help="Keep small: BLIP is run once per prompt on every triggered frame.",
    )
    parser.add_argument(
        "--sampling-mode",
        default="trajectory",
        choices=["random", "trajectory"],
        help="Scene coverage strategy, aligned with evaluate_hm3d.py.",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=15,
        help="Number of steps per trajectory when using trajectory sampling.",
    )
    return parser.parse_args()


def run_blip_with_question(frame: np.ndarray, processor, blip, question: str) -> tuple[str | None, str]:
    """Run BLIP VQA on one frame with a specific question, matching project parsing."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, question, return_tensors="pt")
    inputs = {
        key: value.to(DEVICE).half() if value.dtype == torch.float32 and DEVICE == "cuda" else value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = blip.generate(**inputs, max_new_tokens=10)

    answer = processor.decode(output[0], skip_special_tokens=True).lower().strip()
    for direction, keywords in DIRECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in answer:
                return direction, answer
    return None, answer


def should_trigger(frame: np.ndarray, models: dict) -> tuple[bool, dict]:
    """Run the geometry + uncertainty trigger path and return a compact record."""
    yolo = models["yolo"]
    midas = models["midas"]
    transform = models["transform"]
    ocr = models["ocr"]

    obstacles = detect_obstacles(frame, yolo)
    raw_depth = estimate_depth(frame, midas, transform)
    normalized_depth = normalize_depth(raw_depth)
    fused_depth = fuse_depth_with_obstacles(normalized_depth, obstacles)
    scores = compute_region_scores(fused_depth)
    center_blocked = is_center_blocked(obstacles, frame.shape[1])

    trigger_record = evaluate_trigger(
        scores,
        center_blocked,
        frame,
        ocr,
        mode="uncertainty",
    )

    return trigger_record["triggered"], {
        "scores": scores,
        "delta_norm": float(relative_separability(scores)),
        "entropy": float(uncertainty_entropy(scores)),
        "center_blocked": bool(center_blocked),
        "reasons": list(trigger_record["reasons"]),
    }


def compute_pairwise_agreement(frame_logs: list[dict], prompt_keys: list[str]) -> dict:
    """Compute pairwise agreement on parsed navigation decisions."""
    agreements = {}
    for index, key_a in enumerate(prompt_keys):
        for key_b in prompt_keys[index + 1 :]:
            match_count = 0
            for frame_log in frame_logs:
                if frame_log["decisions"][key_a] == frame_log["decisions"][key_b]:
                    match_count += 1
            total = len(frame_logs)
            agreements[f"{key_a}_vs_{key_b}"] = {
                "agreement_rate": match_count / total if total else 0.0,
                "matching_frames": match_count,
                "total_frames": total,
            }
    return agreements


def print_ablation_table(
    frame_logs: list[dict],
    agreements: dict,
    prompt_keys: list[str],
    total_frames: int,
    triggered_count: int,
) -> None:
    """Print decision distributions and pairwise agreement."""
    print("\n" + "=" * 70)
    print("PROMPT ABLATION — DECISION DISTRIBUTION ON TRIGGERED FRAMES")
    print("=" * 70)
    print(f"Total frames evaluated   : {total_frames}")
    print(
        f"Triggered frames (BLIP)  : {triggered_count} "
        f"({(triggered_count / total_frames * 100.0) if total_frames else 0.0:.1f}%)"
    )
    print(
        f"Non-triggered (geometry) : {total_frames - triggered_count} "
        f"({((total_frames - triggered_count) / total_frames * 100.0) if total_frames else 0.0:.1f}%)"
    )
    print()

    directions = ["left", "right", "center", "stop", None]
    print(f"{'Prompt':<20} {'left':>7} {'right':>7} {'center':>7} {'stop':>7} {'none':>7}")
    print("-" * 70)

    for key in prompt_keys:
        decisions = [frame_log["decisions"][key] for frame_log in frame_logs]
        count_total = len(decisions) if decisions else 1
        marker = " <- current" if key == CURRENT_PROMPT_KEY else ""
        row = f"{key:<20}"
        for direction in directions:
            count = decisions.count(direction)
            row += f" {count / count_total * 100.0:>6.1f}%"
        print(row + marker)

    print("\nPairwise decision agreement:")
    for pair_key, result in agreements.items():
        print(
            f"  {pair_key:<35}: "
            f"{result['agreement_rate'] * 100.0:.1f}% "
            f"({result['matching_frames']}/{result['total_frames']} frames)"
        )

    min_agreement = min((value["agreement_rate"] for value in agreements.values()), default=0.0)
    print("\nInterpretation:")
    if min_agreement >= 0.80:
        print(
            f"  Decision outputs show limited prompt sensitivity under this prompt set "
            f"({min_agreement * 100.0:.1f}% minimum pairwise agreement)."
        )
    elif min_agreement >= 0.60:
        print(
            f"  Decision outputs show moderate prompt sensitivity "
            f"({min_agreement * 100.0:.1f}% minimum pairwise agreement)."
        )
    else:
        print(
            f"  Decision outputs show substantial prompt sensitivity "
            f"({min_agreement * 100.0:.1f}% minimum pairwise agreement)."
        )
    print("=" * 70)


def save_agreement_plot(frame_logs: list[dict], agreements: dict, prompt_keys: list[str], output_dir: Path) -> None:
    """Save decision distribution and pairwise agreement plots if matplotlib is available."""
    if plt is None:
        print("[INFO] matplotlib not available; skipping prompt ablation plot")
        return

    save_path = output_dir / "prompt_ablation_agreement.png"
    directions = ["left", "right", "center", "stop", "none"]
    denominator = len(frame_logs) if frame_logs else 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prompt Ablation — BLIP Decision Stability", fontsize=12)

    x_positions = np.arange(len(directions))
    bar_width = 0.25
    colors = ["steelblue", "darkorange", "seagreen"]

    for index, key in enumerate(prompt_keys):
        decisions = [frame_log["decisions"][key] for frame_log in frame_logs]
        counts = [
            decisions.count(direction if direction != "none" else None) / denominator * 100.0
            for direction in directions
        ]
        axes[0].bar(
            x_positions + index * bar_width,
            counts,
            bar_width,
            label=key,
            color=colors[index % len(colors)],
            alpha=0.85,
        )

    axes[0].set_xticks(x_positions + bar_width)
    axes[0].set_xticklabels(directions)
    axes[0].set_ylabel("Decision rate (%)")
    axes[0].set_title("Decision Distribution per Prompt")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    pair_keys = list(agreements.keys())
    pair_rates = [agreements[key]["agreement_rate"] * 100.0 for key in pair_keys]
    colors = ["green" if value >= 80 else "orange" if value >= 60 else "red" for value in pair_rates]
    axes[1].bar(range(len(pair_keys)), pair_rates, color=colors, alpha=0.85)
    axes[1].axhline(80, color="green", linestyle="--", linewidth=1.5, label="80% agreement")
    axes[1].axhline(60, color="orange", linestyle="--", linewidth=1.5, label="60% agreement")
    axes[1].set_xticks(range(len(pair_keys)))
    axes[1].set_xticklabels([key.replace("_vs_", "\nvs\n") for key in pair_keys], fontsize=8)
    axes[1].set_ylabel("Agreement rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Pairwise Decision Agreement")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved prompt ablation plot to {save_path}")


def main():
    args = parse_args()
    prompt_keys = list(PROMPT_VARIANTS.keys())

    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(dataset_root, args.split, args.annotated_config)
    output_dir = split_output_dir(args.split) / "prompt_ablation"
    ensure_dir(output_dir)

    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Max scenes: {args.max_scenes}")
    print(f"[INFO] Frames per scene: {args.frames_per_scene}")
    print(f"[INFO] Sampling mode: {args.sampling_mode}")

    scenes = find_scenes(dataset_root, max_scenes=args.max_scenes)
    if not scenes:
        raise RuntimeError(f"No HM3D scenes found under {dataset_root}")

    models = load_models(semantic_backend="blip")
    frame_logs = []
    total_frames = 0
    triggered_count = 0

    for scene_id, scene_path in scenes:
        frames, scene_spec = render_frames_from_scene(
            scene_path,
            scene_id,
            dataset_root,
            annotated_config,
            args.frames_per_scene,
            args.sampling_mode,
            args.trajectory_length,
        )
        print(
            f"[INFO] Scene {scene_id} "
            f"({'semantic-enabled' if scene_spec['semantic_enabled'] else 'geometry-only stage load'}) "
            f"with {len(frames)} frames"
        )

        for frame_index, frame in enumerate(frames):
            total_frames += 1
            fired, trigger_info = should_trigger(frame, models)
            if not fired:
                continue

            triggered_count += 1
            decisions = {}
            raw_answers = {}
            for key, question in PROMPT_VARIANTS.items():
                decision, answer = run_blip_with_question(
                    frame,
                    models["processor"],
                    models["blip"],
                    question,
                )
                decisions[key] = decision
                raw_answers[key] = answer

            frame_logs.append(
                {
                    "scene_id": scene_id,
                    "frame_index": frame_index,
                    "trigger": trigger_info,
                    "decisions": decisions,
                    "raw_answers": raw_answers,
                }
            )

    if not frame_logs:
        print("[WARN] No triggered frames found in this run.")
        print("[WARN] Try increasing --max-scenes or --frames-per-scene.")
        return

    agreements = compute_pairwise_agreement(frame_logs, prompt_keys)
    print_ablation_table(frame_logs, agreements, prompt_keys, total_frames, triggered_count)
    save_agreement_plot(frame_logs, agreements, prompt_keys, output_dir)

    output = {
        "split": args.split,
        "dataset_root": str(dataset_root),
        "annotated_config": str(annotated_config) if annotated_config is not None else None,
        "max_scenes": args.max_scenes,
        "frames_per_scene": args.frames_per_scene,
        "total_frames": total_frames,
        "triggered_frames": triggered_count,
        "trigger_rate": triggered_count / total_frames if total_frames else 0.0,
        "prompt_variants": PROMPT_VARIANTS,
        "current_prompt": CURRENT_PROMPT_KEY,
        "agreements": agreements,
        "frame_logs": frame_logs,
    }
    output_path = output_dir / f"prompt_ablation_{args.split}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[INFO] Saved prompt ablation results to {output_path}")


if __name__ == "__main__":
    main()
