"""Evaluate the Stage 2 closed-loop prototype across multiple HM3D scenes."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import habitat_sim

from audio_narration import generate_spoken_audio
from closed_loop_controller import ClosedLoopController
from config import IMG_HEIGHT, IMG_WIDTH, OUTPUT_ROOT, SEMANTIC_BACKEND_OPTIONS
from evaluate_hm3d import ensure_dir, find_scenes
from hm3d_dataset import resolve_annotated_config, resolve_dataset_root
from models import load_models
from run_closed_loop_hm3d import (
    initialize_agent_state,
    make_sim_config,
    make_video_writer,
    run_rollout_episode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch closed-loop evaluation on HM3D.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["example", "minival", "val", "train"],
        help="HM3D split to evaluate.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional explicit dataset root.",
    )
    parser.add_argument(
        "--annotated-config",
        default=None,
        help="Optional explicit semantic dataset config path.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=10,
        help="Maximum number of scenes to evaluate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Maximum rollout steps per scene.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional custom filename for the aggregate results JSON.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rollout videos for a limited number of evaluated scenes.",
    )
    parser.add_argument(
        "--max-video-scenes",
        type=int,
        default=2,
        help="Maximum number of evaluated scenes for which videos are saved.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=5,
        help="Playback FPS for any saved evaluation rollout videos.",
    )
    parser.add_argument(
        "--add-audio",
        action="store_true",
        help="Add spoken narration to any saved evaluation rollout videos.",
    )
    parser.add_argument(
        "--audio-mode",
        default="direction",
        choices=["direction", "action"],
        help="Whether spoken narration follows meaningful navigation events or executed actions.",
    )
    parser.add_argument(
        "--semantic-policy",
        default="event_triggered",
        choices=["geometry_only", "always_semantic", "event_triggered"],
        help="Semantic invocation policy to evaluate.",
    )
    parser.add_argument(
        "--semantic-backend",
        default=None,
        choices=SEMANTIC_BACKEND_OPTIONS,
        help="Semantic backend to use in the pipeline.",
    )
    return parser.parse_args()


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(dataset_root, args.split, args.annotated_config)
    scenes = find_scenes(dataset_root)
    if not scenes:
        raise RuntimeError(f"No HM3D scenes found under {dataset_root}")

    scene_subset = scenes[: args.max_scenes]
    print(f"[INFO] Stage 2 batch evaluation split: {args.split}")
    print(f"[INFO] Evaluating {len(scene_subset)} scene(s) with {args.steps} step max per scene")
    print(f"[INFO] Semantic policy: {args.semantic_policy}")

    models = load_models(semantic_backend=args.semantic_backend)
    controller = ClosedLoopController()
    episodes = []
    all_records = []
    action_counts = Counter()
    trigger_reason_counts = Counter()
    output_dir = OUTPUT_ROOT / args.split / "closed_loop"
    ensure_dir(output_dir)

    for scene_index, (scene_id, scene_path) in enumerate(scene_subset):
        print(f"[INFO] Closed-loop episode scene: {scene_id}")
        sim_cfg, scene_spec = make_sim_config(scene_path, dataset_root, annotated_config)
        sim = habitat_sim.Simulator(sim_cfg)
        agent = sim.initialize_agent(0)
        initialize_agent_state(sim, agent)

        writer = None
        video_path = None
        silent_video_path = None
        if args.save_video and scene_index < args.max_video_scenes:
            video_stem = f"{scene_id}_closed_loop_eval_{args.semantic_policy}"
            if args.semantic_policy == "event_triggered":
                video_stem = f"{scene_id}_closed_loop_eval"
            video_path = output_dir / f"{video_stem}.mp4"
            silent_video_path = output_dir / f"{video_stem}_silent.mp4" if args.add_audio else video_path
            writer = make_video_writer(
                silent_video_path,
                width=IMG_WIDTH * 2,
                height=IMG_HEIGHT,
                fps=args.video_fps,
            )

        episode_records, episode_summary = run_rollout_episode(
            sim=sim,
            agent=agent,
            models=models,
            controller=controller,
            scene_id=scene_id,
            semantic_enabled=scene_spec["semantic_enabled"],
            steps=args.steps,
            semantic_policy=args.semantic_policy,
            writer=writer,
        )
        if writer is not None:
            writer.release()
        narrated_video_path = None
        if args.save_video and args.add_audio and video_path is not None:
            narrated_video_path = generate_spoken_audio(
                video_path=silent_video_path,
                records=episode_records,
                fps=args.video_fps,
                mode=args.audio_mode,
                output_video_path=video_path,
            )
        sim.close()

        for record in episode_records:
            all_records.append(record)
            action_counts[record["action"]] += 1
            for reason in record["trigger_reasons"]:
                trigger_reason_counts[reason] += 1

        episodes.append(
            {
                "scene_id": scene_id,
                "semantic_policy": args.semantic_policy,
                "semantic_enabled": scene_spec["semantic_enabled"],
                "video_path": str(video_path) if video_path is not None else None,
                "silent_video_path": str(silent_video_path) if args.add_audio and silent_video_path is not None else None,
                "video_with_audio_path": str(narrated_video_path) if narrated_video_path is not None else None,
                **episode_summary,
            }
        )
    default_output_name = f"closed_loop_eval_{args.split}_{args.semantic_policy}.json"
    if args.semantic_policy in {"event_triggered", "always_semantic"}:
        backend_suffix = models["semantic_backend"] or "none"
        default_output_name = (
            f"closed_loop_eval_{args.split}_{args.semantic_policy}_{backend_suffix}.json"
        )
    output_name = args.output_name or default_output_name
    output_path = output_dir / output_name

    summary = {
        "num_episodes": len(episodes),
        "max_steps_per_episode": args.steps,
        "total_steps": sum(ep["num_steps"] for ep in episodes),
        "total_triggered_steps": sum(ep["num_triggered_steps"] for ep in episodes),
        "total_semantic_calls": sum(ep["semantic_calls"] for ep in episodes),
        "overall_trigger_rate": mean(
            [record["triggered"] for record in all_records]
        )
        * 100.0,
        "mean_episode_trigger_rate": mean(ep["trigger_rate"] for ep in episodes),
        "mean_semantic_calls_per_episode": mean(ep["semantic_calls"] for ep in episodes),
        "mean_path_length_m": mean(ep["path_length_m"] for ep in episodes),
        "mean_net_displacement_m": mean(ep["net_displacement_m"] for ep in episodes),
        "mean_progress_efficiency": mean(ep["progress_efficiency"] for ep in episodes),
        "mean_stalled_forward_steps": mean(ep["stalled_forward_steps"] for ep in episodes),
        "mean_collision_like_steps": mean(ep["collision_like_steps"] for ep in episodes),
        "mean_max_consecutive_stalled_forward_steps": mean(
            ep["max_consecutive_stalled_forward_steps"] for ep in episodes
        ),
        "stuck_episodes": sum(1 for ep in episodes if ep["stuck_episode"]),
        "poor_progress_episodes": sum(1 for ep in episodes if ep["poor_progress_episode"]),
        "semantic_enabled_episodes": sum(1 for ep in episodes if ep["semantic_enabled"]),
        "action_counts": dict(action_counts),
        "trigger_reason_counts": dict(trigger_reason_counts),
        "mean_backbone_latency_ms": mean(record["backbone_latency_ms"] for record in all_records),
        "mean_proposed_latency_ms": mean(record["proposed_latency_ms"] for record in all_records),
    }

    output = {
        "split": args.split,
        "semantic_policy": args.semantic_policy,
        "semantic_backend": models["semantic_backend"],
        "dataset_root": str(dataset_root),
        "annotated_config": str(annotated_config) if annotated_config is not None else None,
        "video_enabled": args.save_video,
        "max_video_scenes": args.max_video_scenes,
        "video_fps": args.video_fps,
        "audio_enabled": args.add_audio,
        "audio_mode": args.audio_mode,
        "summary": summary,
        "episodes": episodes,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[INFO] Closed-loop evaluation saved to {output_path}")


if __name__ == "__main__":
    main()
