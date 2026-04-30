"""Run the Stage 1 trigger pipeline inside a minimal closed-loop HM3D rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import habitat_sim
import numpy as np

from audio_narration import generate_spoken_audio
from closed_loop_controller import ClosedLoopController
from config import IMG_HEIGHT, IMG_WIDTH, OUTPUT_ROOT, SEMANTIC_BACKEND_OPTIONS
from evaluate_hm3d import build_scene_load_spec, ensure_dir, find_scenes
from hm3d_dataset import resolve_annotated_config, resolve_dataset_root
from models import load_models
from pipeline import process_frame
from visualization import annotate_closed_loop_frame, stack_rgb_and_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal closed-loop HM3D rollout.")
    parser.add_argument(
        "--split",
        default="example",
        choices=["example", "minival", "val", "train"],
        help="HM3D split to use.",
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
        "--scene-id",
        default=None,
        help="Optional scene folder name to run. Defaults to the first discovered scene.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Maximum number of closed-loop steps.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the annotated rollout video.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=5,
        help="Playback FPS for the saved rollout video.",
    )
    parser.add_argument(
        "--add-audio",
        action="store_true",
        help="Add spoken direction audio to the saved rollout video.",
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
        help="Semantic invocation policy to use during closed-loop control.",
    )
    parser.add_argument(
        "--semantic-backend",
        default=None,
        choices=SEMANTIC_BACKEND_OPTIONS,
        help="Semantic backend to use in the pipeline.",
    )
    return parser.parse_args()


def make_sim_config(scene_glb_path: Path, dataset_root: Path, annotated_config: Path | None):
    """Create a Habitat-Sim configuration for a closed-loop rollout."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    scene_spec = build_scene_load_spec(scene_glb_path, dataset_root, annotated_config)
    backend_cfg.scene_id = scene_spec["scene_id"]
    if scene_spec["scene_dataset_config_file"] is not None:
        backend_cfg.scene_dataset_config_file = scene_spec["scene_dataset_config_file"]
    backend_cfg.enable_physics = False

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_spec.hfov = 90

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]

    action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.25),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
        "stop": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.0),
        ),
    }
    agent_cfg.action_space = action_space

    return habitat_sim.Configuration(backend_cfg, [agent_cfg]), scene_spec


def observation_to_bgr(observation):
    """Convert a Habitat-Sim observation to BGR."""
    rgb = np.asarray(observation)[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def initialize_agent_state(sim, agent):
    """Place the agent at a random navigable point with a random heading."""
    nav_point = sim.pathfinder.get_random_navigable_point()
    state = habitat_sim.AgentState()
    state.position = nav_point
    yaw = np.random.uniform(0, 2 * np.pi)
    state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
        yaw,
        np.array([0, 1, 0]),
    )
    agent.set_state(state)


def resolve_scene(scenes, requested_scene_id: str | None):
    """Pick the target scene for the rollout."""
    if requested_scene_id is None:
        return scenes[0]

    for scene_id, scene_path in scenes:
        if scene_id == requested_scene_id:
            return scene_id, scene_path

    raise RuntimeError(f"Requested scene_id '{requested_scene_id}' was not found.")


def make_video_writer(path: Path, width: int, height: int, fps: int = 5):
    """Create an mp4 writer for rollout visualization."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def safe_depth_vis(record, height: int, width: int):
    """Return a usable depth visualization, even if the record is missing one."""
    vis = record.get("depth_visualization")
    if vis is not None:
        return vis

    blank = np.full((height, width, 3), 64, dtype=np.uint8)
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


def run_rollout_episode(
    sim,
    agent,
    models,
    controller: ClosedLoopController,
    scene_id: str,
    semantic_enabled: bool,
    steps: int,
    semantic_policy: str = "event_triggered",
    writer=None,
):
    """Execute one closed-loop rollout and return step records plus summary metrics."""
    collision_like_distance_threshold_m = 0.05
    stuck_forward_step_threshold = 3
    poor_progress_displacement_threshold_m = 1.0
    poor_progress_efficiency_threshold = 0.5

    controller.reset()
    rollout_records = []
    path_length_m = 0.0
    stalled_forward_steps = 0
    collision_like_steps = 0
    consecutive_stalled_forward_steps = 0
    max_consecutive_stalled_forward_steps = 0
    action_counts = {
        "move_forward": 0,
        "turn_left": 0,
        "turn_right": 0,
        "stop": 0,
    }

    for step_index in range(steps):
        pre_state = agent.get_state()
        pre_position = np.asarray(pre_state.position, dtype=float)

        observation = sim.get_sensor_observations()["color_sensor"]
        frame = observation_to_bgr(observation)
        record = process_frame(frame, models, semantic_policy=semantic_policy)
        action = controller.select_action(record)
        action_counts[action] = action_counts.get(action, 0) + 1

        if action == "stop":
            post_position = pre_position
            moved_distance = 0.0
            stopped = True
        else:
            agent.act(action)
            post_state = agent.get_state()
            post_position = np.asarray(post_state.position, dtype=float)
            moved_distance = float(np.linalg.norm(post_position - pre_position))
            stopped = False

        controller.observe_transition(action, moved_distance)

        if action == "move_forward":
            path_length_m += moved_distance
            if moved_distance < collision_like_distance_threshold_m:
                stalled_forward_steps += 1
                collision_like_steps += 1
                consecutive_stalled_forward_steps += 1
                max_consecutive_stalled_forward_steps = max(
                    max_consecutive_stalled_forward_steps,
                    consecutive_stalled_forward_steps,
                )
            else:
                consecutive_stalled_forward_steps = 0
        else:
            consecutive_stalled_forward_steps = 0

        rollout_record = {
            "scene_id": scene_id,
            "step_index": step_index,
            "semantic_enabled": semantic_enabled,
            "action": action,
            "agent_position_before": pre_position.tolist(),
            "agent_position_after": post_position.tolist(),
            "moved_distance_m": moved_distance,
            "stalled_forward": action == "move_forward"
            and moved_distance < collision_like_distance_threshold_m,
            "collision_like": action == "move_forward"
            and moved_distance < collision_like_distance_threshold_m,
            "triggered": record["trigger"]["triggered"],
            "trigger_reasons": list(record["trigger"]["reasons"]),
            "semantic_policy": semantic_policy,
            "semantic_invoked": bool(record.get("semantic_invoked", False)),
            "final_direction": record["proposed"]["decision"],
            "guidance": record.get("guidance", {}),
            "guidance_text": record.get("guidance", {}).get("text"),
            "guidance_type": record.get("guidance", {}).get("type"),
            "guidance_raw_text": record.get("guidance", {}).get("raw_text"),
            "guidance_raw_type": record.get("guidance", {}).get("raw_type"),
            "guidance_movement_override": bool(record.get("guidance", {}).get("movement_override", False)),
            "center_clearance": float(record.get("guidance", {}).get("center_clearance", 0.0)),
            "geometry_direction": record["baseline_geometry"]["decision"],
            "semantic_direction": record["proposed"]["semantic_decision"],
            "center_blocked": record["center_blocked"],
            "entropy": float(record["trigger"]["entropy"]),
            "relative_separability": float(record["trigger"]["relative_separability"]),
            "backbone_latency_ms": float(record["backbone_latency_ms"]),
            "proposed_latency_ms": float(record["proposed"]["latency_ms"]),
        }
        rollout_records.append(rollout_record)

        triggered_steps = sum(1 for item in rollout_records if item["triggered"])
        annotated = annotate_closed_loop_frame(
            frame,
            record,
            step_index=step_index,
            action=action,
            triggered_steps=triggered_steps,
        )
        combo = stack_rgb_and_depth(
            annotated,
            safe_depth_vis(record, IMG_HEIGHT, IMG_WIDTH),
        )
        if writer is not None:
            writer.write(combo)

        if stopped:
            print(f"[INFO] Rollout stopped at step {step_index}")
            break

    start_position = np.asarray(rollout_records[0]["agent_position_before"], dtype=float)
    end_position = np.asarray(rollout_records[-1]["agent_position_after"], dtype=float)
    net_displacement_m = float(np.linalg.norm(end_position - start_position))
    triggered_steps = sum(1 for item in rollout_records if item["triggered"])
    semantic_calls = sum(1 for item in rollout_records if item.get("semantic_invoked"))
    progress_efficiency = (
        net_displacement_m / path_length_m if path_length_m > 1e-8 else 0.0
    )
    stuck_episode = (
        collision_like_steps >= stuck_forward_step_threshold
        or max_consecutive_stalled_forward_steps >= stuck_forward_step_threshold
    )
    poor_progress_episode = (
        net_displacement_m < poor_progress_displacement_threshold_m
        or (path_length_m > poor_progress_displacement_threshold_m and progress_efficiency < poor_progress_efficiency_threshold)
    )

    summary = {
        "num_steps": len(rollout_records),
        "num_triggered_steps": triggered_steps,
        "semantic_calls": semantic_calls,
        "trigger_rate": (triggered_steps / len(rollout_records) * 100.0) if rollout_records else 0.0,
        "path_length_m": path_length_m,
        "net_displacement_m": net_displacement_m,
        "progress_efficiency": progress_efficiency,
        "stalled_forward_steps": stalled_forward_steps,
        "collision_like_steps": collision_like_steps,
        "max_consecutive_stalled_forward_steps": max_consecutive_stalled_forward_steps,
        "stuck_episode": stuck_episode,
        "poor_progress_episode": poor_progress_episode,
        "action_counts": action_counts,
    }
    return rollout_records, summary


def main():
    args = parse_args()
    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(dataset_root, args.split, args.annotated_config)
    scenes = find_scenes(dataset_root)
    if not scenes:
        raise RuntimeError(f"No HM3D scenes found under {dataset_root}")

    scene_id, scene_path = resolve_scene(scenes, args.scene_id)
    print(f"[INFO] Stage 2 closed-loop scene: {scene_id}")

    sim_cfg, scene_spec = make_sim_config(scene_path, dataset_root, annotated_config)
    sim = habitat_sim.Simulator(sim_cfg)
    agent = sim.initialize_agent(0)
    initialize_agent_state(sim, agent)

    models = load_models(semantic_backend=args.semantic_backend)
    controller = ClosedLoopController()

    writer = None
    video_path = None
    silent_video_path = None
    rollout_dir = OUTPUT_ROOT / args.split / "closed_loop"
    if args.save_video:
        ensure_dir(rollout_dir)
        video_stem = f"{scene_id}_closed_loop_{args.semantic_policy}"
        if args.semantic_policy in {"event_triggered", "always_semantic"}:
            backend_suffix = models["semantic_backend"] or "none"
            video_stem = f"{video_stem}_{backend_suffix}"
        video_path = rollout_dir / f"{video_stem}.mp4"
        silent_video_path = (
            rollout_dir / f"{video_stem}_silent.mp4" if args.add_audio else video_path
        )
        writer = make_video_writer(
            silent_video_path,
            IMG_WIDTH * 2,
            IMG_HEIGHT,
            fps=args.video_fps,
        )

    rollout_records, rollout_summary = run_rollout_episode(
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
            records=rollout_records,
            fps=args.video_fps,
            mode=args.audio_mode,
            output_video_path=video_path,
        )

    ensure_dir(rollout_dir)
    output_name = f"{scene_id}_closed_loop_{args.semantic_policy}.json"
    if args.semantic_policy in {"event_triggered", "always_semantic"}:
        backend_suffix = models["semantic_backend"] or "none"
        output_name = f"{scene_id}_closed_loop_{args.semantic_policy}_{backend_suffix}.json"
    output_path = rollout_dir / output_name
    output = {
        "split": args.split,
        "dataset_root": str(dataset_root),
        "scene_id": scene_id,
        "semantic_policy": args.semantic_policy,
        "semantic_enabled": scene_spec["semantic_enabled"],
        "video_path": str(video_path) if video_path is not None else None,
        "silent_video_path": str(silent_video_path) if args.add_audio and silent_video_path is not None else None,
        "video_with_audio_path": str(narrated_video_path) if narrated_video_path is not None else None,
        "num_steps": rollout_summary["num_steps"],
        "num_triggered_steps": rollout_summary["num_triggered_steps"],
        "semantic_calls": rollout_summary["semantic_calls"],
        "trigger_rate": rollout_summary["trigger_rate"],
        "path_length_m": rollout_summary["path_length_m"],
        "net_displacement_m": rollout_summary["net_displacement_m"],
        "progress_efficiency": rollout_summary["progress_efficiency"],
        "stalled_forward_steps": rollout_summary["stalled_forward_steps"],
        "collision_like_steps": rollout_summary["collision_like_steps"],
        "max_consecutive_stalled_forward_steps": rollout_summary["max_consecutive_stalled_forward_steps"],
        "stuck_episode": rollout_summary["stuck_episode"],
        "poor_progress_episode": rollout_summary["poor_progress_episode"],
        "action_counts": rollout_summary["action_counts"],
        "records": rollout_records,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[INFO] Closed-loop rollout saved to {output_path}")
    if video_path is not None:
        print(f"[INFO] Closed-loop video saved to {video_path}")

    sim.close()


if __name__ == "__main__":
    main()
