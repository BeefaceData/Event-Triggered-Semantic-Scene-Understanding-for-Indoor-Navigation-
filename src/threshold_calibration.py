"""Calibrate trigger thresholds over HM3D frames using the project pipeline."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
import easyocr

from config import DEVICE, FRAMES_PER_SCENE, IMG_HEIGHT, IMG_WIDTH, YOLO_WEIGHTS
from evaluate_hm3d import (
    ensure_dir,
    find_scenes,
    render_frames_from_scene,
)
from geometry import (
    compute_region_scores,
    detect_obstacles,
    estimate_depth,
    fuse_depth_with_obstacles,
    is_center_blocked,
    normalize_depth,
)
from hm3d_dataset import resolve_annotated_config, resolve_dataset_root, split_output_dir
from trigger import evaluate_trigger, relative_separability, uncertainty_entropy

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency at runtime
    plt = None


DEFAULT_TAU_SEP_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
DEFAULT_TAU_ENT_VALUES = [0.80, 0.85, 0.90, 0.95, 1.00, 1.03, 1.05, 1.08]


def parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated float list."""
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric threshold value.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep trigger separability and entropy thresholds on HM3D frames."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["example", "minival", "val", "train"],
        help="HM3D split to use for calibration.",
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
        default=20,
        help="Maximum number of scenes to use for calibration.",
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
    parser.add_argument(
        "--tau-sep-values",
        default=",".join(str(value) for value in DEFAULT_TAU_SEP_VALUES),
        help="Comma-separated separability thresholds to sweep.",
    )
    parser.add_argument(
        "--tau-ent-values",
        default=",".join(str(value) for value in DEFAULT_TAU_ENT_VALUES),
        help="Comma-separated entropy thresholds to sweep.",
    )
    parser.add_argument(
        "--current-tau-sep",
        type=float,
        default=0.08,
        help="Current separability operating point to highlight in outputs.",
    )
    parser.add_argument(
        "--current-tau-ent",
        type=float,
        default=1.03,
        help="Current entropy operating point to highlight in outputs.",
    )
    return parser.parse_args()


def load_calibration_models():
    """Load only the models needed for trigger calibration."""
    print(f"[INFO] Running on: {DEVICE}")

    print("[INFO] Loading YOLOv8...")
    yolo = YOLO(str(YOLO_WEIGHTS))
    yolo.to(DEVICE)

    print("[INFO] Loading MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(DEVICE)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print("[INFO] Loading EasyOCR...")
    ocr = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))

    print("[INFO] Calibration models loaded")
    return {
        "yolo": yolo,
        "midas": midas,
        "transform": transform,
        "ocr": ocr,
    }


def precompute_frame_metrics(
    frames: list[np.ndarray],
    scene_id: str,
    models: dict,
    trigger_mode: str = "uncertainty",
) -> list[dict]:
    """Run the heavy perception stack once per frame and cache trigger inputs."""
    yolo = models["yolo"]
    midas = models["midas"]
    transform = models["transform"]
    ocr = models["ocr"]

    metrics = []
    for frame_index, frame in enumerate(frames):
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
            mode=trigger_mode,
        )
        metrics.append(
            {
                "scene_id": scene_id,
                "frame_index": frame_index,
                "scores": scores,
                "delta_norm": float(relative_separability(scores)),
                "entropy": float(uncertainty_entropy(scores)),
                "center_blocked": bool(center_blocked),
                "semantic_cue": bool(trigger_record["text_hits"]),
                "trigger_reasons_current": list(trigger_record["reasons"]),
            }
        )
    return metrics


def apply_thresholds(metric: dict, tau_sep: float, tau_ent: float) -> dict:
    """Apply one threshold pair to a cached frame metric record."""
    cond_a = metric["delta_norm"] < tau_sep
    cond_b = metric["entropy"] > tau_ent
    cond_c = metric["center_blocked"]
    cond_d = metric["semantic_cue"]
    fired = cond_a or cond_b or cond_c or cond_d
    return {
        "fired": fired,
        "cond_A": cond_a,
        "cond_B": cond_b,
        "cond_C": cond_c,
        "cond_D": cond_d,
        "only_A": cond_a and not cond_b and not cond_c and not cond_d,
        "only_B": cond_b and not cond_a and not cond_c and not cond_d,
        "AB": cond_a and cond_b,
    }


def run_sweep(metrics: list[dict], tau_sep_values: list[float], tau_ent_values: list[float]) -> dict:
    """Sweep all threshold combinations over cached frame metrics."""
    frame_count = max(len(metrics), 1)
    grid = {}

    for tau_sep, tau_ent in product(tau_sep_values, tau_ent_values):
        applied = [apply_thresholds(metric, tau_sep, tau_ent) for metric in metrics]
        grid[(tau_sep, tau_ent)] = {
            "tau_sep": tau_sep,
            "tau_ent": tau_ent,
            "trigger_rate": sum(item["fired"] for item in applied) / frame_count,
            "selectivity": 1.0 - (sum(item["fired"] for item in applied) / frame_count),
            "cond_A_rate": sum(item["cond_A"] for item in applied) / frame_count,
            "cond_B_rate": sum(item["cond_B"] for item in applied) / frame_count,
            "cond_C_rate": sum(item["cond_C"] for item in applied) / frame_count,
            "cond_D_rate": sum(item["cond_D"] for item in applied) / frame_count,
            "A_only_rate": sum(item["only_A"] for item in applied) / frame_count,
            "B_only_rate": sum(item["only_B"] for item in applied) / frame_count,
            "AB_rate": sum(item["AB"] for item in applied) / frame_count,
        }

    return grid


def print_sweep_table(
    grid: dict,
    tau_sep_values: list[float],
    tau_ent_values: list[float],
    current_tau_sep: float,
    current_tau_ent: float,
) -> None:
    """Print a compact trigger-rate table to the terminal."""
    print("\n" + "=" * 88)
    print("THRESHOLD CALIBRATION — TRIGGER RATE GRID")
    print("=" * 88)

    header_label = "tau_sep \\\\ tau_ent"
    header = f"{header_label:<16}" + "".join(f"{value:>9.2f}" for value in tau_ent_values)
    print(header)
    print("-" * len(header))

    for tau_sep in tau_sep_values:
        row = f"{tau_sep:<16.2f}"
        for tau_ent in tau_ent_values:
            rate = grid[(tau_sep, tau_ent)]["trigger_rate"] * 100.0
            marker = "*" if abs(tau_sep - current_tau_sep) < 1e-9 and abs(tau_ent - current_tau_ent) < 1e-9 else " "
            row += f"{rate:>7.1f}%{marker}"
        print(row)

    print("-" * len(header))
    print(
        f"* current operating point: tau_sep={current_tau_sep:.2f}, "
        f"tau_ent={current_tau_ent:.2f}"
    )

    key = (current_tau_sep, current_tau_ent)
    if key in grid:
        item = grid[key]
        print("\nCurrent operating point detail:")
        print(f"  trigger_rate  : {item['trigger_rate'] * 100.0:.1f}%")
        print(f"  selectivity   : {item['selectivity'] * 100.0:.1f}%")
        print(f"  cond_A_rate   : {item['cond_A_rate'] * 100.0:.1f}%")
        print(f"  cond_B_rate   : {item['cond_B_rate'] * 100.0:.1f}%")
        print(f"  cond_C_rate   : {item['cond_C_rate'] * 100.0:.1f}%")
        print(f"  cond_D_rate   : {item['cond_D_rate'] * 100.0:.1f}%")
        print(f"  A_only_rate   : {item['A_only_rate'] * 100.0:.1f}%")
        print(f"  B_only_rate   : {item['B_only_rate'] * 100.0:.1f}%")
        print(f"  AB_rate       : {item['AB_rate'] * 100.0:.1f}%")


def maybe_save_plots(
    grid: dict,
    metrics: list[dict],
    tau_sep_values: list[float],
    tau_ent_values: list[float],
    current_tau_sep: float,
    current_tau_ent: float,
    output_dir: Path,
) -> None:
    """Save heatmap and tradeoff plots if matplotlib is available."""
    if plt is None:
        print("[INFO] matplotlib not available; skipping plot generation")
        return

    heatmap_path = output_dir / "threshold_calibration_heatmap.png"
    tradeoff_path = output_dir / "threshold_calibration_tradeoff.png"

    trigger_rates = np.zeros((len(tau_sep_values), len(tau_ent_values)))
    for i, tau_sep in enumerate(tau_sep_values):
        for j, tau_ent in enumerate(tau_ent_values):
            trigger_rates[i, j] = grid[(tau_sep, tau_ent)]["trigger_rate"] * 100.0

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(trigger_rates, cmap="YlOrRd", aspect="auto", origin="lower", vmin=0, vmax=100)
    plt.colorbar(image, ax=ax, label="Trigger rate (%)")
    ax.set_xticks(range(len(tau_ent_values)))
    ax.set_yticks(range(len(tau_sep_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in tau_ent_values])
    ax.set_yticklabels([f"{value:.2f}" for value in tau_sep_values])
    ax.set_xlabel("Entropy threshold tau_ent")
    ax.set_ylabel("Separability threshold tau_sep")
    ax.set_title("Trigger Rate Heatmap")

    if current_tau_sep in tau_sep_values and current_tau_ent in tau_ent_values:
        ax.plot(
            tau_ent_values.index(current_tau_ent),
            tau_sep_values.index(current_tau_sep),
            marker="*",
            color="blue",
            markersize=14,
        )

    for i in range(len(tau_sep_values)):
        for j in range(len(tau_ent_values)):
            text_color = "white" if trigger_rates[i, j] > 60 else "black"
            ax.text(j, i, f"{trigger_rates[i, j]:.0f}%", ha="center", va="center", fontsize=8, color=text_color)

    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    delta_values = [item["delta_norm"] for item in metrics]
    entropy_values = [item["entropy"] for item in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Threshold Tradeoff Analysis", fontsize=12)

    for tau_sep in tau_sep_values:
        rates = [grid[(tau_sep, tau_ent)]["trigger_rate"] * 100.0 for tau_ent in tau_ent_values]
        line_width = 2.5 if abs(tau_sep - current_tau_sep) < 1e-9 else 1.0
        line_style = "-" if abs(tau_sep - current_tau_sep) < 1e-9 else "--"
        axes[0].plot(tau_ent_values, rates, linewidth=line_width, linestyle=line_style, label=f"tau_sep={tau_sep:.2f}")
    axes[0].axvline(current_tau_ent, color="red", linestyle=":", linewidth=1.5)
    axes[0].set_xlabel("Entropy threshold tau_ent")
    axes[0].set_ylabel("Trigger rate (%)")
    axes[0].set_title("Trigger Rate vs tau_ent")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    for tau_ent in tau_ent_values:
        rates = [grid[(tau_sep, tau_ent)]["trigger_rate"] * 100.0 for tau_sep in tau_sep_values]
        line_width = 2.5 if abs(tau_ent - current_tau_ent) < 1e-9 else 1.0
        line_style = "-" if abs(tau_ent - current_tau_ent) < 1e-9 else "--"
        axes[1].plot(tau_sep_values, rates, linewidth=line_width, linestyle=line_style, label=f"tau_ent={tau_ent:.2f}")
    axes[1].axvline(current_tau_sep, color="red", linestyle=":", linewidth=1.5)
    axes[1].set_xlabel("Separability threshold tau_sep")
    axes[1].set_ylabel("Trigger rate (%)")
    axes[1].set_title("Trigger Rate vs tau_sep")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)

    axes[2].hist(delta_values, bins=20, alpha=0.6, color="steelblue", density=True, label="delta_norm")
    twin = axes[2].twinx()
    twin.hist(entropy_values, bins=20, alpha=0.6, color="darkorange", density=True, label="entropy")
    axes[2].axvline(current_tau_sep, color="steelblue", linestyle="--", linewidth=2)
    twin.axvline(current_tau_ent, color="darkorange", linestyle="--", linewidth=2)
    axes[2].set_title("Empirical Distributions")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("delta_norm density", color="steelblue")
    twin.set_ylabel("entropy density", color="darkorange")
    handles_a, labels_a = axes[2].get_legend_handles_labels()
    handles_b, labels_b = twin.get_legend_handles_labels()
    axes[2].legend(handles_a + handles_b, labels_a + labels_b, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved heatmap to {heatmap_path}")
    print(f"[INFO] Saved tradeoff plot to {tradeoff_path}")


def main():
    args = parse_args()
    tau_sep_values = parse_float_list(args.tau_sep_values)
    tau_ent_values = parse_float_list(args.tau_ent_values)

    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(dataset_root, args.split, args.annotated_config)
    output_dir = split_output_dir(args.split) / "calibration"
    ensure_dir(output_dir)

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Sampling mode: {args.sampling_mode}")
    print(f"[INFO] Frames per scene: {args.frames_per_scene}")
    print(f"[INFO] Max scenes: {args.max_scenes}")
    print(f"[INFO] tau_sep values: {tau_sep_values}")
    print(f"[INFO] tau_ent values: {tau_ent_values}")

    scenes = find_scenes(dataset_root, max_scenes=args.max_scenes)
    if not scenes:
        raise RuntimeError(f"No HM3D scenes found under {dataset_root}")

    models = load_calibration_models()
    all_metrics = []

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
            f"[INFO] Calibrating on scene {scene_id} "
            f"({'semantic-enabled' if scene_spec['semantic_enabled'] else 'geometry-only stage load'}) "
            f"with {len(frames)} frames"
        )
        all_metrics.extend(precompute_frame_metrics(frames, scene_id, models))

    if not all_metrics:
        raise RuntimeError("No frame metrics were computed.")

    delta_values = [item["delta_norm"] for item in all_metrics]
    entropy_values = [item["entropy"] for item in all_metrics]

    print(
        f"[INFO] Empirical delta_norm: mean={np.mean(delta_values):.4f}, "
        f"median={np.median(delta_values):.4f}, std={np.std(delta_values):.4f}"
    )
    print(
        f"[INFO] Empirical entropy: mean={np.mean(entropy_values):.4f}, "
        f"median={np.median(entropy_values):.4f}, std={np.std(entropy_values):.4f}"
    )
    print(f"[INFO] Max possible entropy for 3 regions: {np.log(3):.4f}")

    grid = run_sweep(all_metrics, tau_sep_values, tau_ent_values)
    print_sweep_table(
        grid,
        tau_sep_values,
        tau_ent_values,
        args.current_tau_sep,
        args.current_tau_ent,
    )
    maybe_save_plots(
        grid,
        all_metrics,
        tau_sep_values,
        tau_ent_values,
        args.current_tau_sep,
        args.current_tau_ent,
        output_dir,
    )

    serializable_grid = {
        f"sep={tau_sep}_ent={tau_ent}": result for (tau_sep, tau_ent), result in grid.items()
    }
    payload = {
        "split": args.split,
        "dataset_root": str(dataset_root),
        "annotated_config": str(annotated_config) if annotated_config is not None else None,
        "frames_per_scene": args.frames_per_scene,
        "max_scenes": args.max_scenes,
        "sampling_mode": args.sampling_mode,
        "trajectory_length": args.trajectory_length,
        "tau_sep_values": tau_sep_values,
        "tau_ent_values": tau_ent_values,
        "current_tau_sep": args.current_tau_sep,
        "current_tau_ent": args.current_tau_ent,
        "num_frames": len(all_metrics),
        "empirical_stats": {
            "delta_norm": {
                "mean": float(np.mean(delta_values)),
                "median": float(np.median(delta_values)),
                "std": float(np.std(delta_values)),
                "min": float(np.min(delta_values)),
                "max": float(np.max(delta_values)),
            },
            "entropy": {
                "mean": float(np.mean(entropy_values)),
                "median": float(np.median(entropy_values)),
                "std": float(np.std(entropy_values)),
                "min": float(np.min(entropy_values)),
                "max": float(np.max(entropy_values)),
                "max_possible": float(np.log(3)),
            },
        },
        "grid": serializable_grid,
    }

    output_json = output_dir / f"threshold_calibration_{args.split}.json"
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Saved calibration results to {output_json}")


if __name__ == "__main__":
    main()
