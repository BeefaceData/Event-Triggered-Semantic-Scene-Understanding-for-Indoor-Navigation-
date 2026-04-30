"""Compare semantic backends on the same triggered HM3D frames.

Backends:
    - BLIP-vqa-base (baseline)
    - SmolVLM2-500M-Instruct
    - Qwen2-VL-2B-Instruct

Design:
    - Same trigger, same triggered frames, same prompt family across backends
    - BLIP agreement is computed from the current run when BLIP is loaded
    - Historical BLIP prompt-ablation agreement is kept only as annotation
    - Pairwise agreements are stored per backend for full inspection
    - Latency is reported as average query latency, not deployment latency
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from config import DEVICE, DIRECTION_KEYWORDS, FRAMES_PER_SCENE, YOLO_WEIGHTS
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
from trigger import evaluate_trigger, relative_separability, uncertainty_entropy

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - optional plotting at runtime
    HAS_MPL = False


ALL_BACKENDS = ["blip", "smolvlm", "qwen"]

PROMPTS = {
    "V1_detailed": (
        "You are helping with indoor navigation. "
        "Answer with one word only: left, center, or right. "
        "Which direction is best to continue safely?"
    ),
    "V2_minimal": "Which direction: left, center, or right?",
    "V3_clearpath": (
        "For indoor navigation, which single direction is clearest: "
        "left, center, or right?"
    ),
}

HISTORICAL_BLIP_MIN_AGREEMENT = 0.554


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare BLIP, SmolVLM2-500M, and Qwen2-VL-2B on triggered HM3D frames."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["example", "minival", "val", "train"],
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--annotated-config", default=None)
    parser.add_argument("--frames-per-scene", type=int, default=FRAMES_PER_SCENE)
    parser.add_argument("--max-scenes", type=int, default=5)
    parser.add_argument(
        "--sampling-mode",
        default="trajectory",
        choices=["random", "trajectory"],
    )
    parser.add_argument("--trajectory-length", type=int, default=15)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=ALL_BACKENDS,
        choices=ALL_BACKENDS,
        help="Backends to test. Default: blip smolvlm qwen",
    )
    return parser.parse_args()


def parse_decision(answer: str) -> str | None:
    """Map raw VLM output to the navigation vocabulary."""
    lowered = answer.lower().strip()
    for direction, keywords in DIRECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                return direction
    return None


def _move_to_device(inputs: dict, device: str) -> dict:
    """Move tensor inputs to a device and cast float tensors on CUDA."""
    moved = {}
    for key, value in inputs.items():
        if value.dtype == torch.float32 and device == "cuda":
            moved[key] = value.to(device).half()
        else:
            moved[key] = value.to(device)
    return moved


def _resolve_device(model) -> str:
    """Resolve a usable device string from a model loaded on one device."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def load_blip(device: str) -> dict:
    from transformers import BlipForQuestionAnswering, BlipProcessor

    print("[INFO] Loading BLIP-vqa-base...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print("[INFO] BLIP ready")
    return {"name": "blip", "model": model, "processor": processor, "device": device}


def load_smolvlm(device: str) -> dict:
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration

    print("[INFO] Loading SmolVLM2-500M-Instruct...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = SmolVLMForConditionalGeneration.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Instruct",
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    print("[INFO] SmolVLM2-500M ready")
    return {"name": "smolvlm", "model": model, "processor": processor, "device": device}


def load_qwen(device: str) -> dict:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    print("[INFO] Loading Qwen2-VL-2B-Instruct (4-bit)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    print("[INFO] Qwen2-VL-2B ready")
    return {"name": "qwen", "model": model, "processor": processor, "device": device}


def load_backend(name: str, device: str) -> dict:
    if name == "blip":
        return load_blip(device)
    if name == "smolvlm":
        return load_smolvlm(device)
    if name == "qwen":
        return load_qwen(device)
    raise ValueError(f"Unknown backend: {name}")


def query_blip(backend: dict, frame: np.ndarray, question: str) -> tuple[str | None, str, float]:
    processor = backend["processor"]
    model = backend["model"]
    device = backend["device"]
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(images=pil_image, text=question, return_tensors="pt")
    inputs = _move_to_device(inputs, device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    latency = time.time() - start

    answer = processor.decode(output[0], skip_special_tokens=True).lower().strip()
    return parse_decision(answer), answer, latency


def query_smolvlm(backend: dict, frame: np.ndarray, question: str) -> tuple[str | None, str, float]:
    processor = backend["processor"]
    model = backend["model"]
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text_input], images=[pil_image], return_tensors="pt")
    target = _resolve_device(model)
    inputs = {key: value.to(target) for key, value in inputs.items()}

    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    latency = time.time() - start

    input_length = inputs["input_ids"].shape[1]
    answer = processor.decode(output[0][input_length:], skip_special_tokens=True).lower().strip()
    return parse_decision(answer), answer, latency


def query_qwen(backend: dict, frame: np.ndarray, question: str) -> tuple[str | None, str, float]:
    processor = backend["processor"]
    model = backend["model"]
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text_input], images=[pil_image], return_tensors="pt")
    target = _resolve_device(model)
    inputs = {key: value.to(target) for key, value in inputs.items()}

    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    latency = time.time() - start

    input_length = inputs["input_ids"].shape[1]
    answer = processor.decode(output[0][input_length:], skip_special_tokens=True).lower().strip()
    return parse_decision(answer), answer, latency


def query_backend(backend: dict, frame: np.ndarray, question: str) -> tuple[str | None, str, float]:
    if backend["name"] == "blip":
        return query_blip(backend, frame, question)
    if backend["name"] == "smolvlm":
        return query_smolvlm(backend, frame, question)
    if backend["name"] == "qwen":
        return query_qwen(backend, frame, question)
    raise ValueError(f"Unknown backend: {backend['name']}")


def load_geometry_models() -> dict:
    print("[INFO] Loading YOLOv8...")
    yolo = YOLO(str(YOLO_WEIGHTS))
    yolo.to(DEVICE)

    print("[INFO] Loading MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(DEVICE)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    print("[INFO] Loading EasyOCR...")
    ocr = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
    return {"yolo": yolo, "midas": midas, "transform": transform, "ocr": ocr}


def check_trigger(frame: np.ndarray, geo_models: dict) -> tuple[bool, dict]:
    obstacles = detect_obstacles(frame, geo_models["yolo"])
    raw_depth = estimate_depth(frame, geo_models["midas"], geo_models["transform"])
    normalized_depth = normalize_depth(raw_depth)
    fused_depth = fuse_depth_with_obstacles(normalized_depth, obstacles)
    scores = compute_region_scores(fused_depth)
    center_blocked = is_center_blocked(obstacles, frame.shape[1])

    trigger_record = evaluate_trigger(
        scores,
        center_blocked,
        frame,
        geo_models["ocr"],
        mode="uncertainty",
    )
    fired = trigger_record.get("triggered", trigger_record.get("fired", False))
    return fired, {
        "scores": scores,
        "delta_norm": float(relative_separability(scores)),
        "entropy": float(uncertainty_entropy(scores)),
        "center_blocked": bool(center_blocked),
        "reasons": list(trigger_record["reasons"]),
    }


def pairwise_agreements(frame_logs: list[dict], backend_name: str) -> dict[str, float]:
    """Return every prompt-pair agreement rate for one backend."""
    prompt_keys = list(PROMPTS.keys())
    agreements: dict[str, float] = {}
    for index, key_a in enumerate(prompt_keys):
        for key_b in prompt_keys[index + 1 :]:
            decisions_a = [frame_log["results"][backend_name][key_a]["decision"] for frame_log in frame_logs]
            decisions_b = [frame_log["results"][backend_name][key_b]["decision"] for frame_log in frame_logs]
            rate = sum(1 for a, b in zip(decisions_a, decisions_b) if a == b) / len(decisions_a) if decisions_a else 0.0
            agreements[f"{key_a}_vs_{key_b}"] = rate
    return agreements


def min_pairwise_agreement(frame_logs: list[dict], backend_name: str) -> float:
    return min(pairwise_agreements(frame_logs, backend_name).values(), default=0.0)


def avg_query_latency_ms(frame_logs: list[dict], backend_name: str, prompt_key: str) -> float:
    latencies = [frame_log["results"][backend_name][prompt_key]["latency"] for frame_log in frame_logs]
    return sum(latencies) / len(latencies) * 1000.0 if latencies else 0.0


def print_comparison_table(
    frame_logs: list[dict],
    backends_tested: list[str],
    total_frames: int,
    triggered_count: int,
) -> None:
    directions = ["left", "right", "center", "stop", "none"]
    denominator = len(frame_logs) if frame_logs else 1

    current_blip = min_pairwise_agreement(frame_logs, "blip") if "blip" in backends_tested else None
    ref_agreement = current_blip if current_blip is not None else HISTORICAL_BLIP_MIN_AGREEMENT
    ref_label = "this run" if current_blip is not None else "historical ref"

    print("\n" + "=" * 78)
    print("BACKEND COMPARISON — SEMANTIC BACKENDS ON TRIGGERED HM3D FRAMES")
    print("=" * 78)
    print(f"Total frames evaluated  : {total_frames}")
    print(
        f"Triggered frames        : {triggered_count} "
        f"({(triggered_count / total_frames * 100.0) if total_frames else 0.0:.1f}%)"
    )
    print(f"Frames compared below   : {len(frame_logs)} (triggered only)")
    if current_blip is not None:
        print(f"BLIP min agreement      : {current_blip * 100.0:.1f}% (current run)")
    print(
        f"Historical reference    : {HISTORICAL_BLIP_MIN_AGREEMENT * 100.0:.1f}% "
        "(prompt_ablation.py annotation)"
    )

    for backend_name in backends_tested:
        print(f"\n--- {backend_name.upper()} ---")
        print(
            f"  {'Prompt':<20}"
            f"{'left':>8}{'right':>8}{'center':>8}{'stop':>8}{'none':>8}"
            f"  {'avg query lat':>14}"
        )
        print("  " + "-" * 76)

        for prompt_key in PROMPTS:
            decisions = [frame_log["results"][backend_name][prompt_key]["decision"] for frame_log in frame_logs]
            row = f"  {prompt_key:<20}"
            for direction in directions:
                count = decisions.count(direction if direction != "none" else None)
                row += f"{count / denominator * 100.0:>8.1f}%"
            row += f"  {avg_query_latency_ms(frame_logs, backend_name, prompt_key):>12.0f}ms"
            if prompt_key == "V1_detailed":
                row += "  <- default"
            print(row)

        sensitivity = min_pairwise_agreement(frame_logs, backend_name)
        print(f"\n  Min prompt agreement: {sensitivity * 100.0:.1f}%", end="")
        if backend_name == "blip":
            print("  (current-run baseline)")
        else:
            delta = sensitivity - ref_agreement
            sign = "+" if delta >= 0 else ""
            if sensitivity >= 0.80:
                print(f"  ✓ stable  ({sign}{delta * 100.0:.1f}pp vs BLIP {ref_label})")
            elif delta > 0:
                print(f"  ~ improved  ({sign}{delta * 100.0:.1f}pp vs BLIP {ref_label})")
            else:
                print(f"  ✗ no improvement  ({delta * 100.0:.1f}pp vs BLIP {ref_label})")

    print("\n" + "-" * 78)
    print("SUMMARY")
    print("-" * 78)
    print(
        f"  {'Backend':<18}{'Min agreement':>16}"
        f"  {'vs BLIP (' + ref_label + ')':>24}  {'Verdict':>12}"
    )
    print("  " + "-" * 74)
    for backend_name in backends_tested:
        sensitivity = min_pairwise_agreement(frame_logs, backend_name)
        delta = sensitivity - ref_agreement
        if backend_name == "blip":
            verdict = "baseline"
            delta_str = "-"
        elif sensitivity >= 0.80:
            verdict = "PROPOSED"
            delta_str = f"+{delta * 100.0:.1f}pp"
        elif delta > 0:
            verdict = "improved"
            delta_str = f"+{delta * 100.0:.1f}pp"
        else:
            verdict = "no gain"
            delta_str = f"{delta * 100.0:.1f}pp"
        print(
            f"  {backend_name:<18}{sensitivity * 100.0:>15.1f}%"
            f"  {delta_str:>26}  {verdict:>12}"
        )
    print("=" * 78)


def save_comparison_plot(frame_logs: list[dict], backends_tested: list[str], output_dir: Path) -> None:
    if not HAS_MPL or not frame_logs:
        return

    directions = ["left", "right", "center", "stop"]
    denominator = len(frame_logs)
    prompt_keys = list(PROMPTS.keys())
    colors = ["steelblue", "darkorange", "seagreen"]

    current_blip = min_pairwise_agreement(frame_logs, "blip") if "blip" in backends_tested else None
    ref_agreement = current_blip if current_blip is not None else HISTORICAL_BLIP_MIN_AGREEMENT
    ref_label = (
        f"BLIP this run {ref_agreement * 100.0:.1f}%"
        if current_blip is not None
        else f"BLIP historical {ref_agreement * 100.0:.1f}%"
    )

    fig, axes = plt.subplots(len(backends_tested), 2, figsize=(14, 4 * len(backends_tested)))
    if len(backends_tested) == 1:
        axes = [axes]

    fig.suptitle("Backend Comparison: Decision Stability on Triggered HM3D Frames", fontsize=12)

    for row_index, backend_name in enumerate(backends_tested):
        ax_dist = axes[row_index][0]
        ax_agree = axes[row_index][1]

        x_positions = np.arange(len(directions))
        bar_width = 0.25
        for index, prompt_key in enumerate(prompt_keys):
            decisions = [frame_log["results"][backend_name][prompt_key]["decision"] for frame_log in frame_logs]
            counts = [decisions.count(direction) / denominator * 100.0 for direction in directions]
            ax_dist.bar(
                x_positions + index * bar_width,
                counts,
                bar_width,
                label=prompt_key,
                color=colors[index],
                alpha=0.85,
            )
        ax_dist.set_xticks(x_positions + bar_width)
        ax_dist.set_xticklabels(directions)
        ax_dist.set_ylabel("Decision rate (%)")
        ax_dist.set_title(f"{backend_name.upper()} — Decision Distribution")
        ax_dist.legend(fontsize=7)
        ax_dist.grid(True, alpha=0.3, axis="y")
        ax_dist.set_ylim(0, 105)

        agreements = pairwise_agreements(frame_logs, backend_name)
        pair_labels = [label.replace("_vs_", "\nvs\n") for label in agreements]
        pair_rates = [rate * 100.0 for rate in agreements.values()]
        bar_colors = [
            "green" if rate >= 80.0 else "orange" if rate >= 60.0 else "red"
            for rate in pair_rates
        ]
        ax_agree.bar(range(len(pair_labels)), pair_rates, color=bar_colors, alpha=0.85)
        ax_agree.axhline(80.0, color="green", linestyle="--", linewidth=1.5, label="80%")
        ax_agree.axhline(ref_agreement * 100.0, color="red", linestyle=":", linewidth=1.5, label=ref_label)
        ax_agree.set_xticks(range(len(pair_labels)))
        ax_agree.set_xticklabels(pair_labels, fontsize=7)
        ax_agree.set_ylabel("Pairwise agreement (%)")
        ax_agree.set_title(f"{backend_name.upper()} — Pairwise Prompt Agreement")
        ax_agree.legend(fontsize=7)
        ax_agree.grid(True, alpha=0.3, axis="y")
        ax_agree.set_ylim(0, 105)

    plt.tight_layout()
    save_path = output_dir / "backend_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to {save_path}")


def main():
    args = parse_args()

    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(
        dataset_root, args.split, args.annotated_config
    )
    output_dir = split_output_dir(args.split) / "backend_comparison"
    ensure_dir(output_dir)

    print(f"[INFO] Split         : {args.split}")
    print(f"[INFO] Max scenes    : {args.max_scenes}")
    print(f"[INFO] Frames/scene  : {args.frames_per_scene}")
    print(f"[INFO] Backends      : {args.backends}")

    scenes = find_scenes(dataset_root, max_scenes=args.max_scenes)
    if not scenes:
        raise RuntimeError(f"No HM3D scenes found under {dataset_root}")

    geo_models = load_geometry_models()

    frame_logs: list[dict] = []
    triggered_frames: list[dict] = []
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
        print(f"\n[INFO] Scene {scene_id} -- {len(frames)} frames")

        for frame_index, frame in enumerate(frames):
            total_frames += 1
            fired, trigger_info = check_trigger(frame, geo_models)
            if not fired:
                continue

            triggered_count += 1
            frame_result = {
                "scene": scene_id,
                "frame_index": frame_index,
                "trigger": trigger_info,
                "results": {},
            }

            frame_logs.append(frame_result)
            triggered_frames.append(
                {
                    "scene": scene_id,
                    "frame_index": frame_index,
                    "frame": frame.copy(),
                }
            )
            print(f"  [f{frame_index:02d} TRIGGERED] queued for backend evaluation")

    if not frame_logs:
        print("[WARN] No triggered frames found.")
        return

    del geo_models
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    loaded_backends: list[str] = []

    print("\n[INFO] Evaluating semantic backends sequentially...")
    for backend_name in args.backends:
        print(f"\n[INFO] Backend {backend_name}...")
        try:
            backend = load_backend(backend_name, DEVICE)
        except Exception as exc:
            print(f"[WARN] Could not load {backend_name}: {exc} -- skipping")
            continue

        loaded_backends.append(backend_name)
        for frame_log, triggered_item in zip(frame_logs, triggered_frames):
            frame = triggered_item["frame"]
            frame_log["results"][backend_name] = {}
            for prompt_key, question in PROMPTS.items():
                decision, raw_answer, latency = query_backend(backend, frame, question)
                frame_log["results"][backend_name][prompt_key] = {
                    "decision": decision,
                    "raw_answer": raw_answer,
                    "latency": latency,
                }

            live_decision = frame_log["results"][backend_name]["V1_detailed"]["decision"]
            print(
                f"  [scene={frame_log['scene']} f{frame_log['frame_index']:02d}] "
                f"{backend_name}={live_decision}"
            )

        del backend
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if not loaded_backends:
        raise RuntimeError("No semantic backends loaded.")

    print_comparison_table(frame_logs, loaded_backends, total_frames, triggered_count)
    save_comparison_plot(frame_logs, loaded_backends, output_dir)

    current_blip = min_pairwise_agreement(frame_logs, "blip") if "blip" in loaded_backends else None

    payload = {
        "split": args.split,
        "total_frames": total_frames,
        "triggered_frames": triggered_count,
        "trigger_rate": triggered_count / total_frames if total_frames else 0.0,
        "backends_tested": loaded_backends,
        "prompt_variants": PROMPTS,
        "blip_agreement": {
            "current_run": current_blip,
            "historical_reference": HISTORICAL_BLIP_MIN_AGREEMENT,
            "note": (
                "current_run computed from this experiment. "
                "historical_reference from prompt_ablation.py on a prior subset."
            ),
        },
        "current_run_min_agreement": {
            backend_name: min_pairwise_agreement(frame_logs, backend_name)
            for backend_name in loaded_backends
        },
        "current_run_pairwise_agreements": {
            backend_name: pairwise_agreements(frame_logs, backend_name)
            for backend_name in loaded_backends
        },
        "frame_logs": frame_logs,
    }
    out_path = output_dir / f"backend_comparison_{args.split}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()
