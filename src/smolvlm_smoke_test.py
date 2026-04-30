"""Minimal smoke test for SmolVLM as a semantic backend candidate."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from PIL import Image
import torch

from config import DEVICE, DIRECTION_KEYWORDS
from evaluate_hm3d import find_scenes, render_frames_from_scene
from hm3d_dataset import resolve_annotated_config, resolve_dataset_root


DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Instruct"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a practical SmolVLM smoke test on one HM3D frame."
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
        "--scene-id",
        default=None,
        help="Optional scene folder name. Defaults to the first discovered scene.",
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
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SmolVLM checkpoint to test.",
    )
    return parser.parse_args()


def resolve_scene(scenes: list[tuple[str, Path]], requested_scene_id: str | None):
    """Return the scene tuple for the requested or default scene."""
    if not scenes:
        raise RuntimeError("No scenes available for smoke test.")
    if requested_scene_id is None:
        return scenes[0]
    for scene_id, scene_path in scenes:
        if scene_id == requested_scene_id:
            return scene_id, scene_path
    raise RuntimeError(f"Scene '{requested_scene_id}' not found in the selected split.")


def parse_decision(answer: str) -> str | None:
    """Map raw model output into the project's navigation vocabulary."""
    lowered = answer.lower().strip()
    for direction, keywords in DIRECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                return direction
    return None


def _cuda_allocated_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0


def _cuda_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / 1e9 if DEVICE == "cuda" else 0.0


def load_smolvlm_backend(model_name: str):
    """Load SmolVLM with practical settings for a feasibility-first smoke test."""
    try:
        from transformers import SmolVLMForConditionalGeneration, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "SmolVLM smoke test requires a recent transformers version "
            "with vision-language support. Try: pip install --upgrade transformers"
        ) from exc

    print(f"[INFO] Running on: {DEVICE}")
    print(f"[INFO] Loading SmolVLM model: {model_name}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"[INFO] CUDA allocated before load: {_cuda_allocated_gb():.2f} GB")
        print(f"[INFO] CUDA reserved before load : {_cuda_reserved_gb():.2f} GB")

    load_start = time.time()
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    load_time = time.time() - load_start

    if DEVICE == "cuda":
        print(f"[INFO] CUDA allocated after load : {_cuda_allocated_gb():.2f} GB")
        print(f"[INFO] CUDA reserved after load  : {_cuda_reserved_gb():.2f} GB")

    print(f"[INFO] SmolVLM loaded in {load_time:.2f}s")
    return model, processor


def query_smolvlm(model, processor, frame, prompt: str) -> tuple[str | None, str, float]:
    """Run one SmolVLM inference on one frame."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text_input],
        images=[pil_image],
        return_tensors="pt",
    )

    target_device = next(model.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    infer_start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    infer_time = time.time() - infer_start

    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_length:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).lower().strip()
    decision = parse_decision(answer)
    return decision, answer, infer_time


def main():
    args = parse_args()
    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(
        dataset_root, args.split, args.annotated_config
    )
    scenes = find_scenes(dataset_root, max_scenes=None)
    scene_id, scene_path = resolve_scene(scenes, args.scene_id)

    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Scene: {scene_id}")
    print(f"[INFO] Model: {args.model_name}")

    frames, scene_spec = render_frames_from_scene(
        scene_path,
        scene_id,
        dataset_root,
        annotated_config,
        frames_per_scene=1,
        sampling_mode=args.sampling_mode,
        trajectory_length=args.trajectory_length,
    )
    if not frames:
        raise RuntimeError("No frames were rendered for the smoke test.")

    print(
        f"[INFO] Scene annotations: "
        f"{'semantic-enabled' if scene_spec['semantic_enabled'] else 'geometry-only stage load'}"
    )

    model, processor = load_smolvlm_backend(args.model_name)

    print("\n" + "=" * 72)
    print("SMOLVLM SMOKE TEST")
    print("=" * 72)
    print(f"{'Prompt':<15} {'Decision':<10} {'Latency':>10}  Raw answer")
    print("-" * 72)

    decisions = []
    parseable = True

    for key, prompt in PROMPTS.items():
        decision, answer, latency = query_smolvlm(model, processor, frames[0], prompt)
        decisions.append(decision)
        parseable = parseable and decision is not None
        print(
            f"{key:<15} {str(decision):<10} {latency*1000:>8.0f}ms  {answer}"
        )

    print("-" * 72)
    unique_decisions = len(set(decisions))
    if unique_decisions == 1:
        stability_note = "All prompts agreed on this frame."
    else:
        stability_note = (
            f"Prompts produced {unique_decisions} distinct decisions on this frame."
        )
    print(f"[INFO] Prompt consistency: {stability_note}")

    if DEVICE == "cuda":
        print(
            "[INFO] Peak CUDA allocated during last inference: "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
        print(
            "[INFO] CUDA reserved after inference: "
            f"{_cuda_reserved_gb():.2f} GB"
        )

    print("\n[VERDICT]")
    if parseable and unique_decisions == 1:
        print("  PASS — parseable output and no prompt disagreement on this frame.")
    elif parseable:
        print("  PARTIAL — parseable output, but prompt sensitivity is already visible.")
    else:
        print("  FAIL — one or more outputs were not parseable into navigation labels.")


if __name__ == "__main__":
    main()
