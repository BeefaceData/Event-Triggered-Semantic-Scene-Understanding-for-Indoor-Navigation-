"""Minimal smoke test for Qwen2-VL as a semantic backend candidate."""

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


DEFAULT_PROMPT = (
    "Answer with exactly one word: left, center, or right. "
    "Which direction should the agent take?"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a minimal Qwen2-VL smoke test on one HM3D frame."
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
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to test.",
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


def load_qwen_backend():
    """Load Qwen2-VL with conservative settings for a practical smoke test."""
    try:
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise ImportError(
            "Qwen smoke test requires transformers with Qwen2-VL support. "
            "Try: pip install --upgrade transformers"
        ) from exc

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Qwen 4-bit loading requires bitsandbytes. "
            "Try: pip install bitsandbytes"
        ) from exc

    print(f"[INFO] Running on: {DEVICE}")
    print("[INFO] Loading Qwen2-VL-2B-Instruct (4-bit)...")
    load_start = time.time()

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

    load_time = time.time() - load_start
    print(f"[INFO] Qwen loaded in {load_time:.2f}s")
    return model, processor


def query_qwen(model, processor, frame, prompt: str) -> tuple[str | None, str, float]:
    """Run one Qwen inference on one frame."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
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

    infer_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=1.0,
        )
    infer_time = time.time() - infer_start

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).lower().strip()
    decision = parse_decision(answer)
    return decision, answer, infer_time


def main():
    args = parse_args()
    dataset_root = resolve_dataset_root(args.split, args.dataset_root)
    annotated_config = resolve_annotated_config(dataset_root, args.split, args.annotated_config)
    scenes = find_scenes(dataset_root, max_scenes=None)
    scene_id, scene_path = resolve_scene(scenes, args.scene_id)

    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Scene: {scene_id}")
    print(f"[INFO] Prompt: {args.prompt}")

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

    model, processor = load_qwen_backend()
    decision, answer, infer_time = query_qwen(model, processor, frames[0], args.prompt)

    print("\n[RESULT] Qwen smoke test")
    print(f"  Parsed decision : {decision}")
    print(f"  Raw answer      : {answer}")
    print(f"  Inference time  : {infer_time:.2f}s")


if __name__ == "__main__":
    main()
