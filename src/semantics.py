"""Semantic reasoning helpers."""

from __future__ import annotations

import cv2
from PIL import Image
import torch

from config import (
    BLIP_QUESTION,
    DEVICE,
    DIRECTION_KEYWORDS,
    OCR_CONFIDENCE_THRESHOLD,
    SEMANTIC_BACKEND,
    SIGNAGE_KEYWORDS,
    SMOLVLM_MAX_IMAGE_EDGE,
    SMOLVLM_QUESTION,
)


def blip_decision(frame, processor, blip):
    """Query BLIP for a directional decision on a frame."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, BLIP_QUESTION, return_tensors="pt")
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


def _move_inputs_to_device(inputs, device):
    moved = {}
    for key, value in inputs.items():
        if value.dtype == torch.float32 and device == "cuda":
            moved[key] = value.to(device).half()
        else:
            moved[key] = value.to(device)
    return moved


def _resize_for_smolvlm(pil_image):
    resized = pil_image.copy()
    if max(resized.size) <= SMOLVLM_MAX_IMAGE_EDGE:
        return resized

    resample = getattr(Image, "Resampling", Image).BICUBIC
    resized.thumbnail((SMOLVLM_MAX_IMAGE_EDGE, SMOLVLM_MAX_IMAGE_EDGE), resample)
    return resized


def smolvlm_decision(frame, processor, model):
    """Query SmolVLM for a directional decision on a frame."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_image = _resize_for_smolvlm(pil_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SMOLVLM_QUESTION},
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
    target_device = _resolve_device(model)
    if target_device.startswith("cuda"):
        torch.cuda.empty_cache()
    inputs = _move_inputs_to_device(inputs, target_device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    input_length = inputs["input_ids"].shape[1]
    answer = processor.decode(output[0][input_length:], skip_special_tokens=True).lower().strip()
    if target_device.startswith("cuda"):
        del inputs
        del output
        torch.cuda.empty_cache()
    for direction, keywords in DIRECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in answer:
                return direction, answer
    return None, answer


def _resolve_device(model):
    """Resolve a usable device string from a loaded model."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def semantic_decision(frame, processor, model, backend: str | None = None):
    """Dispatch semantic reasoning to the configured backend."""
    backend = backend or SEMANTIC_BACKEND
    if backend == "blip":
        return blip_decision(frame, processor, model)
    if backend == "smolvlm":
        return smolvlm_decision(frame, processor, model)
    raise ValueError(f"Unsupported semantic backend: {backend}")


def ocr_decision(frame, reader):
    """Return the first useful OCR directional decision found in a frame."""
    results = reader.readtext(frame)
    for (_, text, confidence) in results:
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            continue
        lowered = text.lower().strip()
        for direction, keywords in DIRECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered:
                    return direction, lowered
    return None, None


def detect_text_cues(frame, reader, trigger_keywords):
    """Detect whether the frame contains text that should trigger semantics."""
    results = reader.readtext(frame)
    hits = []
    for (_, text, confidence) in results:
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            continue
        lowered = text.lower().strip()
        if any(keyword in lowered for keyword in trigger_keywords):
            hits.append({"text": lowered, "confidence": float(confidence)})
    return hits


def detect_navigation_signage(frame, reader):
    """Collect OCR hits that look like route-following signage."""
    results = reader.readtext(frame)
    hits = []
    for (_, text, confidence) in results:
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            continue
        lowered = text.lower().strip()
        if any(keyword in lowered for keyword in SIGNAGE_KEYWORDS):
            direction = None
            for candidate_direction, keywords in DIRECTION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in lowered:
                        direction = candidate_direction
                        break
                if direction is not None:
                    break
            hits.append(
                {
                    "text": lowered,
                    "confidence": float(confidence),
                    "direction": direction,
                }
            )
    return hits
