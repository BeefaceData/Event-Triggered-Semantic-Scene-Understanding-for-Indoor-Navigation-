"""Model loading utilities."""

from __future__ import annotations

import torch
from ultralytics import YOLO
import easyocr

from config import (
    BLIP_MODEL_NAME,
    DEVICE,
    SEMANTIC_BACKEND,
    SMOLVLM_MODEL_NAME,
    YOLO_WEIGHTS,
)


def _load_blip(device: str):
    from transformers import BlipForQuestionAnswering, BlipProcessor

    print("[INFO] Loading BLIP-vqa-base...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, use_fast=False)
    model = BlipForQuestionAnswering.from_pretrained(
        BLIP_MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return processor, model


def _load_smolvlm(device: str):
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration

    print("[INFO] Loading SmolVLM2-500M-Instruct...")
    semantic_device = device
    if semantic_device != device:
        print("[INFO] Loading SmolVLM on CPU to preserve GPU memory for Habitat + geometry.")
    model = SmolVLMForConditionalGeneration.from_pretrained(
        SMOLVLM_MODEL_NAME,
        dtype=torch.float16 if semantic_device == "cuda" else torch.float32,
        device_map="auto" if semantic_device == "cuda" else None,
    )
    if semantic_device != "cuda":
        model = model.to(semantic_device)
    model.eval()
    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_NAME)
    return processor, model


def load_models(semantic_backend: str | None = None):
    """Load all models once and return a shared model bundle."""
    print(f"[INFO] Running on: {DEVICE}")
    semantic_backend = semantic_backend or SEMANTIC_BACKEND

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

    if semantic_backend == "blip":
        processor, semantic_model = _load_blip(DEVICE)
    elif semantic_backend == "smolvlm":
        processor, semantic_model = _load_smolvlm(DEVICE)
    else:
        raise ValueError(f"Unsupported semantic backend: {semantic_backend}")

    print("[INFO] All models loaded")
    bundle = {
        "yolo": yolo,
        "midas": midas,
        "transform": transform,
        "ocr": ocr,
        "semantic_backend": semantic_backend,
        "semantic_processor": processor,
        "semantic_model": semantic_model,
    }
    if semantic_backend == "blip":
        bundle["processor"] = processor
        bundle["blip"] = semantic_model
    return bundle
