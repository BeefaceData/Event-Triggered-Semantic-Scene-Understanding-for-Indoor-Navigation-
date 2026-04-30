"""Reusable frame-level pipeline."""

import time

from geometry import (
    build_guidance,
    compute_region_scores,
    compute_region_clearance,
    detect_obstacles,
    estimate_depth,
    fuse_depth_with_obstacles,
    geometry_decision,
    is_center_blocked,
    normalize_depth,
    visualize_depth,
)
from semantics import detect_navigation_signage, ocr_decision, semantic_decision as run_semantic_decision
from trigger import evaluate_trigger
from config import USE_OBSTACLE_FUSION


def process_frame(frame, models, trigger_mode="uncertainty", semantic_policy="event_triggered"):
    """
    Process a single frame end to end.

    Returns a structured dictionary containing:
    - geometry information
    - OCR information
    - BLIP information
    - trigger information
    - final proposed decision
    """
    yolo = models["yolo"]
    midas = models["midas"]
    transform = models["transform"]
    ocr = models["ocr"]
    semantic_processor = models["semantic_processor"]
    semantic_model = models["semantic_model"]
    semantic_backend = models["semantic_backend"]

    # Shared geometry backbone.
    backbone_start = time.time()
    obstacles = detect_obstacles(frame, yolo)
    depth_map = estimate_depth(frame, midas, transform)
    depth_map = normalize_depth(depth_map)
    fused_depth_map = fuse_depth_with_obstacles(depth_map, obstacles) if USE_OBSTACLE_FUSION else depth_map
    scores = compute_region_scores(fused_depth_map)
    clearance = compute_region_clearance(fused_depth_map, obstacles)
    center_blocked = is_center_blocked(obstacles, frame.shape[1])
    geo_dir = geometry_decision(scores, center_blocked)
    backbone_latency = (time.time() - backbone_start) * 1000

    # OCR baseline and semantic cue source.
    ocr_start = time.time()
    ocr_dir, ocr_text = ocr_decision(frame, ocr)
    signage_hits = detect_navigation_signage(frame, ocr)
    ocr_latency = (time.time() - ocr_start) * 1000

    # Semantic baseline / semantic query backend.
    semantic_query_latency = 0.0
    semantic_query_dir = None
    semantic_query_answer = None

    # Trigger evaluation.
    trigger_start = time.time()
    trigger_record = evaluate_trigger(scores, center_blocked, frame, ocr, mode=trigger_mode)
    semantic_invoked = False
    selected_semantic_decision = None
    selected_semantic_answer = None

    if semantic_policy == "always_semantic":
        semantic_start = time.time()
        selected_semantic_decision, selected_semantic_answer = run_semantic_decision(
            frame,
            semantic_processor,
            semantic_model,
            semantic_backend,
        )
        semantic_query_latency = (time.time() - semantic_start) * 1000
        semantic_query_dir = selected_semantic_decision
        semantic_query_answer = selected_semantic_answer
        semantic_invoked = True
        final_direction = selected_semantic_decision if selected_semantic_decision else geo_dir
    elif semantic_policy == "event_triggered" and trigger_record.get("triggered", trigger_record.get("fired", False)):
        semantic_start = time.time()
        selected_semantic_decision, selected_semantic_answer = run_semantic_decision(
            frame,
            semantic_processor,
            semantic_model,
            semantic_backend,
        )
        semantic_query_latency = (time.time() - semantic_start) * 1000
        semantic_query_dir = selected_semantic_decision
        semantic_query_answer = selected_semantic_answer
        semantic_invoked = True
        final_direction = selected_semantic_decision if selected_semantic_decision else geo_dir
    else:
        final_direction = geo_dir

    guidance = build_guidance(
        clearance=clearance,
        center_blocked=center_blocked,
        signage_hits=signage_hits,
        proposed_direction=final_direction,
    )

    proposed_latency = (time.time() - trigger_start) * 1000

    return {
        "backbone_latency_ms": backbone_latency,
        "obstacle_count": len(obstacles),
        "obstacles": obstacles,
        "center_blocked": center_blocked,
        "depth_visualization": visualize_depth(fused_depth_map),
        "raw_depth_visualization": visualize_depth(depth_map),
        "scores": scores,
        "guidance": guidance,
        "baseline_geometry": {
            "decision": geo_dir,
            "latency_ms": backbone_latency,
        },
        "baseline_ocr": {
            "decision": ocr_dir,
            "ocr_text": ocr_text,
            "signage_hits": signage_hits,
            "latency_ms": ocr_latency,
        },
        "semantic_backend": semantic_backend,
        "baseline_semantic": {
            "backend": semantic_backend,
            "decision": semantic_query_dir,
            "answer": semantic_query_answer,
            "latency_ms": semantic_query_latency,
        },
        "baseline_blip": {
            "decision": semantic_query_dir,
            "answer": semantic_query_answer,
            "latency_ms": semantic_query_latency,
        },
        "semantic_policy": semantic_policy,
        "semantic_invoked": semantic_invoked,
        "trigger": trigger_record,
        "proposed": {
            "decision": final_direction,
            "semantic_decision": selected_semantic_decision,
            "semantic_answer": selected_semantic_answer,
            "latency_ms": proposed_latency,
        },
    }
