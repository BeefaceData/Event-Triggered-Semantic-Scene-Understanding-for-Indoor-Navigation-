"""Geometry utilities for obstacle detection and depth-based navigation."""

import cv2
import numpy as np
import torch

from config import CONFIDENCE_THRESHOLD, DEVICE, OBSTACLE_CLASSES


def estimate_depth(frame, midas, transform):
    """Return a per-frame relative depth map from MiDaS."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        depth = midas(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    return depth.cpu().numpy()


def normalize_depth(depth_map):
    """Normalize a depth map per frame for more stable score comparisons."""
    d_min = float(np.min(depth_map))
    d_max = float(np.max(depth_map))
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth_map, dtype=np.float32)
    return (depth_map - d_min) / (d_max - d_min)


def visualize_depth(depth_map):
    """Convert a normalized depth map into an 8-bit image for display."""
    return cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def detect_obstacles(frame, yolo):
    """Detect a small set of obstacle classes from a frame."""
    results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    obstacles = []
    for result in results:
        for box in result.boxes:
            cls_name = yolo.names[int(box.cls)]
            if cls_name in OBSTACLE_CLASSES:
                obstacles.append(
                    {
                        "class": cls_name,
                        "bbox": box.xyxy[0].cpu().numpy().tolist(),  # Convert to list for JSON serialization
                        "confidence": float(box.conf),
                    }
                )
    return obstacles


def compute_region_scores(depth_map):
    """
    Split the frame into left/center/right regions.
    Lower mean score is treated as more navigable within the current frame.
    """
    width = depth_map.shape[1]
    regions = {
        "left": depth_map[:, : width // 3],
        "center": depth_map[:, width // 3 : 2 * width // 3],
        "right": depth_map[:, 2 * width // 3 :],
    }
    return {region: float(np.mean(values)) for region, values in regions.items()}


def compute_region_clearance(depth_map, obstacles):
    """
    Estimate relative clearance for left/center/right from the fused depth map.

    Higher values mean the region looks more open within the current frame.
    """
    scores = compute_region_scores(depth_map)
    clearance = {region: float(1.0 - value) for region, value in scores.items()}
    width = depth_map.shape[1]
    region_bounds = {
        "left": (0, width // 3),
        "center": (width // 3, 2 * width // 3),
        "right": (2 * width // 3, width),
    }

    for obstacle in obstacles:
        x1, _, x2, _ = map(int, obstacle["bbox"])
        x1 = max(0, x1)
        x2 = min(width, x2)
        if x2 <= x1:
            continue
        for region, (region_x1, region_x2) in region_bounds.items():
            overlap = max(0, min(x2, region_x2) - max(x1, region_x1))
            if overlap <= 0:
                continue
            overlap_ratio = overlap / max(region_x2 - region_x1, 1)
            clearance[region] = max(0.0, clearance[region] - 0.35 * overlap_ratio)

    return clearance


def fuse_depth_with_obstacles(depth_map, obstacles):
    """
    Mark detected obstacle regions as less navigable in the depth map so the
    geometry score reflects obstacle presence more directly.
    """
    fused = depth_map.copy()
    blocked_value = float(np.max(depth_map))
    for obstacle in obstacles:
        x1, y1, x2, y2 = map(int, obstacle["bbox"])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fused.shape[1], x2)
        y2 = min(fused.shape[0], y2)
        if x2 > x1 and y2 > y1:
            fused[y1:y2, x1:x2] = blocked_value
    return fused


def bbox_overlaps_center(bbox, image_width):
    """Return True when a detection overlaps the center navigation region."""
    x1, _, x2, _ = bbox
    center_left = image_width / 3
    center_right = 2 * image_width / 3
    return not (x2 < center_left or x1 > center_right)


def is_center_blocked(obstacles, image_width):
    """Return True when any obstacle overlaps the center region."""
    return any(bbox_overlaps_center(obstacle["bbox"], image_width) for obstacle in obstacles)


def geometry_decision(scores, center_blocked):
    """Return the geometry-only direction from region scores."""
    direction = min(scores, key=scores.get)
    if center_blocked and direction == "center":
        direction = "left" if scores["left"] < scores["right"] else "right"
    return direction


def build_guidance(clearance, center_blocked, signage_hits, proposed_direction):
    """Turn geometry and signage cues into a short assistive guidance phrase."""
    best_region = max(clearance, key=clearance.get)
    center_clearance = float(clearance.get("center", 0.0))
    best_clearance = float(clearance.get(best_region, 0.0))
    second_best_clearance = max(
        (float(value) for region, value in clearance.items() if region != best_region),
        default=0.0,
    )
    clearance_margin = best_clearance - second_best_clearance

    guidance_text = "clear path ahead, continue forward"
    spoken_label = "clear path ahead"
    guidance_type = "clear_path"

    if signage_hits:
        primary_hit = signage_hits[0]
        sign_text = primary_hit["text"]
        sign_direction = primary_hit.get("direction")
        if "stair" in sign_text:
            if sign_direction in {"left", "right"}:
                guidance_text = f"stairs sign ahead, keep {sign_direction}"
                spoken_label = f"stairs ahead, keep {sign_direction}"
            else:
                guidance_text = "stairs sign ahead, follow the stairs"
                spoken_label = "signage ahead, follow the stairs"
        elif "exit" in sign_text:
            if sign_direction in {"left", "right"}:
                guidance_text = f"exit sign ahead, keep {sign_direction}"
                spoken_label = f"exit ahead, keep {sign_direction}"
            else:
                guidance_text = "exit sign ahead, continue toward it"
                spoken_label = "exit sign ahead"
        else:
            if sign_direction in {"left", "right"}:
                guidance_text = f"signage ahead, keep {sign_direction}"
                spoken_label = f"signage ahead, keep {sign_direction}"
            else:
                guidance_text = f"signage ahead: {sign_text}"
                spoken_label = "signage ahead"
        guidance_type = "signage"
    elif center_blocked or center_clearance < 0.28:
        if best_region in {"left", "right"}:
            guidance_text = f"obstacle ahead, move {best_region}"
            spoken_label = f"obstacle ahead, move {best_region}"
        else:
            guidance_text = "obstacle ahead, proceed carefully"
            spoken_label = "obstacle ahead"
        guidance_type = "avoidance"
    elif (
        center_clearance < 0.36
        and best_region in {"left", "right"}
        and clearance_margin > 0.08
    ):
        guidance_text = f"path narrows ahead, keep {best_region}"
        spoken_label = f"path narrows, keep {best_region}"
        guidance_type = "clearance"
    elif (
        proposed_direction in {"left", "right"}
        and best_region in {"left", "right"}
        and center_clearance < 0.30
        and clearance_margin > 0.12
    ):
        guidance_text = f"more space on the {best_region}, keep {best_region}"
        spoken_label = f"more space on the {best_region}"
        guidance_type = "clearance"
    else:
        best_region = "center"

    return {
        "type": guidance_type,
        "text": guidance_text,
        "spoken_label": spoken_label,
        "best_region": best_region,
        "center_clearance": center_clearance,
        "clearance_margin": clearance_margin,
        "region_clearance": {region: float(value) for region, value in clearance.items()},
        "signage_hits": signage_hits,
    }
