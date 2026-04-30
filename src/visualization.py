"""Visualization helpers for demos and HM3D review outputs."""

import cv2


def annotate_navigation_frame(frame, record):
    """Overlay decisions, trigger state, reasons, and region scores on a frame."""
    vis = frame.copy()
    height, width = vis.shape[:2]

    cv2.line(vis, (width // 3, 0), (width // 3, height), (255, 255, 0), 2)
    cv2.line(vis, (2 * width // 3, 0), (2 * width // 3, height), (255, 255, 0), 2)

    for obstacle in record.get("obstacles", []):
        x1, y1, x2, y2 = map(int, obstacle["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis,
            obstacle["class"],
            (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    final_direction = str(record["proposed"]["decision"]).upper()
    reasons = ",".join(record["trigger"]["reasons"]) if record["trigger"]["reasons"] else "none"

    cv2.putText(vis, f"GO: {final_direction}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"Trigger: {record['trigger']['triggered']} ({record['trigger']['mode']})",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
    )
    cv2.putText(vis, f"Reasons: {reasons}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(
        vis,
        f"Entropy: {record['trigger']['entropy']:.3f} | Sep: {record['trigger']['relative_separability']:.3f}",
        (20, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    guidance = record.get("guidance", {})
    guidance_text = guidance.get("text", "no guidance")
    if guidance.get("movement_override"):
        guidance_text = f"{guidance_text} [motion]"
    signage_hits = (
        guidance.get("signage_hits")
        or record.get("baseline_ocr", {}).get("signage_hits", [])
        or []
    )
    signage_text = signage_hits[0]["text"] if signage_hits else "none"
    cv2.putText(
        vis,
        f"Guide: {guidance_text}",
        (20, 154),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (150, 255, 150),
        2,
    )
    cv2.putText(
        vis,
        f"Signage: {signage_text}",
        (20, 182),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 120) if signage_hits else (180, 180, 180),
        2,
    )

    for index, region in enumerate(["left", "center", "right"]):
        clearance = guidance.get("region_clearance", {}).get(region)
        clearance_text = f"{clearance:.2f}" if clearance is not None else "n/a"
        cv2.putText(
            vis,
            f"{region}: {record['scores'][region]:.3f} | clr {clearance_text}",
            (20 + index * 180, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            2,
        )

    return vis


def stack_rgb_and_depth(rgb_frame, depth_vis):
    """Place annotated RGB next to the depth view for videos and demos."""
    if len(depth_vis.shape) == 2:
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    return cv2.hconcat([rgb_frame, depth_vis])


def annotate_closed_loop_frame(frame, record, step_index, action, triggered_steps):
    """Add Stage 2 rollout overlays on top of the standard navigation view."""
    vis = annotate_navigation_frame(frame, record)
    height = vis.shape[0]
    semantic_backend = record.get("semantic_backend", "n/a")
    geometry_decision = str(record.get("baseline_geometry", {}).get("decision")).upper()
    semantic_decision = record.get("baseline_semantic", {}).get("decision")
    semantic_decision_text = str(semantic_decision).upper() if semantic_decision is not None else "NONE"
    final_decision = str(record.get("proposed", {}).get("decision")).upper()
    semantic_invoked = bool(record.get("semantic_invoked", False))
    override_active = semantic_invoked and semantic_decision is not None and final_decision != geometry_decision
    override_text = "YES" if override_active else "NO"

    cv2.putText(
        vis,
        f"Step: {step_index}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"Action: {action}",
        (20, 238),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"Backend: {semantic_backend}",
        (20, 266),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"Triggered Steps: {triggered_steps}",
        (20, 294),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"Geometry: {geometry_decision} | Semantic: {semantic_decision_text}",
        (20, 322),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"Semantic Invoked: {semantic_invoked} | Override: {override_text}",
        (20, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255) if override_active else (200, 200, 200),
        2,
    )
    cv2.putText(
        vis,
        f"Proposed Latency: {record['proposed']['latency_ms']:.1f} ms",
        (20, height - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    return vis
