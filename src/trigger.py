"""Trigger formulation based on uncertainty and semantic cues."""

import math

from config import (
    ENTROPY_THRESHOLD,
    LEGACY_FREE_PATH_DIFF_THRESHOLD,
    SEPARABILITY_THRESHOLD,
    TEXT_TRIGGER_KEYWORDS,
)


def relative_separability(scores):
    """
    Normalized margin between the best and second-best geometry regions.
    Higher means geometry is more confident.
    Lower means multiple regions are similarly plausible.
    """
    ordered = sorted(scores.values())
    if len(ordered) < 2:
        return 0.0
    numerator = abs(ordered[1] - ordered[0])
    denominator = max(scores.values()) + 1e-8
    return float(numerator / denominator)


def region_probabilities(scores):
    """
    Convert region scores to pseudo-probabilities for entropy analysis.
    Lower score means more navigable, so we invert scores before normalizing.
    """
    inv = {region: 1.0 / (value + 1e-8) for region, value in scores.items()}
    total = sum(inv.values()) + 1e-8
    return {region: value / total for region, value in inv.items()}


def uncertainty_entropy(scores):
    """Entropy over region probabilities."""
    probs = region_probabilities(scores)
    entropy = 0.0
    for value in probs.values():
        entropy -= value * math.log(value + 1e-8)
    return float(entropy)


def trigger_from_signage(frame, reader):
    """
    Return OCR cue hits used as an independent semantic trigger signal.
    """
    from semantics import detect_text_cues

    return detect_text_cues(frame, reader, TEXT_TRIGGER_KEYWORDS)


def legacy_trigger(scores, center_blocked):
    """Original threshold-based trigger kept only for legacy baseline comparisons."""
    ordered = sorted(scores.values())
    reasons = []
    if len(ordered) >= 2:
        diff = abs(ordered[1] - ordered[0])
        if diff < LEGACY_FREE_PATH_DIFF_THRESHOLD:
            reasons.append("multiple_open_paths")
    if center_blocked:
        reasons.append("center_blocked")
    return {"triggered": bool(reasons), "reasons": reasons}


def evaluate_trigger(scores, center_blocked, frame, reader, mode="uncertainty"):
    """
    Evaluate the full trigger and return a structured record.
    This makes later analysis easier than returning only a boolean.
    """
    separability = relative_separability(scores)
    entropy = uncertainty_entropy(scores)
    text_hits = trigger_from_signage(frame, reader)

    reasons = []
    if mode == "uncertainty":
        if separability < SEPARABILITY_THRESHOLD:
            reasons.append("low_separability")
        if entropy > ENTROPY_THRESHOLD:
            reasons.append("high_entropy")
        if center_blocked:
            reasons.append("center_blocked")
        if text_hits:
            reasons.append("semantic_cue")
    elif mode == "legacy":
        reasons = legacy_trigger(scores, center_blocked)["reasons"]
    else:
        raise ValueError(f"Unknown trigger mode: {mode}")

    return {
        "triggered": bool(reasons),
        "mode": mode,
        "reasons": reasons,
        "relative_separability": separability,
        "entropy": entropy,
        "text_hits": text_hits,
    }
