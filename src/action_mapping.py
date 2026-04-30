"""Map Stage 1 directional outputs into embodied navigation actions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionMappingConfig:
    """Configuration for translating local directions into discrete actions."""

    forward_action: str = "move_forward"
    left_action: str = "turn_left"
    right_action: str = "turn_right"
    stop_action: str = "stop"


def direction_to_action(direction: str | None, config: ActionMappingConfig | None = None) -> str:
    """Convert a Stage 1 direction into a simulator action string."""
    mapping = config or ActionMappingConfig()

    if direction == "left":
        return mapping.left_action
    if direction == "right":
        return mapping.right_action
    if direction == "center":
        return mapping.forward_action
    if direction == "stop":
        return mapping.stop_action
    return mapping.stop_action
