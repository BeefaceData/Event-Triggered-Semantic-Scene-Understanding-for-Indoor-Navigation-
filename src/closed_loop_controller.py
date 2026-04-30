"""Simple stateful controller for Stage 2 closed-loop rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field

from action_mapping import ActionMappingConfig, direction_to_action


@dataclass
class ClosedLoopControllerConfig:
    """Controller settings for the first embodied prototype."""

    max_consecutive_turns: int = 2
    stuck_action: str = "turn_right"
    collision_like_distance_threshold_m: float = 0.05
    max_stalled_forward_steps: int = 2
    recovery_turn_steps: int = 3
    guidance_change_confirmation_steps: int = 2
    clear_path_recovery_confirmation_steps: int = 3
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)


@dataclass
class ClosedLoopControllerState:
    """Track short-horizon action history to damp simple oscillations."""

    previous_action: str | None = None
    consecutive_turns: int = 0
    consecutive_same_turns: int = 0
    last_turn_action: str | None = None
    stalled_forward_steps: int = 0
    recovery_turns_remaining: int = 0
    recovery_turn_action: str | None = None
    next_recovery_turn_action: str = "turn_right"
    stable_guidance: dict | None = None
    pending_guidance: dict | None = None
    pending_guidance_count: int = 0
    total_steps: int = 0


class ClosedLoopController:
    """Translate Stage 1 decisions into a stable embodied action stream."""

    def __init__(self, config: ClosedLoopControllerConfig | None = None):
        self.config = config or ClosedLoopControllerConfig()
        self.state = ClosedLoopControllerState()

    def reset(self) -> None:
        """Reset controller state between episodes."""
        self.state = ClosedLoopControllerState()

    def _guidance_equals(self, first: dict | None, second: dict | None) -> bool:
        """Return True when two guidance messages should be treated as the same."""
        if first is None or second is None:
            return first is second
        return (
            first.get("type") == second.get("type")
            and first.get("text") == second.get("text")
        )

    def _movement_override_guidance(self) -> dict | None:
        """Prefer realized motion over raw depth guidance after stalled forward steps."""
        if self.state.stalled_forward_steps <= 0:
            return None

        if self.state.stalled_forward_steps >= self.config.max_stalled_forward_steps:
            return {
                "type": "recovery",
                "text": "obstacle ahead, adjust course",
                "spoken_label": "obstacle ahead, adjust course",
                "best_region": "center",
                "center_clearance": 0.0,
                "clearance_margin": 0.0,
                "region_clearance": {},
                "signage_hits": [],
            }

        return {
            "type": "recovery",
            "text": "possible obstacle ahead, adjusting course",
            "spoken_label": "possible obstacle ahead",
            "best_region": "center",
            "center_clearance": 0.0,
            "clearance_margin": 0.0,
            "region_clearance": {},
            "signage_hits": [],
        }

    def _stabilize_guidance(self, record: dict) -> dict:
        """Apply temporal smoothing so guidance does not oscillate frame to frame."""
        raw_guidance = dict(record.get("guidance", {}))
        override_guidance = self._movement_override_guidance()
        target_guidance = override_guidance or raw_guidance
        if not target_guidance:
            target_guidance = {
                "type": "clear_path",
                "text": "clear path ahead, continue forward",
                "spoken_label": "clear path ahead",
                "best_region": "center",
                "center_clearance": 0.0,
                "clearance_margin": 0.0,
                "region_clearance": {},
                "signage_hits": [],
            }

        stable_guidance = self.state.stable_guidance
        if stable_guidance is None:
            stable_guidance = dict(target_guidance)
            self.state.stable_guidance = stable_guidance
            self.state.pending_guidance = None
            self.state.pending_guidance_count = 0
        elif self._guidance_equals(target_guidance, stable_guidance):
            self.state.pending_guidance = None
            self.state.pending_guidance_count = 0
        else:
            if self._guidance_equals(target_guidance, self.state.pending_guidance):
                self.state.pending_guidance_count += 1
            else:
                self.state.pending_guidance = dict(target_guidance)
                self.state.pending_guidance_count = 1

            required_count = (
                self.config.clear_path_recovery_confirmation_steps
                if target_guidance.get("type") == "clear_path"
                else self.config.guidance_change_confirmation_steps
            )
            if self.state.pending_guidance_count >= required_count:
                stable_guidance = dict(target_guidance)
                self.state.stable_guidance = stable_guidance
                self.state.pending_guidance = None
                self.state.pending_guidance_count = 0

        smoothed_guidance = dict(self.state.stable_guidance or target_guidance)
        smoothed_guidance["raw_text"] = raw_guidance.get("text")
        smoothed_guidance["raw_type"] = raw_guidance.get("type")
        smoothed_guidance["movement_override"] = override_guidance is not None
        record["guidance"] = smoothed_guidance
        return smoothed_guidance

    def observe_transition(self, action: str, moved_distance_m: float) -> None:
        """Update controller state with the realized outcome of the last action."""
        if action == self.config.action_mapping.forward_action:
            if moved_distance_m < self.config.collision_like_distance_threshold_m:
                self.state.stalled_forward_steps += 1
                if self.state.stalled_forward_steps >= self.config.max_stalled_forward_steps:
                    self.state.recovery_turn_action = self.state.next_recovery_turn_action
                    self.state.recovery_turns_remaining = self.config.recovery_turn_steps
                    self.state.next_recovery_turn_action = (
                        self.config.action_mapping.left_action
                        if self.state.next_recovery_turn_action == self.config.action_mapping.right_action
                        else self.config.action_mapping.right_action
                    )
            else:
                self.state.stalled_forward_steps = 0
                self.state.recovery_turns_remaining = 0
        else:
            if moved_distance_m > self.config.collision_like_distance_threshold_m:
                self.state.stalled_forward_steps = 0

    def select_action(self, record: dict) -> str:
        """Choose the next simulator action from a Stage 1 frame record."""
        guidance = self._stabilize_guidance(record)
        direction = record["proposed"]["decision"]
        action = direction_to_action(direction, self.config.action_mapping)
        center_blocked = bool(record.get("center_blocked", False))
        triggered = bool(record.get("trigger", {}).get("triggered", False))
        entropy = float(record.get("trigger", {}).get("entropy", 0.0))
        relative_separability = float(
            record.get("trigger", {}).get("relative_separability", 0.0)
        )

        if self.state.recovery_turns_remaining > 0 and self.state.recovery_turn_action is not None:
            action = self.state.recovery_turn_action
            self.state.recovery_turns_remaining -= 1

        if action in {"turn_left", "turn_right"}:
            self.state.consecutive_turns += 1
            if self.state.last_turn_action == action:
                self.state.consecutive_same_turns += 1
            else:
                self.state.consecutive_same_turns = 1
            self.state.last_turn_action = action
        else:
            self.state.consecutive_turns = 0
            self.state.consecutive_same_turns = 0

        # If the controller keeps turning for too long, force forward progress
        # once before allowing another burst of turns.
        if self.state.consecutive_turns > self.config.max_consecutive_turns:
            action = self.config.action_mapping.forward_action
            self.state.consecutive_turns = 0
            self.state.consecutive_same_turns = 0

        # If the policy asks for the same turn repeatedly and the center is not
        # blocked, move forward once to test the updated heading instead of
        # spinning in place.
        if (
            action in {"turn_left", "turn_right"}
            and self.state.previous_action == action
            and not center_blocked
            and self.state.recovery_turns_remaining == 0
        ):
            action = self.config.action_mapping.forward_action
            self.state.consecutive_turns = 0
            self.state.consecutive_same_turns = 0

        # When the scene is not triggered and the center is open, prefer
        # forward motion over incremental turning.
        if (
            action in {"turn_left", "turn_right"}
            and not triggered
            and not center_blocked
            and entropy < 0.95
            and relative_separability > 0.18
            and self.state.recovery_turns_remaining == 0
        ):
            action = self.config.action_mapping.forward_action
            self.state.consecutive_turns = 0
            self.state.consecutive_same_turns = 0

        # If the previous forward move stalled, trust realized motion more than
        # optimistic depth and turn away before trying forward again.
        if (
            action == self.config.action_mapping.forward_action
            and self.state.stalled_forward_steps > 0
            and self.state.recovery_turns_remaining == 0
        ):
            action = self.state.next_recovery_turn_action
            self.state.consecutive_turns += 1
            self.state.consecutive_same_turns = 1
            self.state.last_turn_action = action

        # If a recovery window is active, keep using that turn action instead of
        # immediately forcing forward motion again.
        if self.state.recovery_turns_remaining > 0 and self.state.recovery_turn_action is not None:
            action = self.state.recovery_turn_action
            self.state.consecutive_turns = 1
            self.state.consecutive_same_turns = 1
            self.state.last_turn_action = action

        self.state.previous_action = action
        self.state.total_steps += 1
        return action
