"""Optional spoken audio for rollout and recorded-demo videos."""

from __future__ import annotations

import subprocess
from pathlib import Path


def shell_quote(text: str) -> str:
    """Quote text for ffmpeg filter arguments."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


def is_meaningful_guidance_event(guidance: dict) -> bool:
    """Return True when guidance is important enough to narrate."""
    guidance_type = str(guidance.get("type") or "").strip().lower()
    text = str(guidance.get("text") or "").strip().lower()
    if not guidance_type and not text:
        return False
    if guidance_type in {"avoidance", "signage", "recovery"}:
        return True
    if guidance_type == "clearance":
        return True
    if guidance_type == "clear_path":
        return False
    if "clear path" in text:
        return False
    return False


def narration_event_key(record: dict, mode: str) -> str:
    """Create a stable event key so repeated states are not re-spoken."""
    if mode == "action":
        return narration_label(record, mode)

    guidance = record.get("guidance", {}) or {}
    guidance_type = str(guidance.get("type") or "").strip().lower()
    spoken_label = str(guidance.get("spoken_label") or "").strip().lower()
    final_direction = str(record.get("final_direction") or "").strip().lower()
    trigger_reasons = tuple(sorted(str(reason) for reason in record.get("trigger_reasons", [])))

    if guidance_type == "signage":
        return f"signage:{spoken_label}"
    if guidance_type == "recovery":
        return f"recovery:{final_direction}:{spoken_label}"
    if guidance_type in {"avoidance", "clearance"}:
        return f"{guidance_type}:{final_direction}:{spoken_label}"
    if record.get("triggered"):
        return f"trigger:{final_direction}:{','.join(trigger_reasons)}"
    return ""


def narration_label(record: dict, mode: str) -> str:
    """Choose the spoken label for a rollout step."""
    if mode == "action":
        action = record.get("action", "")
        return {
            "move_forward": "forward",
            "turn_left": "turn left",
            "turn_right": "turn right",
            "stop": "stop",
        }.get(action, action.replace("_", " "))

    guidance = record.get("guidance", {}) or {}
    spoken_label = guidance.get("spoken_label")
    if spoken_label and is_meaningful_guidance_event(guidance):
        return str(spoken_label)

    if record.get("triggered"):
        direction = str(record.get("final_direction") or "").strip().lower()
        if direction == "left":
            return "uncertain path, keep left"
        if direction == "right":
            return "uncertain path, keep right"
        if direction == "center":
            return "uncertain path, continue carefully"

    direction = record.get("final_direction") or ""
    if str(direction).strip().lower() == "stop":
        return "stop"
    return ""


def build_narration_events(records: list[dict], fps: int, mode: str) -> list[tuple[float, str]]:
    """Create timestamped narration events, speaking only meaningful state changes."""
    events = []
    previous_event_key = None
    for record in records:
        event_key = narration_event_key(record, mode)
        label = narration_label(record, mode)
        if not label or not event_key or event_key == previous_event_key:
            continue
        step_index = int(record.get("step_index", record.get("processed_index", len(events))))
        timestamp_s = float(record.get("time_s", step_index / max(fps, 1)))
        events.append((timestamp_s, label))
        previous_event_key = event_key
    return events


def write_concat_file(lines: list[str], path: Path) -> None:
    """Write an ffmpeg concat demuxer file."""
    path.write_text("\n".join(lines) + "\n")


def run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg and raise a readable error if it fails."""
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_spoken_audio(
    video_path: Path,
    records: list[dict],
    fps: int,
    mode: str = "direction",
    output_video_path: Path | None = None,
) -> Path | None:
    """Generate a spoken narration track and mux it into a new video file."""
    events = build_narration_events(records, fps=fps, mode=mode)
    if not events:
        return None

    work_dir = video_path.parent / f"{video_path.stem}_audio_parts"
    work_dir.mkdir(parents=True, exist_ok=True)
    concat_entries: list[str] = []
    total_duration_s = max(len(records) / max(fps, 1), 0.1)

    def add_silence(duration_s: float, index: int) -> None:
        if duration_s <= 0:
            return
        silence_path = work_dir / f"{index:04d}_silence.wav"
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=22050:cl=mono",
                "-t",
                f"{duration_s:.3f}",
                str(silence_path),
            ]
        )
        concat_entries.append(f"file '{silence_path.name}'")

    previous_time = 0.0
    segment_index = 0
    for timestamp_s, label in events:
        add_silence(timestamp_s - previous_time, segment_index)
        segment_index += 1
        speech_path = work_dir / f"{segment_index:04d}_speech.wav"
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"flite=text='{shell_quote(label)}':voice=slt",
                str(speech_path),
            ]
        )
        concat_entries.append(f"file '{speech_path.name}'")
        previous_time = timestamp_s + (1.0 / max(fps, 1))
        segment_index += 1

    add_silence(total_duration_s - previous_time, segment_index)

    concat_path = work_dir / "concat.txt"
    write_concat_file(concat_entries, concat_path)
    narration_wav = video_path.with_name(f"{video_path.stem}_narration.wav")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(narration_wav),
        ]
    )

    narrated_video = output_video_path or video_path.with_name(f"{video_path.stem}_with_audio.mp4")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(narration_wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(narrated_video),
        ]
    )
    return narrated_video
