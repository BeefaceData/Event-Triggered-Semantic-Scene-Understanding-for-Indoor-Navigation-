"""Create timestamped experiment folders and tracking templates."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def slugify(value: str) -> str:
    """Convert a run name into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "run"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content)


def write_metrics_csv(path: Path) -> None:
    if path.exists():
        return

    fieldnames = [
        "date",
        "run_name",
        "stage",
        "dataset",
        "num_scenes",
        "num_frames",
        "trigger_rate",
        "geometry_latency_ms",
        "blip_latency_ms",
        "proposed_latency_ms",
        "manual_review_accuracy",
        "notes",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def build_summary_template(run_name: str, stage: str, goal: str, created_at: str) -> str:
    return f"""# Experiment Summary

## Run Info

- Run name: {run_name}
- Stage: {stage}
- Created: {created_at}
- Goal: {goal or "Fill this in"}

## Environment

- Conda / Python environment:
- Device:
- GPU available:

## Dataset

- Dataset root:
- Scenes used:
- Semantic assets available:

## Commands

- `python ...`

## Outputs

- JSON:
- CSV:
- Images:
- Videos:

## Headline Metrics

- Trigger rate:
- Geometry latency:
- BLIP latency:
- Proposed latency:
- Manual review accuracy:

## Observations

- What worked:
- What failed:
- What surprised you:

## Next Actions

- 
"""


def build_steps_template(created_at: str) -> str:
    return f"""# Run Step Log

Created: {created_at}

Use this file to track each meaningful step during the run.

## Step Template

### Time

`HH:MM`

### Action

What did you do?

### Command

`python ...`

### Result

What happened?

### Files Produced

- 

### Notes

- 
"""


def build_visuals_template() -> str:
    return """# Visual Checklist

- [ ] Annotated RGB screenshot saved
- [ ] RGB + depth screenshot saved
- [ ] Trigger-off example saved
- [ ] Trigger-on example saved
- [ ] Ambiguous-scene example saved
- [ ] Semantic-cue example saved
- [ ] Demo video saved

## Stored Files

- 
"""


def build_results_summary_template() -> str:
    return """# Results Summary

Use this file to capture the headline outcome of the run after it finishes.

## Run Snapshot

- Run name:
- Stage:
- Date:
- Status:

## Setup Summary

- Environment:
- Device:
- Dataset:
- Scenes:

## Main Metrics

- Trigger rate:
- Geometry latency:
- OCR latency:
- BLIP latency:
- Proposed latency:
- Manual review accuracy:

## Comparison Notes

- Geometry-only:
- Semantics-always:
- Event-triggered:

## Best Examples

- 

## Important Failures

- 

## Key Interpretation

- 

## Reporting-Ready Summary

Write 3-5 sentences that you could reuse directly in a progress report, meeting update, or presentation.
"""


def build_failure_notes_template() -> str:
    return """# Failure Notes

Use this file for run-specific failures.

## Failure Template

### Item

- Category:
- Scene / source:
- Frame / time:

### What Went Wrong

- 

### Likely Cause

- 

### Evidence

- screenshot:
- video:
- record file:

### Proposed Next Step

- 
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tracking folder for a new experiment run.")
    parser.add_argument("--name", required=True, help="Human-readable run name")
    parser.add_argument(
        "--stage",
        default="stage1",
        choices=["stage1", "stage2"],
        help="Project stage for this run",
    )
    parser.add_argument(
        "--goal",
        default="",
        help="Short description of what this run is trying to achieve",
    )
    args = parser.parse_args()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / "experiments" / f"{stamp}_{slugify(args.name)}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "failures")

    write_text_if_missing(
        run_dir / "README.md",
        build_summary_template(args.name, args.stage, args.goal, created_at),
    )
    write_text_if_missing(run_dir / "results_summary.md", build_results_summary_template())
    write_text_if_missing(run_dir / "steps.md", build_steps_template(created_at))
    write_text_if_missing(run_dir / "visuals.md", build_visuals_template())
    write_text_if_missing(run_dir / "failures" / "README.md", build_failure_notes_template())
    write_metrics_csv(run_dir / "metrics.csv")

    print(f"[INFO] Created experiment folder: {run_dir}")
    print("[INFO] Files created:")
    print(f"  - {run_dir / 'README.md'}")
    print(f"  - {run_dir / 'results_summary.md'}")
    print(f"  - {run_dir / 'steps.md'}")
    print(f"  - {run_dir / 'visuals.md'}")
    print(f"  - {run_dir / 'failures' / 'README.md'}")
    print(f"  - {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
