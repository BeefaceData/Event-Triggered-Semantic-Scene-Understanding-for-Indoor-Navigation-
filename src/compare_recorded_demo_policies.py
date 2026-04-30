"""Compare two recorded-demo JSON artifacts and summarize qualitative differences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two recorded demo JSON outputs.")
    parser.add_argument("--a", required=True, help="Path to first recorded demo JSON.")
    parser.add_argument("--b", required=True, help="Path to second recorded demo JSON.")
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional markdown output path. Defaults next to the first JSON.",
    )
    return parser.parse_args()


def load_json(path_str: str) -> dict:
    path = Path(path_str).expanduser().resolve()
    return json.loads(path.read_text())


def summarize(data: dict) -> dict:
    records = data.get("records", [])
    reasons = Counter()
    guidance = Counter()
    directions = Counter()
    for record in records:
        reasons.update(record.get("trigger_reasons", []))
        guide = record.get("guidance", {}).get("text")
        if guide:
            guidance[guide] += 1
        directions[record.get("final_direction")] += 1

    return {
        "semantic_policy": data.get("semantic_policy"),
        "num_frames": data.get("num_frames", 0),
        "num_triggered_frames": data.get("num_triggered_frames", 0),
        "num_semantic_calls": data.get("num_semantic_calls", 0),
        "trigger_rate": (data.get("num_triggered_frames", 0) / max(data.get("num_frames", 1), 1)) * 100.0,
        "semantic_rate": (data.get("num_semantic_calls", 0) / max(data.get("num_frames", 1), 1)) * 100.0,
        "top_reasons": reasons.most_common(5),
        "top_guidance": guidance.most_common(5),
        "direction_counts": dict(directions),
        "path": data.get("input_video_path"),
        "video_output": data.get("output_video_path"),
    }


def comparison_markdown(first: dict, second: dict) -> str:
    lines = [
        "# Recorded Demo Policy Comparison",
        "",
        f"Input video: `{first['path']}`",
        "",
        f"- Policy A: `{first['semantic_policy']}`",
        f"- Policy B: `{second['semantic_policy']}`",
        "",
        "## Aggregate Comparison",
        "",
        "| Metric | Policy A | Policy B |",
        "|---|---:|---:|",
        f"| Trigger rate | {first['trigger_rate']:.2f}% | {second['trigger_rate']:.2f}% |",
        f"| Semantic call rate | {first['semantic_rate']:.2f}% | {second['semantic_rate']:.2f}% |",
        f"| Triggered frames | {first['num_triggered_frames']} | {second['num_triggered_frames']} |",
        f"| Semantic calls | {first['num_semantic_calls']} | {second['num_semantic_calls']} |",
        "",
        "## Most Common Guidance",
        "",
        f"Policy A `{first['semantic_policy']}`:",
    ]
    for text, count in first["top_guidance"]:
        lines.append(f"- `{text}`: `{count}` frames")
    lines.append("")
    lines.append(f"Policy B `{second['semantic_policy']}`:")
    for text, count in second["top_guidance"]:
        lines.append(f"- `{text}`: `{count}` frames")

    lines.extend(["", "## Top Trigger Reasons", "", f"Policy A `{first['semantic_policy']}`:"])
    for text, count in first["top_reasons"]:
        lines.append(f"- `{text}`: `{count}`")
    lines.append("")
    lines.append(f"Policy B `{second['semantic_policy']}`:")
    for text, count in second["top_reasons"]:
        lines.append(f"- `{text}`: `{count}`")

    lines.extend(
        [
            "",
            "## Direction Counts",
            "",
            f"- Policy A: `{first['direction_counts']}`",
            f"- Policy B: `{second['direction_counts']}`",
            "",
            "## Output Videos",
            "",
            f"- Policy A video: `{first['video_output']}`",
            f"- Policy B video: `{second['video_output']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    first = summarize(load_json(args.a))
    second = summarize(load_json(args.b))
    output_path = (
        Path(args.output_md).expanduser().resolve()
        if args.output_md
        else Path(args.a).expanduser().resolve().with_name("recorded_demo_policy_comparison.md")
    )
    output_path.write_text(comparison_markdown(first, second))
    print(f"[INFO] Wrote recorded demo comparison to {output_path}")


if __name__ == "__main__":
    main()
