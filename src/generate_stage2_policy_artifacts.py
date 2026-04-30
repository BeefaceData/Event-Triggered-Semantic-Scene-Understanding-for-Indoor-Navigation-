"""Generate Stage 2 fixed-control policy comparison artifacts without extra deps."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


POLICY_FILES = {
    "geometry_only": OUTPUT_ROOT / "val" / "closed_loop" / "closed_loop_eval_val_geometry_only.json",
    "always_semantic": OUTPUT_ROOT / "val" / "closed_loop" / "closed_loop_eval_val_always_semantic.json",
    "event_triggered": OUTPUT_ROOT / "val" / "closed_loop" / "closed_loop_eval_val.json",
}

POLICY_COLORS = {
    "geometry_only": "#2f6b3b",
    "always_semantic": "#9a3d2e",
    "event_triggered": "#2e5f9a",
}


def load_policy_results() -> dict[str, dict]:
    return {policy: json.loads(path.read_text()) for policy, path in POLICY_FILES.items()}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def metric_value(summary: dict, metric: str) -> float:
    return float(summary.get(metric, 0.0))


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #1f2937; }',
        '.title { font-size: 20px; font-weight: bold; }',
        '.label { font-size: 13px; }',
        '.small { font-size: 11px; fill: #4b5563; }',
        '.axis { stroke: #9ca3af; stroke-width: 1; }',
        '</style>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>"]) + "\n")


def generate_summary_bar_chart(data: dict[str, dict], output_path: Path) -> None:
    metrics = [
        ("mean_net_displacement_m", "Net Displacement (m)"),
        ("mean_progress_efficiency", "Progress Efficiency"),
        ("mean_proposed_latency_ms", "Proposed Latency (ms)"),
        ("mean_semantic_calls_per_episode", "Semantic Calls / Episode"),
    ]
    width = 1200
    height = 760
    margin_left = 220
    margin_top = 80
    chart_width = 900
    row_height = 145
    bar_height = 26
    group_gap = 14

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="{margin_left}" y="40">Stage 2 Fixed-Control Policy Comparison</text>')
    lines.append(
        f'<text class="small" x="{margin_left}" y="60">Higher is better except latency and semantic calls, which are lower-is-better.</text>'
    )

    for idx, (metric, label) in enumerate(metrics):
        y_base = margin_top + idx * row_height
        values = {policy: metric_value(result["summary"], metric) for policy, result in data.items()}
        max_value = max(values.values()) if values else 1.0
        if max_value <= 0:
            max_value = 1.0
        lines.append(f'<text class="label" x="20" y="{y_base + 18}">{label}</text>')
        lines.append(
            f'<line class="axis" x1="{margin_left}" y1="{y_base + 90}" x2="{margin_left + chart_width}" y2="{y_base + 90}"/>'
        )

        for policy_index, policy in enumerate(["geometry_only", "always_semantic", "event_triggered"]):
            bar_y = y_base + 28 + policy_index * (bar_height + group_gap)
            bar_w = chart_width * (values[policy] / max_value)
            color = POLICY_COLORS[policy]
            lines.append(
                f'<rect x="{margin_left}" y="{bar_y}" width="{bar_w:.1f}" height="{bar_height}" fill="{color}" rx="4"/>'
            )
            lines.append(
                f'<text class="label" x="{margin_left - 10}" y="{bar_y + 18}" text-anchor="end">{policy}</text>'
            )
            lines.append(
                f'<text class="small" x="{margin_left + bar_w + 8:.1f}" y="{bar_y + 18}">{values[policy]:.2f}</text>'
            )

    write_svg(output_path, lines)


def classify_scene(delta_displacement: float) -> str:
    if delta_displacement >= 0.25:
        return "helps"
    if delta_displacement <= -0.25:
        return "hurts"
    return "neutral"


def generate_scene_delta_chart(data: dict[str, dict], output_path: Path) -> list[dict]:
    geometry_eps = {ep["scene_id"]: ep for ep in data["geometry_only"]["episodes"]}
    event_eps = {ep["scene_id"]: ep for ep in data["event_triggered"]["episodes"]}
    rows = []
    for scene_id in sorted(geometry_eps):
        geo = geometry_eps[scene_id]
        evt = event_eps[scene_id]
        delta_disp = evt["net_displacement_m"] - geo["net_displacement_m"]
        delta_eff = evt["progress_efficiency"] - geo["progress_efficiency"]
        rows.append(
            {
                "scene_id": scene_id,
                "geometry_net_displacement_m": geo["net_displacement_m"],
                "event_net_displacement_m": evt["net_displacement_m"],
                "delta_net_displacement_m": delta_disp,
                "geometry_progress_efficiency": geo["progress_efficiency"],
                "event_progress_efficiency": evt["progress_efficiency"],
                "delta_progress_efficiency": delta_eff,
                "event_semantic_calls": evt["semantic_calls"],
                "classification": classify_scene(delta_disp),
            }
        )

    rows.sort(key=lambda row: row["delta_net_displacement_m"])
    width = 1100
    height = 40 + 30 * len(rows) + 70
    center_x = 580
    scale = 140.0

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="40" y="32">Event-Triggered vs Geometry-Only: Per-Scene Net Displacement Delta</text>')
    lines.append(
        '<text class="small" x="40" y="54">Positive means event-triggered moved farther. Negative means geometry-only moved farther.</text>'
    )
    lines.append(f'<line class="axis" x1="{center_x}" y1="70" x2="{center_x}" y2="{height - 30}"/>')
    lines.append(f'<text class="small" x="{center_x - 6}" y="68" text-anchor="end">0</text>')

    for index, row in enumerate(rows):
        y = 90 + index * 30
        value = row["delta_net_displacement_m"]
        bar_width = abs(value) * scale
        if value >= 0:
            x = center_x
            color = "#2e5f9a"
        else:
            x = center_x - bar_width
            color = "#2f6b3b"
        lines.append(f'<text class="label" x="40" y="{y + 14}">{row["scene_id"]}</text>')
        lines.append(
            f'<rect x="{x:.1f}" y="{y}" width="{max(bar_width, 1):.1f}" height="18" fill="{color}" rx="3"/>'
        )
        lines.append(
            f'<text class="small" x="{center_x + 260}" y="{y + 14}">{value:+.2f} m | {row["classification"]} | calls {row["event_semantic_calls"]}</text>'
        )

    write_svg(output_path, lines)
    return rows


def write_scene_markdown(rows: list[dict], output_path: Path) -> None:
    helps = [row for row in rows if row["classification"] == "helps"]
    hurts = [row for row in rows if row["classification"] == "hurts"]
    neutral = [row for row in rows if row["classification"] == "neutral"]

    lines = [
        "# Stage 2 Scene-by-Scene Comparison",
        "",
        "Comparison: `event_triggered` vs `geometry_only` using net displacement as the primary directional outcome.",
        "",
        f"- Helps scenes: `{len(helps)}`",
        f"- Hurts scenes: `{len(hurts)}`",
        f"- Neutral scenes: `{len(neutral)}`",
        "",
        "## Largest Event-Triggered Gains",
        "",
        "| Scene | Delta Net Displacement (m) | Delta Progress Efficiency | Semantic Calls |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(helps, key=lambda item: item["delta_net_displacement_m"], reverse=True)[:5]:
        lines.append(
            f"| {row['scene_id']} | {row['delta_net_displacement_m']:+.2f} | {row['delta_progress_efficiency']:+.3f} | {row['event_semantic_calls']} |"
        )

    lines.extend(
        [
            "",
            "## Largest Geometry-Only Advantages",
            "",
            "| Scene | Delta Net Displacement (m) | Delta Progress Efficiency | Semantic Calls |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(hurts, key=lambda item: item["delta_net_displacement_m"])[:5]:
        lines.append(
            f"| {row['scene_id']} | {row['delta_net_displacement_m']:+.2f} | {row['delta_progress_efficiency']:+.3f} | {row['event_semantic_calls']} |"
        )

    lines.extend(
        [
            "",
            "## Full Scene Table",
            "",
            "| Scene | Delta Net Displacement (m) | Delta Progress Efficiency | Semantic Calls | Classification |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['scene_id']} | {row['delta_net_displacement_m']:+.2f} | {row['delta_progress_efficiency']:+.3f} | {row['event_semantic_calls']} | {row['classification']} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    data = load_policy_results()
    plots_dir = OUTPUT_ROOT / "plots"
    ensure_dir(plots_dir)

    summary_plot = plots_dir / "stage2_policy_comparison_summary.svg"
    scene_delta_plot = plots_dir / "stage2_event_vs_geometry_scene_delta.svg"
    scene_markdown = plots_dir / "stage2_event_vs_geometry_scene_analysis.md"

    generate_summary_bar_chart(data, summary_plot)
    rows = generate_scene_delta_chart(data, scene_delta_plot)
    write_scene_markdown(rows, scene_markdown)

    print(f"[INFO] Wrote {summary_plot}")
    print(f"[INFO] Wrote {scene_delta_plot}")
    print(f"[INFO] Wrote {scene_markdown}")


if __name__ == "__main__":
    main()
