"""Aggregate per-run metrics CSV files into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs" / "experiments"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "aggregated_metrics.csv"


def read_metrics_rows(metrics_path: Path) -> list[dict[str, str]]:
    """Read rows from a run's metrics.csv file."""
    with metrics_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def discover_metrics_files(root: Path) -> list[Path]:
    """Find metrics.csv files inside experiment folders."""
    return sorted(root.glob("*/metrics.csv"))


def normalize_row(row: dict[str, str], run_dir: Path) -> dict[str, str]:
    """Attach experiment-folder metadata to a metrics row."""
    normalized = dict(row)
    normalized["run_folder"] = run_dir.name
    normalized["run_path"] = str(run_dir)
    return normalized


def collect_rows(root: Path) -> list[dict[str, str]]:
    """Collect normalized rows from all experiment metric files."""
    rows: list[dict[str, str]] = []
    for metrics_path in discover_metrics_files(root):
        run_dir = metrics_path.parent
        for row in read_metrics_rows(metrics_path):
            rows.append(normalize_row(row, run_dir))
    return rows


def write_aggregated_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write the aggregated comparison CSV."""
    fieldnames = [
        "run_folder",
        "run_path",
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics from experiment run folders.")
    parser.add_argument(
        "--experiments-root",
        default=str(EXPERIMENTS_ROOT),
        help="Directory containing per-run experiment folders",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the aggregated CSV",
    )
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    output_path = Path(args.output)

    rows = collect_rows(experiments_root)
    write_aggregated_csv(rows, output_path)

    print(f"[INFO] Aggregated {len(rows)} metric rows")
    print(f"[INFO] Output written to {output_path}")


if __name__ == "__main__":
    main()
