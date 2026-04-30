"""Export a lightweight CSV for manual HM3D review."""

import argparse
import csv
import json
from pathlib import Path

from hm3d_dataset import split_manual_review_csv, split_results_json


def parse_args():
    parser = argparse.ArgumentParser(description="Export manual review CSV from HM3D results.")
    parser.add_argument(
        "--split",
        default="example",
        choices=["example", "minival", "val", "train"],
        help="HM3D split whose results should be exported.",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional explicit path to a results JSON file.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit path for the exported CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_json = (
        split_results_json(args.split)
        if args.results_json is None
        else Path(args.results_json).expanduser().resolve()
    )
    output_csv_path = (
        split_manual_review_csv(args.split)
        if args.output_csv is None
        else Path(args.output_csv).expanduser().resolve()
    )
    data = json.loads(results_json.read_text())
    rows = data.get("review_records", [])
    fieldnames = [
        "scene_id",
        "frame_index",
        "image_path",
        "semantic_backend",
        "semantic_decision",
        "geometry_decision",
        "blip_decision",
        "proposed_decision",
        "triggered",
        "trigger_reasons",
        "entropy",
        "relative_separability",
        "human_label",
        "notes",
    ]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "trigger_reasons": ",".join(row["trigger_reasons"]),
                    "human_label": "",
                    "notes": "",
                }
            )

    print(f"[INFO] Manual review CSV written to {output_csv_path}")


if __name__ == "__main__":
    main()
