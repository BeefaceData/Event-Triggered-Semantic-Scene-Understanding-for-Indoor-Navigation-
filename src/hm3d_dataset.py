"""Helpers for resolving HM3D dataset splits and output locations."""

from __future__ import annotations

from pathlib import Path

from config import MODULE_ROOT, OUTPUT_ROOT, PROJECT_ROOT


KNOWN_SPLITS = {"example", "minival", "val", "train"}


def candidate_dataset_roots(split: str) -> list[Path]:
    """Return likely on-disk locations for an HM3D split."""
    return [
        MODULE_ROOT / "datasets" / "hm3d" / split,
        PROJECT_ROOT / "datasets" / "hm3d" / split,
    ]


def resolve_dataset_root(split: str, dataset_root: str | None = None) -> Path:
    """Resolve the dataset root for a named HM3D split."""
    if dataset_root is not None:
        return Path(dataset_root).expanduser().resolve()

    for candidate in candidate_dataset_roots(split):
        if candidate.exists():
            return candidate

    # Fall back to the in-folder location so errors stay explicit and local.
    return (MODULE_ROOT / "datasets" / "hm3d" / split).resolve()


def candidate_annotated_configs(dataset_root: Path, split: str) -> list[Path]:
    """Return likely annotated config filenames for a split."""
    names = [
        f"hm3d_annotated_{split}_basis.scene_dataset_config.json",
        f"hm3d_annotated_{split}_example_basis.scene_dataset_config.json",
        "hm3d_annotated_example_basis.scene_dataset_config.json",
        "hm3d_annotated_basis.scene_dataset_config.json",
    ]
    return [dataset_root / name for name in names]


def resolve_annotated_config(
    dataset_root: Path, split: str, annotated_config: str | None = None
) -> Path | None:
    """Resolve an annotated scene dataset config if one exists."""
    if annotated_config is not None:
        path = Path(annotated_config).expanduser().resolve()
        return path if path.exists() else path

    for candidate in candidate_annotated_configs(dataset_root, split):
        if candidate.exists():
            return candidate
    return None


def split_output_dir(split: str) -> Path:
    """Store outputs under per-split folders to avoid collisions."""
    return OUTPUT_ROOT / split


def split_results_json(split: str) -> Path:
    return split_output_dir(split) / f"results_hm3d_{split}.json"


def split_review_dir(split: str) -> Path:
    return split_output_dir(split) / "hm3d_review"


def split_manual_review_csv(split: str) -> Path:
    return split_output_dir(split) / f"hm3d_manual_review_{split}.csv"


def split_trigger_comparison_json(split: str) -> Path:
    return split_output_dir(split) / f"trigger_comparison_hm3d_{split}.json"
