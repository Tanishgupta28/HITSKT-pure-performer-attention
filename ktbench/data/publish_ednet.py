"""Publish small reproducibility artifacts from a validated EdNet build."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from ktbench.data.validate_ednet import validate_processed_ednet


PUBLISHED_FILES = (
    "manifest.json",
    "mappings/original_tags.csv",
    "mappings/composite_skills.csv",
    "mappings/questions.csv",
    "reports/composite_skill_distribution.csv",
    "reports/dataset_report.md",
    "reports/summary.json",
    "reports/validation.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(processed_root: Path, output_dir: Path) -> None:
    """Validate, then copy only mappings and aggregate reports (never events)."""

    validate_processed_ednet(processed_root)
    if output_dir.exists():
        raise FileExistsError(f"publication directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = []
    for relative_name in PUBLISHED_FILES:
        source = processed_root / relative_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / Path(relative_name).name
        shutil.copy2(source, target)
        manifest.append(
            {
                "file": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "source": relative_name,
            }
        )
    (output_dir / "published_artifacts.json").write_text(
        json.dumps(
            {
                "processed_root": str(processed_root),
                "contains_raw_interactions": False,
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish validated EdNet mappings and aggregate reports"
    )
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    publish(args.processed_root, args.output_dir)


if __name__ == "__main__":
    main()
