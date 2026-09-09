"""Publish mappings and aggregate reports from validated flat-data builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from ktbench.data.validate_flat import validate_flat


FILES = (
    "manifest.json",
    "mappings/original_tags.csv",
    "mappings/composite_skills.csv",
    "mappings/questions.csv",
    "mappings/question_skills.csv",
    "reports/composite_skill_distribution.csv",
    "reports/dataset_report.md",
    "reports/summary.json",
    "reports/validation.json",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(root: Path, output: Path) -> None:
    validate_flat(root)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records = []
    for relative in FILES:
        source = root / relative
        target = output / Path(relative).name
        shutil.copy2(source, target)
        records.append(
            {
                "file": target.name,
                "source": relative,
                "bytes": target.stat().st_size,
                "sha256": _hash(target),
            }
        )
    (output / "published_artifacts.json").write_text(
        json.dumps(
            {
                "processed_root": str(root),
                "contains_raw_interactions": False,
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    publish(args.root, args.output)


if __name__ == "__main__":
    main()
