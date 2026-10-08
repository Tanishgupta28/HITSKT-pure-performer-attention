"""Validate owner-reviewed content IDs against persisted dataset mappings.

python backend/tools/build_manifest.py --dataset junyi --checkpoint PATH \
    --mapping reviewed-content-mapping.json --output backend/model-assets/manifest.json

Mapping input is an object: platform_id -> {question_id: int, skill_id: int}.
Only use it for the actual original content; numeric validity is not proof of
semantic identity or permission to publish question text.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["assist2017","junyi","ednet_kt1"],required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--mapping",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    reports=root/"reports"/"datasets"/args.dataset
    mapping=json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping,dict) or not mapping:
        raise ValueError("Mapping must be a nonempty reviewed object")
    pairs_path=reports/("questions.csv" if args.dataset=="ednet_kt1" else "question_skills.csv")
    with pairs_path.open(encoding="utf-8",newline="") as file:
        pairs={(int(row["question_id"]),int(row["skill_id"])) for row in csv.DictReader(file)}
    for platform_id,ids in mapping.items():
        if not isinstance(platform_id,str) or not platform_id:
            raise ValueError("Platform IDs must be nonempty strings")
        if type(ids.get("question_id")) is not int or type(ids.get("skill_id")) is not int:
            raise ValueError(f"Noninteger IDs for {platform_id}")
        if (ids["question_id"],ids["skill_id"]) not in pairs:
            raise ValueError(f"Question/skill pair absent from dataset reports for {platform_id}")
    with args.checkpoint.open("rb") as file:
        digest=hashlib.file_digest(file,"sha256").hexdigest()
    manifest={"format_version":1,"dataset":args.dataset,"checkpoint_sha256":digest,"questions":mapping,
              "mapping_report_sha256":hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
              "provenance_note":"Numeric mappings checked against reports. Content owner must separately review original-content identity and publication rights."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Validated {len(mapping)} mappings for {args.dataset}; wrote {args.output}")


if __name__=="__main__":
    main()
