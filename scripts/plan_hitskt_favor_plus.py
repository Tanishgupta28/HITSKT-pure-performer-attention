#!/usr/bin/env python3
"""Read-only production-data/profile compatibility plan; no model or training."""
import argparse
import json
from pathlib import Path

from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.favor_plus_pipeline import ROOT,PROTOCOL,model_config,profile_for,sha256,target_count,write_json
from ktbench.hitskt_registry import HITSKT_TRAINING_CONFIGS
from scripts.diagnose_hitskt_common import original_inventory


def main():
    from dataclasses import asdict
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    captured = original_inventory()
    records = {}
    for dataset in HITSKT_TRAINING_CONFIGS:
        old_root = ROOT/"experiments"/"hitskt"/dataset/"full"
        historical = json.loads((old_root/"config.json").read_text())
        original_stats = json.loads((old_root/"dataset_stats.json").read_text())
        profile = profile_for(dataset)
        mismatches = {k:{"original":historical.get(k),"new":v} for k,v in profile.items()
                      if k != "model" and historical.get(k) != v}
        if mismatches: raise ValueError(f"nonattention settings differ: {dataset}: {mismatches}")
        store = SessionStore(ROOT/"data"/"processed"/dataset/"full"/"session_store")
        counts = {split:target_count(RollingSessionDataset(store,split,history_sessions=15))
                  for split in ("train","validation","test")}
        if counts != original_stats["target_counts"]: raise ValueError("historical target counts differ")
        files = {str(path.relative_to(ROOT)):captured[str(path.relative_to(ROOT))]
                 for path in sorted(store.root.iterdir()) if path.is_file()}
        records[dataset] = dict(status="PASS",profile=profile,
            model_config=asdict(model_config(dataset,int(store.metadata["num_questions"]),int(store.metadata["num_skills"]))),
            original_config_sha256=sha256(old_root/"config.json"),
            original_dataset_stats_sha256=sha256(old_root/"dataset_stats.json"),
            target_counts=counts,metadata=store.metadata,expected_processed_checksums=files,
            checksum_note="captured original hashes; independently rehashed in preservation.json",
            split="chronological within student; not student-disjoint",
            protocol=PROTOCOL,attention_only_difference=True,
            response_availability="not guaranteed; V1 retrospective teacher forcing retained",
            training_status="NOT RUN",cuda_status="NOT VERIFIED")
    write_json(args.output,dict(status="PASS",scope="read-only plan; no model/training/inference",datasets=records))
    print("PASS: all three nonattention profiles and target counts equal historical configurations")


if __name__ == "__main__": main()
