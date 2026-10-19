#!/usr/bin/env python3
"""Read-only check of 375 originals and all prior code/reports at bc262e6."""
import argparse
import hashlib
import subprocess
from pathlib import Path

from scripts.diagnose_hitskt_common import original_inventory,sha,write,now

BASE = "bc262e66a8bb8ae25dc4dc38d7ca47d74c516375"
UNRELATED = {
    "README.md":"e5754c28ae89a628d46e0269523115404be13aba55d252a92c90ece2bf8ac6b8",
    "experiments/sakt/ednet_kt1/full/independent_evaluation.console.log":
        "0f5546314e0911001954ca04e2fccdad7be7e09dce685fb8d61f5fe7a7af76ff"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    original = original_inventory()
    names = subprocess.check_output(["git","ls-tree","-r","--name-only",BASE],text=True).splitlines()
    prior = {n:hashlib.sha256(subprocess.check_output(["git","show",BASE+":"+n])).hexdigest()
             for n in names if n not in original and n.startswith(("reports/","ktbench/","scripts/","tests/"))}
    failures = []
    for kind,items in (("original_research",original),("prior_source_and_evidence",prior),("unrelated",UNRELATED)):
        for path,expected in items.items():
            actual = sha(path)
            if actual != expected: failures.append(dict(kind=kind,path=path,expected=expected,actual=actual))
    branch = subprocess.check_output(["git","branch","--show-current"],text=True).strip()
    assert branch == "research/hitskt-favor-plus-v1"
    write(args.output,dict(utc=now(),base_commit=BASE,branch=branch,
        status="PASS" if not failures else "FAIL",original_count=len(original),
        prior_count=len(prior),originals=original,prior_source_and_evidence=prior,
        unrelated_files=UNRELATED,failures=failures,checker_sha256=sha(__file__)))
    assert not failures
    print("PASS:",len(original),"original research files and",len(prior),"prior source/evidence files; unrelated files unchanged")


if __name__ == "__main__": main()
