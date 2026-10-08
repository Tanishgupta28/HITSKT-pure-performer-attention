"""Check trained checkpoint adapter parity on an explicitly synthetic history.

This verifies serving compatibility, not prediction quality or learning efficacy.
No demonstration math item is mapped to a benchmark embedding by this tool.
"""
import argparse
import csv
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import ROOT, Settings
from app.inference import Predictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--dataset",choices=["assist2017","junyi","ednet_kt1"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import torch
    sys.path.insert(0,str(ROOT))
    from ktbench.batching import Session, SessionExample, collate_hitskt
    reports=ROOT/"reports"/"datasets"/args.dataset
    with (reports/("questions.csv" if args.dataset=="ednet_kt1" else "question_skills.csv")).open(encoding="utf-8",newline="") as file:
        reader=csv.DictReader(file)
        rows=[next(reader) for _ in range(3)]
    mappings={f"benchmark-fixture-{i}":{"question_id":int(row["question_id"]),"skill_id":int(row["skill_id"])} for i,row in enumerate(rows)}
    with args.checkpoint.open("rb") as file:
        digest=hashlib.file_digest(file,"sha256").hexdigest()
    manifest={"format_version":1,"dataset":args.dataset,"checkpoint_sha256":digest,"questions":mappings}
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"manifest.json"
        path.write_text(json.dumps(manifest),encoding="utf-8")
        predictor=Predictor(Settings(hitskt_checkpoint=str(args.checkpoint.resolve()),hitskt_manifest=str(path)))
        timestamp=datetime.now(timezone.utc)
        state={"concepts":{},"history":[{"question_id":"benchmark-fixture-0","correct":True,"answered_at":(timestamp-timedelta(hours=10)).isoformat()},
                                       {"question_id":"benchmark-fixture-1","correct":False,"answered_at":timestamp.isoformat()}],"inference_at":timestamp.isoformat()}
        prediction=predictor.predict_many(state,[{"id":"benchmark-fixture-2","concept_id":"interface-fixture","difficulty":2}])["benchmark-fixture-2"]
        assert prediction["provider"]=="hitskt",prediction
        ids=list(mappings.values())
        # Opposite dummy candidate label proves it cannot enter its own input.
        example=SessionExample([Session([ids[0]["question_id"]],[ids[0]["skill_id"]],[1])],
                               Session([ids[1]["question_id"],ids[2]["question_id"]],[ids[1]["skill_id"],ids[2]["skill_id"]],[0,1]),2)
        batch=collate_hitskt([example],num_questions=predictor.model.config.num_questions,num_skills=predictor.model.config.num_skills)
        with torch.inference_mode():
            direct=torch.sigmoid(predictor.model(batch)[0,1]).item()
        assert prediction["probability"]==direct,(prediction,direct)
        result={"status":"passed","check":"trained checkpoint serving adapter equals direct repository model on a synthetic, mapped multi-session interface fixture",
                "dataset":args.dataset,"checkpoint_sha256":digest,"probability":direct,"torch_version":torch.__version__,
                "session_gap_hours":10,"candidate_label_excluded":True,
                "limitation":"Synthetic labels test serving parity only. No claim of model quality, calibrated mastery, real-student validation or HiTSKT inference on the original platform content."}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
