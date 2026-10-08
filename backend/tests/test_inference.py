import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.config import ROOT, Settings
from app.inference import Predictor


def test_refuse_half_configured_model():
    with pytest.raises(ValueError, match="both"):
        Predictor(Settings(hitskt_checkpoint="missing.pt", hitskt_manifest=None))


def test_real_model_adapter_preserves_session_and_target_causality(tmp_path):
    torch=pytest.importorskip("torch")
    sys.path.insert(0,str(ROOT))
    from app.research import hitskt_types
    HiTSKT, HiTSKTConfig = hitskt_types()
    checkpoint=tmp_path/"fixture.pt"
    torch.manual_seed(42)
    model=HiTSKT(HiTSKTConfig(num_questions=4,num_skills=2,width=8,heads=2,feedforward_width=16,dropout=0)).eval()
    torch.save(model.checkpoint(),checkpoint)
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"format_version":1,"dataset":"untrained-interface-fixture",
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "questions":{"old":{"question_id":1,"skill_id":1},"current":{"question_id":2,"skill_id":1},"candidate":{"question_id":3,"skill_id":2}}}))
    predictor=Predictor(Settings(hitskt_checkpoint=str(checkpoint),hitskt_manifest=str(manifest)))
    stamp=datetime.now(timezone.utc)
    state={"concepts":{},"history":[{"question_id":"old","correct":True,"answered_at":(stamp-timedelta(hours=12)).isoformat()},
                                   {"question_id":"current","correct":False,"answered_at":stamp.isoformat()}],"inference_at":stamp.isoformat()}
    candidate={"id":"candidate","concept_id":"math","difficulty":2}
    value=predictor.predict_many(state,[candidate])["candidate"]
    assert value["provider"] == "hitskt" and 0 < value["probability"] < 1
    # Repeated inference is deterministic; the candidate answer is unavailable.
    assert predictor.predict_many(state,[candidate])["candidate"] == value
    cold={**state,"history":state["history"][-1:]}
    assert predictor.predict_many(cold,[candidate])["candidate"]["provider"] == "bayesian_knowledge_tracing"
    unknown={**candidate,"id":"unmapped"}
    assert predictor.predict_many(state,[unknown])["unmapped"]["provider"] == "bayesian_knowledge_tracing"
    state["history"][0]["question_id"]="unknown-history"
    assert predictor.predict_many(state,[candidate])["candidate"]["provider"] == "bayesian_knowledge_tracing"
    manifest.write_text(manifest.read_text().replace(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"0"*64))
    with pytest.raises(ValueError,match="SHA256"):
        Predictor(Settings(hitskt_checkpoint=str(checkpoint),hitskt_manifest=str(manifest)))
