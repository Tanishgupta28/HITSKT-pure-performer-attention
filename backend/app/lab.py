"""Run a trusted EdNet interface fixture independently of the live math predictor."""
import hashlib
from functools import lru_cache
from pathlib import Path

from app.config import ROOT

DEFAULT_CHECKPOINT = ROOT / 'experiments' / 'hitskt' / 'ednet_kt1' / 'full' / 'best_model.pt'


@lru_cache(maxsize=1)
def verified_model(path: str, expected: str):
    candidate=Path(path)
    with candidate.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
            raise ValueError('The fixture checkpoint does not match the verified receipt.')
    from app.research import hitskt_types
    cls,_=hitskt_types()
    return cls.from_checkpoint(candidate,map_location='cpu').eval()


def run_hitskt_fixture(path, checksum):
    import torch
    model=verified_model(str(Path(path).resolve()),checksum)
    from ktbench.batching import Session, SessionExample, collate_hitskt
    example=SessionExample([Session([1],[2],[1])],Session([2,3],[75,65],[0,0]),2)
    batch=collate_hitskt([example],num_questions=model.config.num_questions,num_skills=model.config.num_skills)
    with torch.inference_mode():
        return torch.sigmoid(model(batch)[0,1]).item()
