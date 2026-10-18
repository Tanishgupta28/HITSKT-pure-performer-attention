#!/usr/bin/env python3
"""Compare audit metrics against the unchanged production CPU evaluation path."""
import os
import json
from pathlib import Path
import torch
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.models.hitskt import HiTSKT
from scripts.train_hitskt import _run_epoch
from scripts.audit_hitskt_replay import sha256,write_json

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
try:Path('/proc/self/oom_score_adj').write_text('1000')
except PermissionError:pass
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.manual_seed(42);torch.use_deterministic_algorithms(True,warn_only=True)
run=Path('experiments/hitskt/assist2017/full')
prior=json.loads(Path('reports/independent_correctness_audit/assist2017_evaluation_manifest.json').read_text())
config=json.loads((run/'config.json').read_text())
store=SessionStore('data/processed/assist2017/full/session_store')
dataset=RollingSessionDataset(store,'test',history_sessions=config['history_sessions'])
model=HiTSKT.from_checkpoint(run/'best_model.pt',map_location='cpu').eval()
metrics=_run_epoch(model,dataset,device=torch.device('cpu'),epoch=config['best_epoch'],optimizer=None,workers=0)
differences={k:float(v-prior['recalculated_metrics'][k]) for k,v in metrics.items() if k in prior['recalculated_metrics']}
differences['loss']=metrics['loss']-prior['recalculated_metrics']['historical_batch_weighted_bce']
result=dict(status='PASS' if all(abs(v)<=1e-15 for v in differences.values()) else 'FAIL',
    description='Full ASSIST2017 CPU production _run_epoch vs independent evaluator, same checkpoint and batch order',
    metrics=metrics,differences=differences,checkpoint_sha256=sha256(run/'best_model.pt'),
    source_sha256=sha256(__file__),production_evaluator_sha256=sha256('scripts/train_hitskt.py'))
write_json(Path('reports/independent_correctness_audit/assist2017_cpu_production_crosscheck.json'),result)
print(json.dumps(result,indent=2))
