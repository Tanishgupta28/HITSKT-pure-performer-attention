#!/usr/bin/env python3
"""Four whole train-only targets per dataset; frozen QKV diagnostics, no updates.

Untrained FAVOR+ and historical trained ELU+1 are distinct, labeled sources.
Historical weights are never loaded into FAVOR+. Only processed indices,
aggregate norms/errors and hashes are published; frozen arrays stay local.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from ktbench.batching import ExampleShape,collate_hitskt,example_shape,padded_token_cost
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.favor_plus_pipeline import ROOT,fresh_run,model_config,sha256,write_json,runtime
from ktbench.models.hitskt import HiTSKT
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.models.favor_plus import FavorPlusConfig,FavorPlusAttention,orthogonal_gaussian_projection
from ktbench.models.performer import CausalLinearAttention
from scripts.diagnose_favor_plus_quality import DESIGN,compare
from scripts.favor_plus_quality_common import describe


def select_samples(store,design):
    """Length-filter only whole sessions; never peek at labels for selection."""
    dataset = RollingSessionDataset(store,'train',history_sessions=15)
    candidates = np.unique(np.linspace(0,len(dataset)-1,min(1024,len(dataset)),dtype=np.int64))
    eligible=[]
    for index in candidates:
        target=int(dataset.targets[index]);start,end=dataset._history_bounds(target)
        lengths=np.diff(store.session_offsets[start:target+2])
        # Include session-encoder EOS slots too; reuse the original helper.
        cost=padded_token_cost([ExampleShape(tuple(int(v) for v in lengths[:-1]),int(lengths[-1]))])
        if lengths.max()>256 or cost>2048:continue
        assert np.all(store.session_split[start:target+1]==0)
        assert np.all(store.session_student[start:target+1]==store.session_student[target])
        eligible.append(dict(dataset_index=int(index),target_session=target,history_start=start,
            history_end_exclusive=end,lengths=[int(v) for v in lengths],
            padded_tokens=cost,student=int(store.session_student[target]),split=0))
    chosen=[];used=set()
    for low,high in design['training_sample']['target_length_bins']:
        choices=[r for r in eligible if r['student'] not in used and low<=r['lengths'][-1]<=high]
        if not choices:choices=[r for r in eligible if r['student'] not in used]
        if not choices:raise RuntimeError('not enough bounded whole training sessions')
        item=choices[0];chosen.append(item);used.add(item['student'])
    return dataset,chosen


def capture_model(model,dataset,selected):
    norms={};frozen={};handles=[]
    capture_target=max(selected,key=lambda r:r['lengths'][-1])['target_session']
    active={'target':None}
    def hook(name):
        def collect(module,args,kwargs):
            q,k,v=args
            qm,km=kwargs['query_mask'],kwargs['key_mask']
            q=module._heads(module.query(torch.where(qm[...,None],q,0.)))
            k=module._heads(module.key(torch.where(km[...,None],k,0.)))
            v=module._heads(module.value(torch.where(km[...,None],v,0.)))
            entry=norms.setdefault(name,{'query':[],'key':[],'key_log_feature':[]})
            for key,x,mask in (('query',q,qm),('key',k,km)):
                values=x.norm(dim=-1).masked_select(mask[:,None,:].expand(x.shape[:3])).numpy()
                entry[key].extend(values.tolist())
            if active['target']==capture_target:
                row=int(km.sum(-1).argmax());length=min(int(km[row].sum()),32)
                frozen[name]=(q[row:row+1,:,:length].clone(),k[row:row+1,:,:length].clone(),
                    v[row:row+1,:,:length].clone(),qm[row:row+1,:length].clone(),km[row:row+1,:length].clone())
        return collect
    for name,module in model.named_modules():
        if isinstance(module,(FavorPlusAttention,CausalLinearAttention)):
            handles.append(module.register_forward_pre_hook(hook(name),with_kwargs=True))
    try:
        with torch.no_grad():
            for item in selected:
                active['target']=item['target_session']
                example=dataset[item['dataset_index']]
                assert padded_token_cost([example_shape(example)])==item['padded_tokens']
                batch=collate_hitskt([example],num_questions=model.config.num_questions,num_skills=model.config.num_skills)
                out=model(batch)
                if not torch.isfinite(out).all():raise FloatingPointError('nonfinite sampled model logits')
    finally:
        for h in handles:h.remove()
    return {name:{kind:describe(values) for kind,values in entry.items() if values}
            for name,entry in norms.items()},frozen


def run_dataset(name,config,run_id):
    store=SessionStore(ROOT/'data'/'processed'/name/'full'/'session_store')
    dataset,selected=select_samples(store,config)
    output=fresh_run(name,run_id)
    payload=dict(dataset=name,kind='train_only_frozen_attention_diagnostics',selected=selected,
        metadata_sha256=sha256(store.root/'metadata.json'),responses_used_from_split='train only',
        no_optimizer_steps=True,whole_sessions=True,attention_comparison_prefix_limit=32,
        design_sha256=sha256(DESIGN),runtime=runtime(torch.device('cpu')),sources={})
    for source in config['training_sample']['weight_sources']:
        if source.startswith('untrained'):
            torch.manual_seed(42)
            cfg=model_config(name,int(store.metadata['num_questions']),int(store.metadata['num_skills']))
            model=HiTSKTFavorPlus(cfg,FavorPlusConfig()).eval()
            weight_note='fresh random initialization; not learned FAVOR+ norms'
            checkpoint_hash=None
        else:
            path=ROOT/'experiments'/'hitskt'/name/'full'/'best_model.pt'
            checkpoint_hash=sha256(path)
            model=HiTSKT.from_checkpoint(path,map_location='cpu').eval()
            weight_note='historical trained deterministic ELU+1 control, not trained FAVOR+'
        started=time.perf_counter()
        norms,frozen=capture_model(model,dataset,selected)
        capture_seconds=time.perf_counter()-started
        record=dict(weight_note=weight_note,checkpoint_sha256=checkpoint_hash,norms=norms,
                    capture_seconds=capture_seconds,contexts={})
        for stage,(q,k,v,qm,km) in frozen.items():
            artifact=output/f'{source}_{stage.replace(".","_")}.npz'
            np.savez(artifact,q=q.numpy(),k=k.numpy(),v=v.numpy(),query_mask=qm.numpy(),key_mask=km.numpy())
            results=[]
            for m in config['features']:
                for seed in config['projection_seeds']:
                    # Production buffer rounding retained, while oracle arithmetic
                    # uses float64 on the identical saved float32 coordinates.
                    omega=orthogonal_gaussian_projection(q.shape[1],m,q.shape[-1],seed).float().double()
                    gradient=stage.startswith('decoder') and seed==0
                    result=compare(q.double(),k.double(),v.double(),omega,qm,km,gradient=gradient)
                    result.update(features=m,seed=seed)
                    results.append(result)
            record['contexts'][stage]=dict(artifact=str(artifact.relative_to(ROOT)),sha256=sha256(artifact),
                shape=list(q.shape),results=results)
        payload['sources'][source]=record
        del model
        print(name,source,'PASS sampled norms/comparisons',flush=True)
    payload['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    payload['status']='PASS' if all(r['implementation_status']=='PASS' for s in payload['sources'].values()
        for c in s['contexts'].values() for r in c['results']) else 'FAIL'
    write_json(output/'training_norms_manifest.json',payload)
    (output/'_DIAGNOSTIC_COMPLETE').write_text('train-only frozen analysis; no model updates\n')
    return dict(dataset=name,status=payload['status'],manifest=str((output/'training_norms_manifest.json').relative_to(ROOT)),
                manifest_sha256=sha256(output/'training_norms_manifest.json'),selected_targets=len(selected))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',default='quality_cpu_20261019_v1')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU diagnostics require CUDA hidden')
    torch.set_num_threads(1)
    config=json.loads(DESIGN.read_text())
    result=[run_dataset(name,config,args.run_id) for name in config['training_sample']['datasets']]
    write_json(args.output,dict(status='PASS' if all(r['status']=='PASS' for r in result) else 'FAIL',
        datasets=result,design_sha256=sha256(DESIGN),script_sha256=sha256(Path(__file__)),
        oracle_sha256=sha256(Path('scripts/favor_plus_quality_common.py')),no_optimizer_steps=True,
        limitation='untrained FAVOR+ norms plus trained ELU stress controls; no learned FAVOR+ distribution'))


if __name__=='__main__':main()
