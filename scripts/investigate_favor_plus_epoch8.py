#!/usr/bin/env python3
"""Bounded CPU-only frozen training representations. No optimizer/test loader."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from ktbench import favor_plus_frozen_investigation as f
from ktbench import favor_plus_post50_diagnostics as d
from ktbench.batching import collate_hitskt
from ktbench.data.session_store import SessionStore, RollingSessionDataset
from ktbench.favor_plus_assist_epochs import checkpoint_validate
from ktbench.favor_plus_pipeline import ROOT, sha256, write_json
from ktbench.favor_plus_real_pilot import host_memory
from ktbench.models.favor_plus import FavorPlusAttention, FavorPlusConfig, causal_favor_attention, orthogonal_gaussian_projection
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus

HERE=ROOT/'reports/favor_plus_v1/postepoch8_approximation_20261024_v1'
DESIGN=HERE/'PREDECLARED_DESIGN.json'
PARENT=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1'
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/postepoch8_frozen_20261024_v1'
DES=json.loads(DESIGN.read_text())


def guard(start):
    d.require_cpu()
    if torch.version.cuda is not None:
        raise RuntimeError('this study requires an isolated CPU-only PyTorch build')
    if 'libtorch_cuda' in Path('/proc/self/maps').read_text():
        raise RuntimeError('CUDA-linked libtorch was loaded into the CPU worker')
    host=host_memory()
    if host['peak_process_rss_bytes']>4*1024**3 or host['cgroup_headroom_bytes']<2*1024**3:
        raise MemoryError('frozen study RSS/headroom gate')
    if time.time()-start>1200:
        raise TimeoutError('1200s CPU action ceiling')
    return host


def setup():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.random.default_generator.manual_seed(42);torch.use_deterministic_algorithms(True)


def batch(ds, item):
    i=item['dataset_index']
    if int(ds.targets[i])!=item['target_session'] or int(ds.store.session_split[int(ds.targets[i])])!=0:
        raise ValueError('frozen TRAIN example identity mismatch')
    b=collate_hitskt([ds[i]],num_questions=ds.store.metadata['num_questions'],num_skills=ds.store.metadata['num_skills'])
    if b.padded_tokens!=item['padded_tokens'] or b.padded_tokens>16384:
        raise ValueError('frozen shape/resource contract mismatch')
    return b


def make_model(state, contract):
    setup();net=HiTSKTFavorPlus(HiTSKTConfig(**contract['model_config']),FavorPlusConfig(**contract['favor_config']))
    if state!='initial':
        epoch=int(state);path=PARENT/f'checkpoints/epoch_{epoch:03d}.pt'
        if sha256(path)!=DES['checkpoints'][state]:raise ValueError('protected checkpoint changed')
        payload=torch.load(path,map_location='cpu',weights_only=True);checkpoint_validate(payload,contract)
        net.load_state_dict(payload['state_dict'],strict=True);del payload
    net.eval();return net


def capture(net,b,names):
    values={};handles=[]
    def collect(name):
        def hook(module,args,kwargs):
            q0,k0,v0=args;qm,km=kwargs['query_mask'],kwargs['key_mask']
            projected=[module._heads(layer(torch.where(mask[...,None],x,0.))) for layer,x,mask in ((module.query,q0,qm),(module.key,k0,km),(module.value,v0,km))]
            row=int(km.sum(-1).argmax());active=qm[row]|km[row];n=int(torch.nonzero(active)[-1])+1
            q,k,v=[x[row:row+1,:,:n] for x in projected];qms,kms=qm[row:row+1,:n],km[row:row+1,:n]
            actual=causal_favor_attention(q,k,v,module.projection,query_mask=qms,key_mask=kms,chunk_size=module.config.chunk_size)
            values[name]={key:x.detach().numpy().copy() for key,x in (('q',q[0]),('k',k[0]),('v',v[0]),('projection',module.projection),('query_mask',qms[0]),('key_mask',kms[0]),('actual',actual[0]))}
        return hook
    try:
        for name,module in net.named_modules():
            if name in names:handles.append(module.register_forward_pre_hook(collect(name),with_kwargs=True))
        with torch.no_grad():logits=net(b)
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite CPU model')
    finally:
        for h in handles:h.remove()
    return values


def prepare():
    start=time.time();setup();host=guard(start);RUN.mkdir(exist_ok=False)
    plan=json.loads((PARENT/'prepared_plan.json').read_text());c=plan['contract']
    for path,expected in c['source_sha256'].items():
        if sha256(ROOT/path)!=expected:raise ValueError('original source changed:'+path)
    for path,expected in c['dataset_identity']['files'].items():
        if sha256(Path(plan['store_path'])/path)!=expected:raise ValueError('processed data changed')
    src=['ktbench/favor_plus_frozen_investigation.py','scripts/investigate_favor_plus_epoch8.py']
    write_json(RUN/'prepared_plan.json',dict(design_sha256=sha256(DESIGN),sources={p:sha256(ROOT/p) for p in src},parent_plan_sha256=sha256(PARENT/'prepared_plan.json'),cpu_runtime=dict(torch=torch.__version__,cuda_build=torch.version.cuda,numpy=np.__version__),initial_host=host,optimizer_calls=0,test_evaluations=0))
    print('prepared frozen CPU study',flush=True)


def check_plan():
    plan=json.loads((RUN/'prepared_plan.json').read_text())
    if plan['design_sha256']!=sha256(DESIGN):raise ValueError('design drift')
    for path,expected in plan['sources'].items():
        if sha256(ROOT/path)!=expected:raise ValueError('diagnostic source drift:'+path)
    return plan


def captures():
    start=time.time();setup();guard(start);check_plan()
    plan=json.loads((PARENT/'prepared_plan.json').read_text());ds=RollingSessionDataset(SessionStore(plan['store_path']),'train',history_sessions=15)
    results=[];projection_checks={}
    for state in ('initial','7','8'):
        net=make_model(state,plan['contract'])
        for name,module in net.named_modules():
            if isinstance(module,FavorPlusAttention):
                seed=42+['action_encoder','session_encoder','correct_padding_encoder','decoder'].index(name.split('.')[0])
                expected=orthogonal_gaussian_projection(4,128,64,seed).float()
                if not torch.equal(expected,module.projection):raise ValueError('saved/regenerated ORF mismatch')
                unit=module.projection.double()/module.projection.double().norm(dim=-1,keepdim=True)
                err=max(float((x@x.transpose(-1,-2)-torch.eye(x.shape[1])).abs().max()) for x in unit.split(64,dim=1))
                projection_checks[name]=dict(seed=seed,regenerated_bit_exact=True,orthogonal_block_max_abs=err,radius_squared=d.describe(module.projection.double().square().sum(-1).numpy()))
        for case,item in enumerate(DES['training_cohort']):
            guard(start);arrays=capture(net,batch(ds,item),DES['stages'])
            for stage,a in arrays.items():
                path=RUN/f'{state}_case{case:02d}_{stage.split(".")[0]}.npz'
                with path.open('xb') as stream:np.savez_compressed(stream,**a)
                n=a['q'].shape[1];record=dict(state=state,case=case,stage=stage,length=n,prefixes=[])
                for length in sorted({x for x in (1,8,32,128,256,512) if x<=n}|{n}):
                    m={k:v[:length] for k,v in a.items() if k in ('query_mask','key_mask')}
                    stats=d.attention_diagnostics(a['q'][:,:length],a['k'][:,:length],a['v'][:,:length],a['projection'],m['query_mask'],m['key_mask'],a['actual'][:,:length])
                    if stats['feature_kernel_agreement']!='PASS':raise FloatingPointError('CPU feature-kernel mismatch')
                    record['prefixes'].append(dict(length=length,metrics=stats,geometry=f.pair_geometry(a['q'][:,:length],a['k'][:,:length],m['query_mask'],m['key_mask'])))
                results.append(record)
            print('captured',state,case,flush=True)
        del net;gc.collect()
    write_json(RUN/'CAPTURE_RESULTS.json',dict(records=results,projection_checks=projection_checks,host=guard(start),seconds=time.time()-start,cuda_initialized=torch.cuda.is_initialized(),optimizer_calls=0,test_evaluations=0))


def load_array(state,case,stage,limit=128):
    with np.load(RUN/f'{state}_case{case:02d}_{stage.split(".")[0]}.npz') as z:a={k:z[k] for k in z.files}
    n=min(limit,a['q'].shape[1])
    return dict(q=a['q'][:,:n],k=a['k'][:,:n],v=a['v'][:,:n],w=a['projection'],qm=a['query_mask'][:n],km=a['key_mask'][:n])


def grid():
    start=time.time();setup();guard(start);check_plan();results=[]
    pools={seed:orthogonal_gaussian_projection(4,512,64,seed).numpy() for seed in DES['feature_sweep']['independent_seeds']}
    for state in ('7','8'):
        for case in range(len(DES['training_cohort'])):
            guard(start)
            for stage in DES['stages']:
                a=load_array(state,case,stage);q,k,v,w,qm,km=[a[x] for x in ('q','k','v','w','qm','km')]
                row=dict(state=state,case=case,stage=stage,length=q.shape[1],saved128=d.attention_diagnostics(q,k,v,w,qm,km),saved128_centered=f.evaluate_arm(q,k,v,w,qm,km,'first_valid_key_centering_ORF'),grid=[])
                for seed,pool in pools.items():
                    for m in DES['feature_sweep']['counts']:
                        for arm in DES['feature_sweep']['arms']:
                            stamp=time.perf_counter();result=f.evaluate_arm(q,k,v,f.projection_arm(pool,m,arm),qm,km,arm)
                            row['grid'].append(dict(seed=seed,features=m,arm=arm,metrics=result,diagnostic_cpu_seconds=time.perf_counter()-stamp))
                results.append(row)
            write_json(RUN/f'grid_{state}_case{case:02d}.json',dict(records=results[-4:]))
            print('grid',state,case,flush=True)
    write_json(RUN/'FEATURE_GRID_RESULTS.json',dict(records=results,host=guard(start),seconds=time.time()-start,optimizer_calls=0,test_evaluations=0))


def controls():
    start=time.time();setup();guard(start);check_plan();rows=[]
    for state in ('7','8'):
        for case in range(len(DES['training_cohort'])):
            guard(start)
            for stage in DES['stages']:
                a,b=load_array('initial',case,stage),load_array(state,case,stage)
                for name in ('w','qm','km'):
                    if not np.array_equal(a[name],b[name]):raise ValueError('frozen mask/projection mismatch')
                swaps={}
                for name,choose in [('initial',(a,a,a)),('trained_q_only',(b,a,a)),('trained_k_only',(a,b,a)),('trained_v_only',(a,a,b)),('trained_qk',(b,b,a)),('trained_qkv',(b,b,b))]:
                    swaps[name]=d.attention_diagnostics(choose[0]['q'],choose[1]['k'],choose[2]['v'],a['w'],a['qm'],a['km'])
                scales={str(scale):d.attention_diagnostics(b['q']*scale,b['k']*scale,b['v'],b['w'],b['qm'],b['km']) for scale in DES['controls']['qk_scale']}
                fp64=causal_favor_attention(*(torch.tensor(b[x],dtype=torch.float64)[None] for x in ('q','k','v')),torch.tensor(b['w'],dtype=torch.float64),query_mask=torch.tensor(b['qm'])[None],key_mask=torch.tensor(b['km'])[None],accumulation_dtype='float64').numpy()[0]
                stats64=d.attention_diagnostics(b['q'],b['k'],b['v'],b['w'],b['qm'],b['km'],fp64)
                rows.append(dict(state=state,case=case,stage=stage,swaps=swaps,scales=scales,float64=stats64,value_offset=f.value_offset_control(b['q'],b['k'],b['v'],b['w'],b['qm'],b['km'])))
            print('controls',state,case,flush=True)
    write_json(RUN/'CONTROL_RESULTS.json',dict(records=rows,host=guard(start),seconds=time.time()-start,optimizer_calls=0,test_evaluations=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','capture','grid','controls'));a=parser.parse_args()
    {'prepare':prepare,'capture':captures,'grid':grid,'controls':controls}[a.action]()


if __name__=='__main__':main()
