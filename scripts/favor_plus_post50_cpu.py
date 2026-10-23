#!/usr/bin/env python3
"""Preregistered CPU-only post50 investigation. No CUDA/optimizer/test entry."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from ktbench import favor_plus_post50_diagnostics as d
from ktbench.batching import LengthBucketTokenBatchSampler,collate_hitskt,padded_token_cost
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.favor_plus_bounded_assist import descriptor,stratum
from ktbench.favor_plus_continuation_recovery import validate
from ktbench.favor_plus_pipeline import ROOT,fresh_run,sha256,write_json
from ktbench.favor_plus_real_pilot import host_memory
from ktbench.models.favor_plus import FavorPlusAttention,FavorPlusConfig,causal_favor_attention,orthogonal_gaussian_projection
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.metrics import masked_bce_loss

HERE=ROOT/'reports/favor_plus_v1/post50_cpu_investigation_20261023_v1'
DESIGN=HERE/'PREDECLARED_DESIGN.json'
PREVIOUS=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1'
PARENT25=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist25_shared_cuda_20261023_v1'
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/post50_cpu_diagnostics_20261023_v1'
CHECKPOINT=PREVIOUS/'checkpoints/checkpoint_050.pt'
CHECKPOINT_SHA='93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73'
STORE=ROOT/'data/processed/assist2017/full/session_store'
NAMES=('action_encoder.layers.0.attention','correct_padding_encoder.layers.0.attention')


def load(path):return json.loads(path.read_text())


def host_check(start=None):
    d.require_cpu();m=host_memory()
    if m['peak_process_rss_bytes']>4*1024**3 or m['cgroup_headroom_bytes']<2*1024**3:raise MemoryError('CPU RSS/headroom limit')
    if start is not None and time.time()-start>1200:raise TimeoutError('CPU diagnostic1200s ceiling')
    return m


def cpu_seed():
    # CPU generator only; do not call shared seed/RNG helpers that query CUDA.
    torch.set_num_threads(1);torch.random.default_generator.manual_seed(42)
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)


def require_cpu_autograd_build():
    """CPU tensors alone do not prevent this CUDA-linked engine enumerating GPUs.

    The completed original profiles/logs and exact source snapshot are retained.
    Future strict CPU backward profiles require a CPU-only build before any graph.
    """
    d.require_cpu()
    if torch.version.cuda is not None:
        raise RuntimeError('strict CPU backward profiling requires CPU-only PyTorch; refusing CUDA-linked autograd before device enumeration')


def model():
    host_check();cpu_seed()
    p=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    c={**load(PREVIOUS/'prepared_plan.json')['contract_base'],'prepared_plan_sha256':sha256(PREVIOUS/'prepared_plan.json')}
    validate(p,c)
    if sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('checkpoint50 hash changed')
    net=HiTSKTFavorPlus(HiTSKTConfig(**p['config']),FavorPlusConfig(**p['favor_config']))
    net.load_state_dict(p['state_dict'],strict=True);net.eval();del p
    return net


def dataset():return RollingSessionDataset(SessionStore(STORE),'train',history_sessions=15)


def batch(train,indices):
    if any(int(train.store.session_split[int(train.targets[i])])!=0 for i in indices):raise ValueError('TRAIN only')
    return collate_hitskt([train[i] for i in indices],num_questions=train.store.metadata['num_questions'],num_skills=train.store.metadata['num_skills'])


def prepare():
    host=host_check();design=load(DESIGN)
    cp=load(PREVIOUS/'prepared_plan.json')['contract_base']
    for name,expected in cp['source_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('bound source changed: '+name)
    for name,expected in cp['dataset_identity']['files'].items():
        if sha256(STORE/name)!=expected:raise ValueError('dataset hash changed')
    net=model();parameter_bytes=sum(v.numel()*v.element_size() for v in net.parameters());buffer_bytes=sum(v.numel()*v.element_size() for v in net.buffers());del net
    train=dataset();shapes=[train.shape(i) for i in range(len(train))]
    items=[descriptor(train,i) for i in range(len(train))];cohort=[dict(x,selection='original_frozen_case') for x in cp['diagnostic_samples']]
    targets={x['target_session'] for x in cohort};students={x['student'] for x in cohort}
    for group in range(6):
        candidates=[x for x in items if stratum(x)==group and x['target_session'] not in targets and x['padded_tokens']<=16384]
        independent=[x for x in candidates if x['student'] not in students];candidates=independent or candidates
        if candidates:
            candidates.sort(key=lambda x:(x['padded_tokens'],x['dataset_index']));chosen=candidates[len(candidates)//2]
            cohort.append(dict(chosen,selection='median_cost_new_stratum',stratum=group));targets.add(chosen['target_session']);students.add(chosen['student'])
    for key in ('target_length','padded_tokens'):
        chosen=max(items,key=lambda x:(x[key],-x['dataset_index']))
        if chosen['target_session'] not in targets and chosen['padded_tokens']<=16384:
            cohort.append(dict(chosen,selection='extreme_'+key));targets.add(chosen['target_session'])
    assert len(cohort)<=14 and all(x['split']=='train' for x in cohort)
    sampler=LengthBucketTokenBatchSampler(shapes,token_budget=32768,max_batch_size=64,bucket_size=512,shuffle=True,seed=42);sampler.set_epoch(1)
    batches=list(sampler);seen=[i for b in batches for i in b]
    assert sorted(seen)==list(range(len(train)))
    memory=[d.shape_memory([shapes[i] for i in b],parameter_bytes,buffer_bytes) for b in batches]
    for b,m in zip(batches,memory):
        m['dataset_indices']=b;m['useful_tokens']=sum(sum(x+1 for x in shapes[i].history_lengths)+shapes[i].target_length+1+len(shapes[i].history_lengths)+1 for i in b)
        m['padding_fraction']=1-m['useful_tokens']/m['token_cost']
    singleton=sorted([x for x in items if x['padded_tokens']<=2048],key=lambda x:(x['padded_tokens'],x['dataset_index']))
    profile=[dict(name='median_singleton',indices=[singleton[len(singleton)//2]['dataset_index']]),
        dict(name='largest_singleton_under2048',indices=[singleton[-1]['dataset_index']])]
    eligible=[m for m in memory if m['batch_size']<=4 and m['token_cost']<=2048]
    if eligible:profile.append(dict(name='small_original_batch',indices=max(eligible,key=lambda m:m['token_cost'])['dataset_indices']))
    source={**cp['source_sha256'],**{n:sha256(ROOT/n) for n in ('ktbench/favor_plus_post50_diagnostics.py','scripts/favor_plus_post50_cpu.py')},str(DESIGN.relative_to(ROOT)):sha256(DESIGN)}
    output=fresh_run('assist2017',RUN.name)
    plan=dict(kind=design['kind'],design_sha256=sha256(DESIGN),source_sha256=source,checkpoint50_sha256=CHECKPOINT_SHA,
        previous_plan_sha256=sha256(PREVIOUS/'prepared_plan.json'),started_unix=time.time(),cohort=cohort,profile_cases=profile,
        parameter_bytes=parameter_bytes,buffer_bytes=buffer_bytes,host=host,no_cuda=True,optimizer_calls=0,test_evaluations=0)
    write_json(output/'prepared_plan.json',plan)
    summary=dict(train_examples=len(train),train_target_labels=sum(x.target_length for x in shapes),
        target_length=d.describe([x.target_length for x in shapes]),history_count=d.describe([len(x.history_lengths) for x in shapes]),
        historical_session_length=d.describe([v for x in shapes for v in x.history_lengths]),
        singleton_padded_cost=d.describe([x['padded_tokens'] for x in items]),
        singleton_cost_above2048=sum(x['padded_tokens']>2048 for x in items),original_batch_count=len(batches),
        original_batch_size=d.describe([m['batch_size'] for m in memory]),original_padded_cost=d.describe([m['token_cost'] for m in memory]),
        original_padding_fraction=d.describe([m['padding_fraction'] for m in memory]),parameter_bytes=parameter_bytes,buffer_bytes=buffer_bytes,
        component_sum_bytes=d.describe([m['component_sum_bytes'] for m in memory]),
        workspace_component_bytes=d.describe([m['max_one_weighted_state_per_recomputed_token_bytes'] for m in memory]),
        cohort=cohort,profile_cases=profile,full_batches=memory,limitations='Static CPU component accounting, not measured GPU peak; exact train shapes/sampler, no test labels or prediction selection.')
    write_json(output/'TRAIN_SHAPE_ACCOUNTING.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('full_batches','cohort')},indent=2))


def plan():
    p=load(RUN/'prepared_plan.json');host_check(p['started_unix'])
    if p['design_sha256']!=sha256(DESIGN) or sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('frozen design/checkpoint mismatch')
    for name,expected in p['source_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('diagnostic source changed: '+name)
    return p


def capture_case(net,b):
    captured={};handles=[]
    def hook(name):
        def collect(module,args,kwargs):
            q0,k0,v0=args;qm,km=kwargs['query_mask'],kwargs['key_mask']
            tensors=[module._heads(layer(torch.where(mask[...,None],value,0.))) for layer,value,mask in
                ((module.query,q0,qm),(module.key,k0,km),(module.value,v0,km))]
            row=int(km.sum(-1).argmax());n=int(km[row].sum());q,k,v=[x[row:row+1,:,:n] for x in tensors]
            qms,kms=qm[row:row+1,:n],km[row:row+1,:n]
            actual=causal_favor_attention(q,k,v,module.projection,query_mask=qms,key_mask=kms,
                chunk_size=module.config.chunk_size,accumulation_dtype=module.config.accumulation_dtype)
            captured[name]={key:x.detach().numpy() for key,x in [('q',q[0]),('k',k[0]),('v',v[0]),('projection',module.projection),
                ('query_mask',qms[0]),('key_mask',kms[0]),('actual',actual[0])]}
        return collect
    try:
        for name,module in net.named_modules():
            if name in NAMES:handles.append(module.register_forward_pre_hook(hook(name),with_kwargs=True))
        with torch.no_grad():logits=net(b)
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite CPU frozen model')
    finally:
        for handle in handles:handle.remove()
    return captured


def attention():
    p=plan();net=model();train=dataset();results=[];projections={};projection_checks={}
    for name,mod in net.named_modules():
        if isinstance(mod,FavorPlusAttention):
            stage_seed=42+['action_encoder','session_encoder','correct_padding_encoder','decoder'].index(name.split('.')[0])
            expected=orthogonal_gaussian_projection(mod.heads,128,mod.head_width,stage_seed).float()
            assert torch.equal(expected,mod.projection)
            rows=mod.projection.double();unit=rows/rows.norm(dim=-1,keepdim=True)
            orth=max(float((block@block.transpose(-1,-2)-torch.eye(block.shape[1])).abs().max()) for block in unit.split(mod.head_width,dim=1))
            assert orth<5e-6
            projection_checks[name]=dict(exact_regenerated_seed=stage_seed,orthogonal_block_max_error=orth,radii_squared=d.describe(rows.square().sum(-1).numpy()))
    for case_id,item in enumerate(p['cohort']):
        host_check(p['started_unix']);arrays=capture_case(net,batch(train,[item['dataset_index']]))
        for name,a in arrays.items():
            with (RUN/f'case{case_id:02d}_{name.split(".")[0]}.npz').open('xb') as f:np.savez_compressed(f,**a)
            q,k,v,w=a['q'],a['k'],a['v'],a['projection'];qm,km=a['query_mask'],a['key_mask'];n=q.shape[1]
            lengths=sorted({x for x in (1,8,32,128,256,512,939) if x<=n}|{n})
            record=dict(case=case_id,selection=item,stage=name,actual_positions=n,production_lengths=[],feature_comparison=[])
            short=min(16,n);weights=d.attention_weights(q[:,:short],k[:,:short],w,qm[:short],km[:short])[0]
            direct=d.direct_log_feature_weights(q[:,:short].astype(float),k[:,:short].astype(float),w.astype(float),qm[:short],km[:short])
            assert np.max(np.abs(weights-direct))<=1e-10
            for length in lengths:
                stats=d.attention_diagnostics(q[:,:length],k[:,:length],v[:,:length],w,qm[:length],km[:length],a['actual'][:,:length])
                if stats['feature_kernel_agreement']!='PASS':raise FloatingPointError('production/reference mismatch')
                record['production_lengths'].append(dict(length=length,**stats))
            length=min(128,n)
            for m in (128,256,512):
                for seed in (0,1,2,3):
                    key=(m,seed)
                    if key not in projections:projections[key]=orthogonal_gaussian_projection(q.shape[0],m,q.shape[-1],seed).numpy()
                    stats=d.attention_diagnostics(q[:,:length],k[:,:length],v[:,:length],projections[key],qm[:length],km[:length])
                    record['feature_comparison'].append(dict(features=m,diagnostic_projection_seed=seed,length=length,**stats))
            results.append(record);write_json(RUN/f'attention_case{case_id:02d}_{name.split(".")[0]}.json',record)
        print('completed frozen training case',case_id,'cost',item['padded_tokens'],flush=True)
    write_json(RUN/'ATTENTION_RESULTS.json',dict(results=results,projection_checks=projection_checks,host=host_check(p['started_unix']),
        optimizer_calls=0,cuda_initialized=torch.cuda.is_initialized(),test_evaluations=0))


def metrics():
    p=plan();rows={};prob={};truth=None
    for step in (20,25,30,40,50):
        folder=PARENT25/'stage_025' if step==20 else PREVIOUS/f'stage_{30 if step<=30 else step:03d}'
        with np.load(folder/f'validation_predictions_{step:03d}.npz') as z:
            prob[step]=z['probability'];y=z['label']
        if truth is not None:assert np.array_equal(truth,y)
        truth=y.copy();stored=load(folder/f'diagnostics_result_{step:03d}.json')['sampled_validation']
        independent=d.binary_summary(prob[step],truth)
        for key in ('roc_auc','accuracy','precision','recall','f1'):assert abs(independent[key]-stored[key])<1e-12
        rows[str(step)]=dict(official=independent,stored_logit_bce=stored['loss'],threshold_sensitivity={str(t):d.binary_summary(prob[step],truth,t) for t in (.4,.5,.6)},
            probability=d.describe(prob[step]),positive_probability=d.describe(prob[step][truth==1]),negative_probability=d.describe(prob[step][truth==0]))
    transitions={}
    for initial,final in ((20,25),(25,30),(30,40),(40,50),(25,50),(20,50)):
        a,b=prob[initial]>=.5,prob[final]>=.5
        transitions[f'{initial}_to_{final}']=dict(positive_to_negative_true1=int(np.sum(a&~b&(truth==1))),
            negative_to_positive_true1=int(np.sum(~a&b&(truth==1))),positive_to_negative_true0=int(np.sum(a&~b&(truth==0))),
            negative_to_positive_true0=int(np.sum(~a&b&(truth==0))),probability_change=d.describe(prob[final]-prob[initial]),
            changed_classifications=int(np.sum(a!=b)))
    train=dataset();samples=load(PREVIOUS/'prepared_plan.json')['contract_base']['exact_training_and_probe_samples'][:50];training=[]
    for item in samples:
        b=batch(train,[item['dataset_index']]);y=b.target_labels[b.target_metric_mask].numpy()
        step=len(training)+1;folder=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist20_recovery_20261020_v1'/f'stage_{5 if step<=5 else 10 if step<=10 else 20:03d}' if step<=20 else PARENT25/'stage_025' if step<=25 else PREVIOUS/f'stage_{30 if step<=30 else 40 if step<=40 else 50:03d}'
        row=load(folder/f'update_{step:03d}.json');training.append(dict(update=step,targets=len(y),positive=int(y.sum()),positive_rate=float(y.mean()),
            loss=row['training_bce'],gradient_l2=row['gradient_l2']))
    prevalence=float(truth.mean());constant=d.binary_summary(np.full(len(truth),prevalence),truth)
    write_json(RUN/'PREDICTIVE_METRIC_ANALYSIS.json',dict(milestones=rows,transitions=transitions,validation_prevalence=prevalence,
        fixed_prevalence_descriptive_baseline=constant,majority_accuracy=max(prevalence,1-prevalence),training_updates=training,
        train_interaction_prevalence=sum(x['positive'] for x in training)/sum(x['targets'] for x in training),
        train_mean_session_prevalence=float(np.mean([x['positive_rate'] for x in training])),
        labels_identical_all_milestones=True,no_threshold_selected=True,optimizer_calls=0,cuda_initialized=torch.cuda.is_initialized(),test_evaluations=0))


def probes():
    p=plan();rows=[]
    for name in NAMES:
        for case in range(6):
            a={}
            for step in (25,50):
                folder=PREVIOUS/f'stage_{30 if step==25 else 50:03d}'
                path=folder/f'm{step:03d}_sample{case}_e0_{name.replace(".","_")}.npz'
                with np.load(path) as z:a[step]={key:z[key] for key in z.files}
            for key in ('query_mask','key_mask','projection'):assert np.array_equal(a[25][key],a[50][key])
            qm,km=a[25]['query_mask'][0],a[25]['key_mask'][0];w=a[25]['projection']
            swaps={}
            for label,steps in [('all25',(25,25,25)),('q50',(50,25,25)),('k50',(25,50,25)),('v50',(25,25,50)),('qk50',(50,50,25)),('all50',(50,50,50))]:
                swaps[label]=d.attention_diagnostics(*(a[s][key][0] for key,s in zip(('q','k','v'),steps)),w,qm,km)
            norm={}
            for factor in (.5,1.,1.5):norm[str(factor)]=d.attention_diagnostics(a[50]['q'][0]*factor,a[50]['k'][0]*factor,a[50]['v'][0],w,qm,km)
            rows.append(dict(stage=name,case=case,swapped_frozen_representations=swaps,frozen_qk_scale_probes=norm))
    write_json(RUN/'FROZEN_REPRESENTATION_PROBES.json',dict(rows=rows,optimizer_calls=0,cuda_initialized=torch.cuda.is_initialized(),
        interpretation='Algebraic frozen-array sensitivity, not new training or causal attribution of learning. Q/K scaling is an explicit diagnostic counterfactual only; no model/configuration change.'))


def profile_case(index):
    require_cpu_autograd_build()
    p=plan();selection=p['profile_cases'][index];net=model();train=dataset();b=batch(train,selection['indices'])
    assert b.padded_tokens<=2048 and len(selection['indices'])<=4
    # Bound only this diagnostic child's virtual growth, never another workload.
    vm=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmSize:')))*1024
    soft,hard=resource.getrlimit(resource.RLIMIT_AS);cap=vm+1536*1024**2
    resource.setrlimit(resource.RLIMIT_AS,(min(cap,hard) if hard!=-1 else cap,hard))
    excluded=[d.SavedStorageTracker.key(x) for x in [*net.parameters(),*net.buffers()]]
    tracker=d.SavedStorageTracker(excluded);net.train();begin=time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack,tracker.unpack):
        logits=net(b);loss=masked_bce_loss(logits,b.target_labels,b.target_metric_mask);forward_live=tracker.live
        tracker.phase='backward';loss.backward()
    finite=all(x.grad is not None and torch.isfinite(x.grad).all() for x in net.parameters())
    assert finite
    result=dict(selection=selection,padded_tokens=b.padded_tokens,peak_unique_live_saved_activation_bytes=tracker.peak,
        saved_storage_phase_peaks=tracker.phase_peak,forward_saved_live_bytes=forward_live,remaining_saved_live_bytes=tracker.live,
        pack_calls=tracker.pack_calls,cpu_forward_backward_seconds=time.perf_counter()-begin,host=host_check(p['started_unix']),
        gradients_finite=True,optimizer_calls=0,cuda_initialized=torch.cuda.is_initialized(),test_evaluations=0,
        interpretation='Measured CPU RSS and autograd saved storage, not CUDA peak. No Adam constructed or called; saved-storage measure excludes unsaved temporaries/allocator metadata.')
    write_json(RUN/f'CPU_PROFILE_{index}.json',result);print(json.dumps(result,indent=2))


def profiles():
    require_cpu_autograd_build()
    p=plan()
    for i in range(len(p['profile_cases'])):
        command=[sys.executable,str(Path(__file__).resolve()),'profile-case','--case',str(i)]
        with (RUN/f'cpu_profile_{i}.log').open('x') as f:
            subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
    write_json(RUN/'CPU_PROFILE_RESULTS.json',dict(results=[load(RUN/f'CPU_PROFILE_{i}.json') for i in range(len(p['profile_cases']))],
        optimizer_calls=0,cuda_initialized=torch.cuda.is_initialized(),test_evaluations=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','attention','metrics','probes','profiles','profile-case']);parser.add_argument('--case',type=int)
    args=parser.parse_args();d.require_cpu()
    if (RUN/'INVESTIGATION_RESULT.json').exists():raise RuntimeError('completed diagnostic campaign; preserve results, no automatic replay')
    try:
        if args.phase=='profile-case':profile_case(args.case)
        else:globals()[args.phase]()
    except BaseException as exc:
        if RUN.exists():
            path=RUN/f'failure_{args.phase}_{args.case}.json'
            if not path.exists():write_json(path,dict(status='FAIL',type=type(exc).__name__,message=str(exc),optimizer_calls=0,cuda_initialized=torch.cuda.is_initialized(),no_automatic_retry=True))
        raise
