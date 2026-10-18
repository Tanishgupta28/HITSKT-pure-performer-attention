#!/usr/bin/env python3
"""Bounded CPU forward diagnostics, preserving historical batch shapes and weights.

Full FP64 ASSIST replay is a numerical verification, not model training. Other
datasets can use explicitly labelled sampled diagnostics. No CUDA allocation.
"""
import argparse
from datetime import datetime,timezone
from functools import partial
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from ktbench.batching import collate_hitskt
from ktbench.config import seed_everything
from ktbench.data.session_store import SessionStore,RollingSessionDataset,make_session_dataloader
from ktbench.models.hitskt import HiTSKT
from scripts.audit_hitskt_replay import independent_metrics,METRIC_ABSOLUTE_TOLERANCE
from scripts.diagnose_hitskt_common import OUT,now,old_evaluation,sha

LIMIT=16*1024**3


def own_write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def settings():
    return dict(torch=torch.__version__,numpy=np.__version__,torch_cuda_build=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,threads=torch.get_num_threads(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'))


def estimate(batch,model):
    slots=max(batch.history_question.numel(),batch.target_question.numel())
    c=model.config;size=next(model.parameters()).element_size()
    value=3*slots*c.heads*(c.width//c.heads)**2*size+1024**3
    root=Path('/sys/fs/cgroup/memory')
    stat=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
    total=int((root/'memory.limit_in_bytes').read_text())
    if value>LIMIT or int(stat['total_rss'])+value>total-24*1024**3:
        raise MemoryError(f'CPU inference estimate {value} exceeds guarded headroom')
    return value


def forward(model,batch,capture=False):
    estimate(batch,model);states={};handles=[]
    if capture:
        for name in ['action_encoder','session_encoder','correct_padding_encoder','decoder','prediction']:
            def save(module,inputs,output,key=name):states[key]=output.detach().clone()
            handles.append(getattr(model,name).register_forward_hook(save))
    try:
        with torch.inference_mode():out=model(batch)
    finally:
        for h in handles:h.remove()
    return out,states


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['assist2017','junyi','ednet_kt1'])
    parser.add_argument('--mode',choices=['full','sample'],required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if args.mode=='full' and args.dataset!='assist2017':raise ValueError('Only the bounded ASSIST full FP64 verification is authorized by this runner')
    try:Path('/proc/self/oom_score_adj').write_text('1000')
    except PermissionError:pass
    torch.set_num_threads(2);torch.set_num_interop_threads(1);seed_everything()
    name=f'{args.dataset}_fp64_{args.mode}'
    output=OUT/f'{name}.json';assert not output.exists()
    old=old_evaluation(args.dataset);cache=Path(old['prediction_artifacts_local'])
    arrays={k:np.load(cache/f'{k}.npy',mmap_mode='r') for k in ['logit','probability','label','event_id']}
    run=Path('experiments/hitskt')/args.dataset/'full'
    assert sha(run/'best_model.pt')==old['historical_artifact_sha256']['best_model.pt']
    store=SessionStore(Path('data/processed')/args.dataset/'full/session_store');config=old['historical_config']
    ds=RollingSessionDataset(store,'test',history_sessions=config['history_sessions'])
    _,sampler=make_session_dataloader(ds,token_budget=config['token_budget'],maximum_batch_size=config['maximum_batch_size'],
        bucket_size=config['bucket_size'],shuffle=False,seed=42,workers=0,pin_memory=False)
    sampler.set_epoch(0);batches=list(sampler)
    counts=[];cursor=0;chosen={0,len(batches)//3,len(batches)//2}
    closest=int(np.argmin(np.abs(arrays['logit'])))
    for i,indices in enumerate(batches):
        count=sum(int(store.session_offsets[ds.targets[j]+1]-store.session_offsets[ds.targets[j]]) for j in indices)
        if cursor<=closest<cursor+count:chosen.add(i)
        counts.append((cursor,count));cursor+=count
    assert cursor==old['expected_targets']
    chosen.update([len(batches)//4,3*len(batches)//4])
    net32=HiTSKT.from_checkpoint(run/'best_model.pt',map_location='cpu').eval()
    net64=HiTSKT.from_checkpoint(run/'best_model.pt',map_location='cpu').double().eval()
    collate=partial(collate_hitskt,num_questions=store.metadata['num_questions'],num_skills=store.metadata['num_skills'])
    scratch=Path('.inspection/reproducibility_diagnostics')/name;scratch.mkdir(exist_ok=False)
    total=old['expected_targets'] if args.mode=='full' else sum(counts[i][1] for i in chosen)
    z64=np.lib.format.open_memmap(scratch/'logit.npy',mode='w+',dtype=np.float64,shape=(total,))
    p64=np.lib.format.open_memmap(scratch/'probability.npy',mode='w+',dtype=np.float64,shape=(total,))
    labels=np.lib.format.open_memmap(scratch/'label.npy',mode='w+',dtype=np.uint8,shape=(total,))
    ids=np.lib.format.open_memmap(scratch/'event_id.npy',mode='w+',dtype=np.int64,shape=(total,))
    result=dict(status='RUNNING',started_utc=now(),dataset=args.dataset,mode=args.mode,
        runtime=settings(),checkpoint_sha256=sha(run/'best_model.pt'),selected_epoch=old['best_checkpoint_epoch'],
        unchanged_absolute_metric_tolerance=METRIC_ABSOLUTE_TOLERANCE,implementation_sha256=sha(__file__),
        expected_targets=total,selected_original_batches=sorted(chosen) if args.mode=='sample' else 'ALL',
        memory_estimate_limit_bytes=LIMIT,precision_change='Same saved FP32 weights cast to FP64; unchanged equations/architecture',
        historical_cuda_pointwise_comparison='NOT VERIFIED: no historical vectors and visible MIG occupied')
    own_write(output,result);start=last=time.perf_counter();written=0;weighted=0.;skipped=[];probes=[]
    max_z=max_p=sum_z=sum_p=0.;changed=to_positive=to_negative=0;stage_max={}
    for bi,indices in enumerate(batches):
        if args.mode=='sample' and bi not in chosen:continue
        cursor,count=counts[bi];batch=collate([ds[j] for j in indices]);mask=batch.target_metric_mask
        ranges=[(int(store.session_offsets[ds.targets[j]]),int(store.session_offsets[ds.targets[j]+1])) for j in indices]
        truth=np.concatenate([store.correct[a:b] for a,b in ranges]);event=np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in ranges])
        assert np.array_equal(event,arrays['event_id'][cursor:cursor+count])
        assert np.array_equal(truth,arrays['label'][cursor:cursor+count])
        capture=bi<3 if args.mode=='full' else True
        try:out64,st64=forward(net64,batch,capture)
        except MemoryError as error:
            if args.mode=='full':raise
            skipped.append(dict(batch_index=bi,reason=str(error)));continue
        valid=out64[mask];prob=torch.sigmoid(valid)
        loss=F.binary_cross_entropy_with_logits(out64,batch.target_labels.to(out64.dtype),reduction='none')[mask].mean()
        newz=valid.numpy();newp=prob.numpy();oldz=arrays['logit'][cursor:cursor+count];oldp=arrays['probability'][cursor:cursor+count]
        dz=np.abs(newz-oldz);dp=np.abs(newp-oldp);max_z=max(max_z,float(dz.max()));max_p=max(max_p,float(dp.max()))
        sum_z+=float(dz.sum());sum_p+=float(dp.sum());before=oldp>=.5;after=newp>=.5
        changed+=int(np.count_nonzero(before!=after));to_positive+=int(np.count_nonzero(~before&after));to_negative+=int(np.count_nonzero(before&~after))
        for target,value in [(z64,newz),(p64,newp),(labels,truth),(ids,event)]:target[written:written+count]=value
        weighted+=float(loss)*count;written+=count
        if capture:
            out32,st32=forward(net32,batch,True);repeat,_=forward(net32,batch)
            probe=dict(batch_index=bi,targets=count,fp32_repeat_bitwise_equal=bool(torch.equal(out32,repeat)),
                fp32_cache_maximum_absolute_logit_difference=float((out32[mask]-torch.tensor(np.asarray(oldz))).abs().max()))
            with torch.backends.mkldnn.flags(enabled=False):no_mkldnn,_=forward(net32,batch)
            probe['mkldnn_off_maximum_logit_difference']=float((no_mkldnn[mask]-out32[mask]).abs().max())
            torch.set_num_threads(1);one,_=forward(net32,batch);torch.set_num_threads(2)
            probe['threads1_vs2_maximum_logit_difference']=float((one[mask]-out32[mask]).abs().max())
            probes.append(probe)
            for key in st64:
                a=st32[key].double();b=st64[key]
                if key in ['correct_padding_encoder','decoder']:a=a[batch.target_attention_mask];b=b[batch.target_attention_mask]
                if key=='prediction':a=a.squeeze(-1)[mask];b=b.squeeze(-1)[mask]
                stage_max[key]=max(stage_max.get(key,0.),float((a-b).abs().max()))
            del out32,st32,repeat,no_mkldnn,one
        del out64,st64,valid,prob,batch
        if time.perf_counter()-last>=30:
            print(json.dumps(dict(dataset=args.dataset,mode=args.mode,batches=bi+1,targets=written,total=total,seconds=time.perf_counter()-start)),flush=True);last=time.perf_counter()
    for a in [z64,p64,labels,ids]:a.flush()
    metrics=independent_metrics(p64[:written],labels[:written],z64[:written]);metrics['batch_weighted_bce']=weighted/written
    comparison={}
    if args.mode=='full':
        assert written==total and not skipped
        for key in ['roc_auc','accuracy','precision','recall','f1','mse','bce_float64','batch_weighted_bce']:
            expected=old['stored_metrics']['loss' if 'bce' in key else key]
            comparison[key]=dict(stored=expected,recalculated=metrics[key],absolute_difference=abs(expected-metrics[key]),within_unchanged_tolerance=abs(expected-metrics[key])<=METRIC_ABSOLUTE_TOLERANCE)
    result.update(status='COMPLETE',completed_utc=now(),runtime_seconds=time.perf_counter()-start,
        actual_targets=written,all_original_test_targets_evaluated=args.mode=='full',skipped_batches=skipped,
        metrics=metrics,historical_comparison=comparison,
        historical_metric_gate=('PASS' if all(v['within_unchanged_tolerance'] for v in comparison.values()) else 'FAIL') if comparison else 'NOT VERIFIED: sampled diagnostics only',
        cpu_fp32_vs_fp64=dict(maximum_absolute_logit_difference=max_z,maximum_absolute_probability_difference=max_p,
            mean_absolute_logit_difference=sum_z/written,mean_absolute_probability_difference=sum_p/written,
            changed_threshold_classifications=changed,negative_to_positive=to_positive,positive_to_negative=to_negative,
            scope='All targets' if args.mode=='full' else 'Only selected sampled batches'),
        stage_maximum_absolute_differences=stage_max,fp32_determinism_backend_probes=probes,
        prediction_artifacts_local=str(scratch),prediction_sha256={p.name:sha(p) for p in scratch.glob('*.npy')})
    own_write(output,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
