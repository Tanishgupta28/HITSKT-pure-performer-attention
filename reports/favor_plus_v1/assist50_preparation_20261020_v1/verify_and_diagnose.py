"""Reproducible bounded CPU review; no real-data forward/optimizer or CUDA.

Run from repo root with CUDA_VISIBLE_DEVICES='' and one CPU thread. Original
checkpoints/raw prefixes are read only; all output publication refuses overwrite.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from ktbench.favor_plus_assist20 import load_plan, source_identity
from ktbench.favor_plus_continuation_contract import proposed_handoff
from ktbench.favor_plus_pilot_diagnostics import cpu_tree, equal_tree
from ktbench.favor_plus_pilot_recovery import digest, validate
from ktbench.favor_plus_pipeline import ROOT, sha256
from ktbench.models.favor_plus import causal_favor_attention, orthogonal_gaussian_projection
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from scripts.favor_plus_quality_common import describe, independent_attention, output_errors

HERE = Path(__file__).resolve().parent
RUN = ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist20_recovery_20261020_v1'


def publish(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def raw_digest(arrays):
    h = hashlib.sha256()
    for key in ('q','k','v','query_mask','key_mask'):
        a = arrays[key]
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def read_prefix(step, example):
    directory = RUN/f'stage_{5 if step==0 else step:03d}'
    path = directory/f'm{step:03d}_e{example}_action_encoder_layers_0_attention.npz'
    with np.load(path) as z:
        return path, {k:z[k] for k in z.files}


def cpu_scan(a, dtype):
    values = [torch.as_tensor(a[k], dtype=dtype) for k in ('q','k','v','projection')]
    with torch.no_grad():
        return causal_favor_attention(*values,query_mask=torch.as_tensor(a['query_mask']),
            key_mask=torch.as_tensor(a['key_mask']),chunk_size=32,
            accumulation_dtype='float64' if dtype==torch.float64 else 'float32').numpy()


def checkpoints():
    plan, contract, _, _ = load_plan(RUN/'prepared_plan.json')
    old_summary = json.loads((HERE.parent/'assist20_recovery_20261020_v1/VERIFICATION_SUMMARY.json').read_text())
    rows=[]
    for step in (5,10,20):
        path=RUN/'checkpoints'/f'checkpoint_{step:03d}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        row=validate(payload,contract)
        prior=next(c for c in old_summary['checkpoints'] if c['update']==step)
        assert sha256(path)==prior['sha256']
        assert row['component_sha256']==prior['component_sha256']
        assert row['projection_sha256']==prior['projection_sha256']
        rng=payload['rng_state']
        random.Random().setstate(rng['python'])
        n=rng['numpy']
        np.random.RandomState().set_state((n['bit_generator'],n['state'].numpy().astype(np.uint32),
            n['position'],n['has_gauss'],n['cached_gaussian']))
        torch.Generator(device='cpu').set_state(rng['torch_cpu'])
        assert len(rng['torch_cuda'])==1
        assert all(s.dtype==torch.uint8 and s.ndim==1 and s.numel()>0 for s in rng['torch_cuda'])
        states=payload['optimizer_state_dict']['state']
        assert all((s['exp_avg_sq']>=0).all() for s in states.values())
        row.update(sha256=sha256(path),bytes=path.stat().st_size,
            cpu_rng_schema_and_local_generator_restore='PASS',cuda_rng_saved_not_restored=True,
            cuda_rng_sizes=[s.numel() for s in rng['torch_cuda']],
            adam_second_moment_minimum=min(float(s['exp_avg_sq'].min()) for s in states.values()),
            adam_second_moment_maximum=max(float(s['exp_avg_sq'].max()) for s in states.values()),
            configuration_sha256=contract['configuration_sha256'])
        if step==20:
            model=HiTSKTFavorPlus.from_checkpoint(path,map_location='cpu')
            assert equal_tree(cpu_tree(model.state_dict()),payload['state_dict'])
            optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)
            optimizer.load_state_dict(payload['optimizer_state_dict'])
            assert equal_tree(cpu_tree(optimizer.state_dict()),payload['optimizer_state_dict'])
            row['cpu_strict_model_and_adam_load']='EXACT; NO_FORWARD_OR_UPDATE'
            frozen=json.loads((ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2/prepared_plan.json').read_text())
            proposal=proposed_handoff(payload,sha256(path),frozen['samples'])
            proposal.update(parent_plan_path=str((RUN/'prepared_plan.json').relative_to(ROOT)),
                parent_checkpoint_path=str(path.relative_to(ROOT)),
                original_source_verification=source_identity()['source_sha256'],
                preparation_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in
                    (ROOT/'ktbench/favor_plus_continuation_contract.py',Path(__file__).resolve())})
            publish('PROPOSED_CONTINUATION_PLAN.json',proposal)
            del optimizer, model
        rows.append(row)
        del payload
    assert all(r['projection_sha256']==rows[0]['projection_sha256'] for r in rows)
    publish('CHECKPOINT_VERIFICATION.json',dict(status='PASS',checkpoints=rows,
        processed_files_verified=len(contract['dataset_identity']['files']),
        scientific_and_control_source_files_verified=len(contract['source_sha256']),
        original_validation_sample_sha256=contract['validation_sample_sha256'],
        additional_optimizer_calls=0,new_gpu_verification='NOT RUN',
        limitation='CPU integrity/load check now; exact CUDA predictions/next-step zero-call recovery passed previously, not rerun'))


def action_review():
    rows=[]; feature_grid=[]; finals=[]
    for step in (0,5,10,20):
        for example in range(6):
            path,a=read_prefix(step,example)
            _,initial=read_prefix(0,example)
            qm,km=a['query_mask'],a['key_mask']
            f,e,stats=independent_attention(a['q'],a['k'],a['v'],a['projection'],qm,km)
            fp32,fp64=cpu_scan(a,torch.float32),cpu_scan(a,torch.float64)
            agreement=output_errors(a['actual'],f,qm)
            assert agreement['max_abs']<=5e-5
            assert output_errors(fp64,f,qm)['max_abs']<1e-10
            err=output_errors(f,e,qm)
            eligible=np.broadcast_to(qm[:,None,:],a['q'].shape[:3])
            per_query=np.linalg.norm(f-e,axis=-1)/np.maximum(np.linalg.norm(e,axis=-1),1e-30)
            norms={}; drift={}
            for key in ('q','k','v'):
                z=a[key].astype(float);z0=initial[key].astype(float)
                norms[key]=describe(np.linalg.norm(z,axis=-1)[eligible])
                drift[key]=dict(relative_l2=float(np.linalg.norm(z-z0)/np.linalg.norm(z0)),
                    cosine=float(np.sum(z*z0)/(np.linalg.norm(z)*np.linalg.norm(z0))))
            row=dict(update=step,example=example,evidence=str(path.relative_to(ROOT)),sha256=sha256(path),
                frozen_qkv_mask_digest=raw_digest(a),finite=all(np.isfinite(v).all() for v in a.values()),
                deployed_error=err,review_required=err['relative_l2']>.20,
                gpu_saved_vs_independent_feature=agreement,
                cpu_fp32_vs_saved_gpu=output_errors(fp32,a['actual'],qm),
                cpu_fp64_vs_independent_feature=output_errors(fp64,f,qm),
                cpu_fp64_vs_exact_softmax=output_errors(fp64,e,qm),
                per_head_query_relative_l2=describe(per_query[eligible]),
                bounded_prefix_norms=norms,representation_drift_from_initial=drift,
                kernel_statistics=stats)
            rows.append(row)
            if step==20:
                finals.append(row)
                for m in (128,256,512):
                    for seed in range(8):
                        omega=orthogonal_gaussian_projection(a['q'].shape[1],m,a['q'].shape[-1],seed).numpy()
                        feature,exact,stat=independent_attention(a['q'],a['k'],a['v'],omega,qm,km)
                        feature_grid.append(dict(example=example,features=m,projection_seed=seed,
                            output_error=output_errors(feature,exact,qm),
                            median_pair_ess=stat['feature_ess']['median']))
    aggregated=[]
    for step in (0,5,10,20):
        subset=[r for r in rows if r['update']==step]
        aggregated.append(dict(update=step,
            relative_l2_median=float(np.median([r['deployed_error']['relative_l2'] for r in subset])),
            relative_l2_max=max(r['deployed_error']['relative_l2'] for r in subset),
            review_count=sum(r['review_required'] for r in subset),
            unique_raw_qkv_prefixes=len({r['frozen_qkv_mask_digest'] for r in subset}),
            median_pair_ess=float(np.median([r['kernel_statistics']['feature_ess']['median'] for r in subset]))))
    grid_summary=[]
    for m in (128,256,512):
        subset=[r for r in feature_grid if r['features']==m]
        grid_summary.append(dict(features=m,cases=len(subset),
            relative_l2=describe([r['output_error']['relative_l2'] for r in subset]),
            review_count=sum(r['output_error']['relative_l2']>.20 for r in subset),
            per_example=[dict(example=i,errors_by_seed=[r['output_error']['relative_l2'] for r in subset if r['example']==i]) for i in range(6)]))
    result=dict(status='REVIEW_RETAINED; NO_FLOATING_POINT_KERNEL_DEFECT_OBSERVED_IN_BOUNDED_CASES',
        thresholds_unchanged=True,review_relative_l2=.20,aggregated=aggregated,
        maximum_saved_gpu_feature_kernel_abs=max(r['gpu_saved_vs_independent_feature']['max_abs'] for r in rows),
        maximum_fp64_feature_kernel_abs=max(r['cpu_fp64_vs_independent_feature']['max_abs'] for r in rows),
        maximum_cpu_fp32_saved_gpu_abs=max(r['cpu_fp32_vs_saved_gpu']['max_abs'] for r in rows),
        frozen_feature_count_comparison=grid_summary,rows=rows,feature_grid=feature_grid,
        projection_comparison_policy='Frozen update20 QKV, diagnostic seeds0..7; deployed seed42 unchanged. Generation streams are not nested across feature counts. This is not independent training or feature selection.',
        scope='Action attention raw outputs only, six frozen train examples/first32 positions; no real-data model forward, no optimizer update, no validation/test inspection')
    publish('ACTION_APPROXIMATION_REVIEW.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','feature_grid','frozen_feature_count_comparison')},indent=2))


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):
        raise RuntimeError('CPU review requires CUDA hidden')
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    stamp=time.perf_counter()
    checkpoints();action_review()
    assert not torch.cuda.is_initialized()
    publish('CPU_REVIEW_EXECUTION.json',dict(status='PASS',seconds=time.perf_counter()-stamp,
        cuda_initialized=False,optimizer_calls=0,script_sha256=sha256(Path(__file__).resolve())))
