#!/usr/bin/env python3
"""Staged GPU verification after allocation and bounded-action approval.

An allocation record is an operator-supplied assertion backed by scheduler/
provider evidence or explicit allocation-owner transfer, not an allocator or
automatic proof of ownership. --plan-only is CPU-safe.
"""
import argparse
import copy
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ktbench.batching import Session,SessionExample,collate_hitskt
from ktbench.config import PROJECT_SEED
from ktbench.favor_plus_gpu_access import CUDA_VALIDATION,require_gpu_authorization,require_execution_access
from ktbench.favor_plus_pipeline import fresh_run,model_config,runtime,sha256,write_json,source_identity
from ktbench.metrics import masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig,causal_favor_attention,orthogonal_gaussian_projection
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from scripts.favor_plus_quality_common import independent_attention,torch_feature_reference

LENGTHS=(17,65,257,939,3925,13081)


def admit_allocation(record,visible,*,now=None):
    """Pure metadata validation; performs zero CUDA API calls."""
    now=now or dt.datetime.now(dt.timezone.utc)
    if record.get('allocation_kind')=='owner_confirmed_existing_research_allocation':
        return require_gpu_authorization(record,visible,CUDA_VALIDATION,now=now)
    if (record.get('exclusive') is not True or not record.get('owner') or not record.get('external_allocation_evidence')
        or not record.get('resource_uuid','').startswith(('MIG-','GPU-'))):
        raise ValueError('exclusive independent ownership evidence is required')
    expires=dt.datetime.fromisoformat(record.get('expires_utc','').replace('Z','+00:00'))
    if expires.tzinfo is None or expires<=now:raise ValueError('allocation expired or lacks timezone')
    if visible!=record['resource_uuid']:raise ValueError('CUDA_VISIBLE_DEVICES must exactly identify the allocated UUID')
    return record['resource_uuid']


def plan():
    return dict(status='PENDING',cuda_initialized=False,resource='external exclusive allocation required',
        lengths=list(LENGTHS),features=[128,256,512],batch=1,heads=2,head_width=64,chunks=[16,32,64],
        dtype='float32, TF32 disabled',max_gpu_allocated_bytes=8*1024**3,max_stage_seconds=120,
        output_max_abs_tolerance=5e-5,gradient_atol=5e-5,gradient_rtol=1e-3,
        steps=['CPU/NumPy oracle and CUDA outputs/gradients','causal and padded prefix invariance',
               'short-to-long feature attention memory/runtime','full hierarchy own-label shifting/backward',
               'tagged checkpoint exact reload'],stop_on_first_failure=True,
        allocation_record_fields=['resource_uuid','owner','exclusive','expires_utc','external_allocation_evidence'],
        additional_required_access_fields=['independent','job_id','starts_utc','user_approval'],
        limitation='metadata validation is not independent proof; external allocation must be established first')


def execute(record,run_id,features,profile):
    admit_allocation(record,os.environ.get('CUDA_VISIBLE_DEVICES'))  # BEFORE CUDA or CUDA RNG checks
    require_execution_access(record,os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_VALIDATION)
    out=fresh_run(profile,run_id)
    write_json(out/'allocation_record.json',record)
    dev=torch.device('cuda:0')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    rows=[];started_all=time.perf_counter();phase='CUDA initialization'
    try:
        torch.set_num_threads(1)
        torch.manual_seed(PROJECT_SEED)
        capacity=torch.cuda.get_device_properties(dev).total_memory
        if capacity<10*1024**3:raise MemoryError('8GiB budget plus runtime headroom required')
        torch.cuda.set_per_process_memory_fraction(8*1024**3/capacity,dev)
        for length in LENGTHS:
            for chunk in (16,32,64):
                phase=f'kernel_{length}_{chunk}'
                admit_allocation(record,os.environ.get('CUDA_VISIBLE_DEVICES'))
                require_execution_access(record,os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_VALIDATION)
                torch.manual_seed(42)
                cpu=[(torch.randn(1,2,length,64)*.35).requires_grad_() for _ in range(3)]
                omega=orthogonal_gaussian_projection(2,features,64,42).float()
                mask=torch.ones(1,length,dtype=torch.bool);mask[:,1::7]=False
                expected=causal_favor_attention(*cpu,omega,query_mask=mask,key_mask=mask,chunk_size=chunk)
                cg=torch.autograd.grad(expected.square().mean(),cpu)
                gpu=[t.detach().to(dev).requires_grad_() for t in cpu]
                om=omega.to(dev);gm=mask.to(dev)
                torch.cuda.reset_peak_memory_stats(dev);torch.cuda.synchronize(dev)
                started=time.perf_counter()
                actual=causal_favor_attention(*gpu,om,query_mask=gm,key_mask=gm,chunk_size=chunk)
                grads=torch.autograd.grad(actual.square().mean(),gpu)
                torch.cuda.synchronize(dev);elapsed=time.perf_counter()-started
                delta=float((actual.detach().cpu()-expected.detach()).abs().max())
                gd=[float((a.cpu()-b).abs().max()) for a,b in zip(grads,cg)]
                assert delta<=5e-5 and all(torch.allclose(a.cpu(),b,atol=5e-5,rtol=1e-3) for a,b in zip(grads,cg))
                assert torch.isfinite(actual).all() and all(torch.isfinite(g).all() for g in grads)
                assert actual.masked_select(~gm[:,None,:,None].expand_as(actual)).eq(0).all()
                if length<=65:
                    reference,_,_=independent_attention(*[t.detach().numpy() for t in cpu],omega.numpy(),mask.numpy(),mask.numpy())
                    assert np.max(np.abs(actual.detach().cpu().numpy()-reference))<=5e-5
                    explicit_inputs=[t.detach().double().requires_grad_() for t in cpu]
                    explicit=torch_feature_reference(*explicit_inputs,omega.double(),mask,mask)
                    explicit_grads=torch.autograd.grad(explicit.square().mean(),explicit_inputs)
                    reference_gradient_deltas=[float((g.detach().cpu().double()-e).abs().max())
                                               for g,e in zip(grads,explicit_grads)]
                    assert all(torch.allclose(g.detach().cpu().double(),e,atol=5e-5,rtol=1e-3)
                               for g,e in zip(grads,explicit_grads))
                # Identical prefix with arbitrarily changed future Q/K/V.
                boundary=length//2
                if length<=65:
                    probe=causal_favor_attention(*gpu,om,query_mask=gm,key_mask=gm,chunk_size=chunk)
                    causal_grads=torch.autograd.grad(probe[:,:,:boundary].sum(),gpu)
                    assert all(g[:,:,boundary:].eq(0).all() for g in causal_grads)
                perturbed=[g.detach().clone() for g in gpu]
                for t in perturbed:t[:,:,boundary:]+=100
                with torch.no_grad():
                    changed=causal_favor_attention(*perturbed,om,query_mask=gm,key_mask=gm,chunk_size=chunk)
                assert torch.equal(actual[:,:,:boundary],changed[:,:,:boundary])
                peak=torch.cuda.max_memory_allocated(dev)
                assert peak<=8*1024**3 and elapsed<=120
                row=dict(length=length,chunk=chunk,features=features,max_abs_output=delta,
                         max_abs_gradients=gd,seconds=elapsed,peak_allocated_bytes=peak,status='PASS',
                         independent_gradient_max_abs=reference_gradient_deltas if length<=65 else None,
                         output_and_gradients_finite=True,positions_per_second=length/elapsed)
                rows.append(row);write_json(out/f'kernel_{length}_{chunk}.json',row)
                print(json.dumps(row),flush=True)
                del cpu,gpu,actual,expected,changed,grads,cg,perturbed
        require_execution_access(record,os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_VALIDATION)
        phase='full_model';model_started=time.perf_counter();torch.cuda.reset_peak_memory_stats(dev)
        cfg=model_config(profile,20,5)
        cpu_model=HiTSKTFavorPlus(cfg,FavorPlusConfig(features=features)).eval()
        model=copy.deepcopy(cpu_model).to(dev)
        examples=[SessionExample([Session([1,2],[1,2],[0,1])],Session([3,4,5,6],[1]*4,[1,0,1,0]),0),
                  SessionExample([Session([1],[1],[0]),Session([2,3,4],[1]*3,[1,0,1])],Session([2,3],[1,1],[1,0]),0)]
        cpu_batch=collate_hitskt(examples,num_questions=20,num_skills=5)
        batch=cpu_batch.to(dev)
        cpu_logits=cpu_model(cpu_batch)
        masked_bce_loss(cpu_logits,cpu_batch.target_labels,cpu_batch.target_metric_mask).backward()
        logits=model(batch)
        masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.allclose(logits.cpu(),cpu_logits,atol=5e-5,rtol=0)
        assert all(torch.allclose(a.grad.cpu(),b.grad,atol=5e-5,rtol=1e-3)
                   for a,b in zip(model.parameters(),cpu_model.parameters()))
        model_gradient_delta=max(float((a.grad.cpu()-b.grad).abs().max())
                                 for a,b in zip(model.parameters(),cpu_model.parameters()))
        # Re-collate raw responses to exercise the real BOS/previous-response/EOS
        # construction, including each prediction's own label and batch isolation.
        assert cpu_batch.target_correct_input[:,0].eq(4).all()
        assert int(cpu_batch.target_metric_mask.sum())==6
        label_checks=[]
        for position in range(4):
            require_execution_access(record,os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_VALIDATION)
            altered_examples=copy.deepcopy(examples)
            altered_examples[0].target.correct[position]=1-altered_examples[0].target.correct[position]
            altered=collate_hitskt(altered_examples,num_questions=20,num_skills=5).to(dev)
            with torch.no_grad():changed_logits=model(altered)
            assert torch.equal(logits[0,:position+1],changed_logits[0,:position+1])
            assert torch.equal(logits[1],changed_logits[1])
            retained=[]
            def retain(module,args,output):retained.append(output)
            handle=model.correct_padding_encoder.correct.register_forward_hook(retain)
            try:
                model.zero_grad(set_to_none=True)
                current=model(batch)
                response_grad=torch.autograd.grad(current[0,position],retained[0])[0]
                assert response_grad[0,position+1:].eq(0).all() and response_grad[1].eq(0).all()
            finally:handle.remove()
            label_checks.append(dict(response_position=position,own_and_earlier_predictions_exact=True,
                other_batch_row_exact=True,future_correctness_embedding_gradients_zero=True))
        with torch.no_grad():
            padded=copy.deepcopy(batch)
            padded.target_correct_input[~padded.target_attention_mask]=1
            padded.target_question[~padded.target_attention_mask]=1
            padded.target_skill[~padded.target_attention_mask]=1
            padded.history_correct[~padded.history_mask]=1
            padded.history_question[~padded.history_mask]=1
            padded.history_skill[~padded.history_mask]=1
            assert torch.equal(logits.masked_select(batch.target_metric_mask),
                model(padded).masked_select(batch.target_metric_mask))
        write_json(out/'full_model_causal_checks.json',dict(status='PASS',raw_response_checks=label_checks,
            bos_eos_metric_count_and_padding='PASS',variable_action_session_target_lengths=True))
        phase='checkpoint_reload'
        torch.save(model.checkpoint(),out/'gpu_model.pt')
        restored=HiTSKTFavorPlus.from_checkpoint(out/'gpu_model.pt').to(dev).eval()
        with torch.no_grad():assert torch.equal(logits,restored(batch))
        require_execution_access(record,os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_VALIDATION)
        full_model_peak=torch.cuda.max_memory_allocated(dev)
        assert full_model_peak<=8*1024**3
        result=dict(status='PASS',scope='bounded synthetic GPU verification',kernels=rows,
                    profile=profile,features=features,allocation_resource_uuid=record['resource_uuid'],
                    source=source_identity(),access_guard_sha256=sha256(Path(__file__).resolve().parents[1]/'ktbench/favor_plus_gpu_access.py'),
                    oracle_sha256=sha256(Path(__file__).with_name('favor_plus_quality_common.py')),
                    full_model_shift_gradients_checkpoint='PASS',runtime=runtime(dev),
                    full_model_cpu_cuda_max_abs_logit_difference=float((logits.cpu()-cpu_logits).abs().max()),
                    full_model_cpu_cuda_max_abs_parameter_gradient_difference=model_gradient_delta,
                    raw_response_own_label_future_padding_checks='PASS',full_model_peak_allocated_bytes=full_model_peak,
                    full_model_wall_seconds=time.perf_counter()-model_started,
                    total_wall_seconds=time.perf_counter()-started_all,
                    checkpoint_sha256=sha256(out/'gpu_model.pt'),script_sha256=sha256(Path(__file__)))
        write_json(out/'gpu_validation.json',result)
        return result
    except Exception as exc:
        write_json(out/'gpu_validation_failure.json',dict(status='FAIL',phase=phase,completed=rows,
            failure_type=type(exc).__name__,message=str(exc),action='stop; no larger stage executed'))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--allocation-record',type=Path);p.add_argument('--run-id')
    p.add_argument('--features',type=int,choices=(128,256,512),default=128)
    p.add_argument('--profile',choices=('assist2017','junyi','ednet_kt1'),default='junyi')
    args=p.parse_args()
    if args.plan_only:
        if not args.output:p.error('--plan-only requires --output')
        write_json(args.output,plan());print('PENDING: CPU-safe plan written; no CUDA calls');return
    if not args.allocation_record or not args.run_id:p.error('external allocation record and fresh run id required')
    print(json.dumps(execute(json.loads(args.allocation_record.read_text()),args.run_id,args.features,args.profile),indent=2))


if __name__=='__main__':main()
