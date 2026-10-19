#!/usr/bin/env python3
"""Predeclared CPU-only approximation/distribution/precision study. No training."""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from ktbench.favor_plus_pipeline import sha256,write_json,runtime
from ktbench.models.favor_plus import causal_favor_attention,orthogonal_gaussian_projection
from scripts.favor_plus_quality_common import describe,independent_attention,output_errors,torch_feature_reference

DESIGN = Path('reports/favor_plus_v1/numerical_quality_v1/PREDECLARED_DESIGN.json')


def provenance():
    names = [Path(__file__),Path('scripts/favor_plus_quality_common.py'),Path('ktbench/models/favor_plus.py')]
    return dict(design_sha256=sha256(DESIGN),predeclared_commit='49ed057',
                source_sha256={str(p):sha256(p) for p in names},runtime=runtime(torch.device('cpu')))


def distribution(config):
    rows = torch.cat([orthogonal_gaussian_projection(2,512,64,s).reshape(-1,64)
                      for s in config['distribution']['seeds']]).numpy()
    unit = rows/np.linalg.norm(rows,axis=-1,keepdims=True)
    blocks = unit.reshape(-1,64,64)
    orthogonal_error = float(np.max(np.abs(blocks@blocks.transpose(0,2,1)-np.eye(64))))
    covariance = np.cov(rows,rowvar=False)
    off = covariance-np.diag(np.diag(covariance))
    values = np.sort(rows.reshape(-1))
    cdf_distance = max(np.max(np.arange(1,len(values)+1)/len(values)-ndtr(values)),
                       np.max(ndtr(values)-np.arange(len(values))/len(values)))
    moments = dict(mean=float(rows.mean()),variance=float(rows.var()),fourth_moment=float((rows**4).mean()),
                   mean_squared_radius=float((rows**2).sum(-1).mean()),
                   variance_squared_radius=float((rows**2).sum(-1).var()),
                   max_coordinate_mean_abs=float(np.max(np.abs(rows.mean(0)))),
                   covariance_diagonal_max_error=float(np.max(np.abs(np.diag(covariance)-1))),
                   covariance_off_diagonal_max_abs=float(np.max(np.abs(off))),
                   normal_empirical_cdf_max_distance=float(cdf_distance),
                   within_block_orthogonality_max_abs=orthogonal_error)
    passed = (abs(moments['mean'])<.02 and abs(moments['variance']-1)<.03
              and abs(moments['fourth_moment']-3)<.12 and abs(moments['mean_squared_radius']-64)<1
              and abs(moments['variance_squared_radius']-128)<10 and orthogonal_error<2e-12)
    # Exact identities: zero and antiparallel inputs. These do not depend on
    # Monte Carlo convergence and independently detect wrong radial/scaling math.
    identity = []
    for m in config['features']:
        omega = orthogonal_gaussian_projection(2,m,64,0).numpy()
        for radius in (0.,.5,3.):
            x = np.zeros(64);x[0] = radius
            a = np.exp(omega@x-np.sum(x*x)/2)/m**.5
            b = np.exp(omega@(-x)-np.sum(x*x)/2)/m**.5
            estimate = np.sum(a*b,-1)
            expected = np.exp(-radius**2)
            error = float(np.max(np.abs(estimate/expected-1)))
            identity.append(dict(features=m,radius=radius,relative_error=error))
    return dict(status='PASS' if passed and max(r['relative_error'] for r in identity)<1e-12 else 'FAIL',
                moments=moments,antiparallel_kernel_identities=identity,
                note='Descriptive pooled moments/CDF, not an iid significance test; within-block rows dependent')


def compare(q,k,v,omega,qm,km,*,chunk=32,gradient=False):
    arrays = [t.detach().double().cpu().numpy() for t in (q,k,v,omega)]
    feature,exact,quality = independent_attention(*arrays,qm.cpu().numpy(),km.cpu().numpy())
    inputs = [t.detach().clone().requires_grad_(gradient) for t in (q,k,v)]
    actual = causal_favor_attention(*inputs,omega,query_mask=qm,key_mask=km,chunk_size=chunk)
    out = actual.detach().double().numpy()
    quality.update(implementation_vs_feature_reference=output_errors(out,feature,qm.numpy()),
                   feature_vs_exact_softmax=output_errors(feature,exact,qm.numpy()),
                   finite_output=bool(np.isfinite(out).all()))
    passed = np.allclose(out,feature,atol=2e-12,rtol=2e-11) if q.dtype == torch.float64 else float(np.max(np.abs(out-feature))) <= 5e-5
    if gradient:
        refs = [t.detach().clone().requires_grad_() for t in (q,k,v)]
        expected = torch_feature_reference(*refs,omega,qm,km)
        multiplier = torch.linspace(-.7,.9,actual.numel(),dtype=q.dtype).reshape_as(actual)
        ga = torch.autograd.grad((actual*multiplier).sum(),inputs)
        ge = torch.autograd.grad((expected*multiplier).sum(),refs)
        errors = [float((a-b).abs().max()) for a,b in zip(ga,ge)]
        gradient_pass = all(torch.allclose(a,b,atol=3e-11,rtol=3e-10) for a,b in zip(ga,ge))
        quality['gradient_max_abs'] = dict(zip(('query','key','value'),errors))
        quality['finite_gradients'] = all(bool(torch.isfinite(t).all()) for t in ga)
        passed = passed and gradient_pass and quality['finite_gradients']
    quality['implementation_status'] = 'PASS' if passed and quality['finite_output'] else 'FAIL'
    return quality


def synthetic(config):
    records = []
    for input_seed in config['input_seeds']:
        rng = np.random.default_rng(input_seed)
        for length in config['synthetic']['lengths']:
            raw = [rng.standard_normal((1,2,length,64)) for _ in range(3)]
            qm = km = torch.ones(1,length,dtype=torch.bool)
            for scale in config['synthetic']['scales']:
                q,k,v = [torch.from_numpy(a*scale if i<2 else a) for i,a in enumerate(raw)]
                for m in config['features']:
                    for seed in config['projection_seeds']:
                        omega = orthogonal_gaussian_projection(2,m,64,seed)
                        gradients = input_seed==91 and length==16 and scale in (1.,3.) and seed==0
                        record = compare(q,k,v,omega,qm,km,gradient=gradients)
                        record.update(input_seed=input_seed,length=length,scale=scale,features=m,seed=seed,
                            query_norm=describe(q.norm(dim=-1).numpy()),key_norm=describe(k.norm(dim=-1).numpy()))
                        records.append(record)
    return records


def precision(config):
    torch.manual_seed(913)
    raw = [torch.randn(1,2,17,64) for _ in range(3)]
    mask = torch.ones(1,17,dtype=torch.bool)
    result = []
    for m in config['features']:
        omega = orthogonal_gaussian_projection(2,m,64,0).float()
        for scale in config['numerical_controls']['stress_scales']:
            inputs = [(r*scale if i<2 else r).requires_grad_() for i,r in enumerate(raw)]
            # Identical rounded inputs and projections isolate arithmetic dtype.
            feature,_,_ = independent_attention(*[r.detach().double().numpy() for r in inputs],
                omega.double().numpy(),mask.numpy(),mask.numpy())
            outputs = [];gradients = []
            for accumulation in ('float32','float64'):
                out = causal_favor_attention(*inputs,omega,query_mask=mask,key_mask=mask,
                                             chunk_size=5,accumulation_dtype=accumulation)
                grads = torch.autograd.grad(out.square().mean(),inputs)
                outputs.append(out.detach().numpy());gradients.append(grads)
            finite = all(np.isfinite(o).all() for o in outputs) and all(
                bool(torch.isfinite(g).all()) for group in gradients for g in group)
            difference = [float(np.max(np.abs(o-feature))) for o in outputs]
            gdelta = [float((a-b).abs().max()) for a,b in zip(*gradients)]
            result.append(dict(features=m,scale=scale,finite_outputs_and_gradients=finite,
                float32_vs_float64_reference_max_abs=difference[0],
                float64_accumulation_cast_to_float32_vs_reference_max_abs=difference[1],
                gradient_max_abs_between_accumulations=dict(zip(('q','k','v'),gdelta)),
                output_gate_status='PASS' if finite and difference[0]<=5e-5 else 'FAIL'))
    return result


def benchmark(length,m):
    torch.manual_seed(42)
    args = [torch.randn(1,2,length,64,requires_grad=True) for _ in range(3)]
    omega = orthogonal_gaussian_projection(2,m,64,42).float()
    mask = torch.ones(1,length,dtype=torch.bool)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    start = time.perf_counter()
    out = causal_favor_attention(*args,omega,query_mask=mask,key_mask=mask,chunk_size=32)
    forward = time.perf_counter()-start
    start = time.perf_counter();out.square().mean().backward();backward=time.perf_counter()-start
    finite = bool(torch.isfinite(out).all() and all(torch.isfinite(a.grad).all() for a in args))
    return dict(length=length,features=m,forward_seconds=forward,backward_seconds=backward,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                baseline_rss_bytes=baseline,finite_outputs_and_gradients=finite)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--benchmark-length',type=int)
    p.add_argument('--features',type=int,choices=(128,256,512))
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise RuntimeError('CPU study requires CUDA hidden before process start')
    torch.set_num_threads(1)
    if args.benchmark_length:
        print(json.dumps(benchmark(args.benchmark_length,args.features)));return
    if not args.output:p.error('--output required')
    config = json.loads(DESIGN.read_text())
    dist = distribution(config);print('distribution',dist['status'],flush=True)
    synth = synthetic(config);print('synthetic comparisons',len(synth),flush=True)
    numeric = precision(config);print('precision complete',flush=True)
    bench=[]
    for length in config['runtime_memory']['lengths_including_eos']:
        for m in config['features']:
            child = subprocess.run([sys.executable,__file__,'--benchmark-length',str(length),'--features',str(m)],
                                   check=True,text=True,capture_output=True,timeout=180)
            bench.append(json.loads(child.stdout));print('benchmark',length,m,bench[-1],flush=True)
    status = 'PASS' if dist['status']=='PASS' and all(r['implementation_status']=='PASS' for r in synth) and all(r['finite_outputs_and_gradients'] for r in numeric+bench) else 'FAIL'
    write_json(args.output,dict(status=status,distribution=dist,synthetic=synth,precision=numeric,
        resource_benchmarks=bench,provenance=provenance(),no_optimizer_steps=True,no_test_data=True))
    print('COMPLETE',status,flush=True)


if __name__=='__main__':main()
