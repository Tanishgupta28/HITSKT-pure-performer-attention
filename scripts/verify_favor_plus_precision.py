#!/usr/bin/env python3
"""Independent float64-gradient controls for float32 FAVOR+, no weight updates."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ktbench.favor_plus_pipeline import ROOT,sha256,write_json
from ktbench.models.favor_plus import causal_favor_attention,orthogonal_gaussian_projection
from scripts.favor_plus_quality_common import torch_feature_reference


def check(q,k,v,m,*,source):
    omega=orthogonal_gaussian_projection(q.shape[1],m,q.shape[-1],0).float()
    mask=torch.ones(q.shape[0],q.shape[2],dtype=torch.bool)
    raw=[a.detach().float().requires_grad_() for a in (q,k,v)]
    ref=[a.detach().double().requires_grad_() for a in raw]
    exact=torch_feature_reference(*ref,omega.double(),mask,mask)
    reference_grad=torch.autograd.grad(exact.square().mean(),ref)
    rows=[]
    for precision in ('float32','float64'):
        out=causal_favor_attention(*raw,omega,query_mask=mask,key_mask=mask,chunk_size=5,
                                  accumulation_dtype=precision)
        gradients=torch.autograd.grad(out.square().mean(),raw)
        delta=float((out.double()-exact.detach()).abs().max())
        gd=[float((a.double()-b).abs().max()) for a,b in zip(gradients,reference_grad)]
        finite=bool(torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in gradients))
        passed=finite and delta<=5e-5 and all(torch.allclose(a.double(),b,atol=5e-5,rtol=1e-3)
                                             for a,b in zip(gradients,reference_grad))
        rows.append(dict(accumulation=precision,output_max_abs_vs_independent64=delta,
            gradient_max_abs_vs_independent64=dict(zip(('q','k','v'),gd)),finite=finite,
            status='PASS' if passed else 'FAIL'))
    x=raw[1].detach().double()/q.shape[-1]**.25
    naive_log=torch.einsum('bhld,hmd->bhlm',x,omega.double())-x.square().sum(-1,keepdim=True)/2-np.log(m)/2
    return dict(source=source,features=m,shape=list(q.shape),checks=rows,
                naive_float32_key_features_zero_fraction=float((naive_log.float().exp()==0).float().mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only; CUDA must be hidden')
    torch.set_num_threads(1);torch.manual_seed(913)
    raw=[torch.randn(1,2,17,64) for _ in range(3)]
    rows=[]
    for m in (128,256,512):
        for scale in (1.,30.,1000.):
            q,k,v=[a*scale if i<2 else a for i,a in enumerate(raw)]
            rows.append(check(q,k,v,m,source=f'synthetic_scale_{scale}'))
    samples=json.loads(args.sample_summary.read_text())
    for dataset in samples['datasets']:
        manifest=json.loads((ROOT/dataset['manifest']).read_text())
        for source,source_data in manifest['sources'].items():
            for stage,context in source_data['contexts'].items():
                if not stage.startswith('decoder'):continue
                frozen=np.load(ROOT/context['artifact'])
                tensors=[torch.from_numpy(frozen[key]) for key in ('q','k','v')]
                assert frozen['query_mask'].all() and frozen['key_mask'].all()
                for m in (128,256,512):
                    rows.append(check(*tensors,m,source=f"{dataset['dataset']}:{source}:{stage}"))
    status='PASS' if all(r['status']=='PASS' for row in rows for r in row['checks']) else 'FAIL'
    write_json(args.output,dict(status=status,cases=rows,checks=len(rows)*2,source_sha256=sha256(Path(__file__)),
        sample_summary_sha256=sha256(args.sample_summary),oracle_sha256=sha256(Path('scripts/favor_plus_quality_common.py')),
        output_tolerance=5e-5,gradient_atol=5e-5,gradient_rtol=1e-3,no_model_updates=True))
    print(status,len(rows)*2,'output/gradient precision controls')


if __name__=='__main__':main()
