#!/usr/bin/env python3
"""Summarize completed receipts; no model, CUDA, data loader or optimizer."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/four_arm_cuda_readiness_fp64_20261024_v2'
OUT=ROOT/'reports/favor_plus_v1/four_arm_cuda_comparison_20261024_v1/fp64_v2'
ARMS=('orf128','orf512','anti128','anti512')
STAGES=('ActionEncoder','SessionEncoder','CorrectPaddingEncoder','Decoder')


def main():
    arms={};summary=[]
    for arm in ARMS:
        result=json.loads((RUN/arm/'cuda_result.json').read_text())
        comparison=json.loads((RUN/f'{arm}_attention/comparison_result.json').read_text())
        kernels=json.loads((RUN/arm/'kernel_results.json').read_text())['checks']
        full=json.loads((RUN/arm/'full_model_results.json').read_text())
        if result['status']!='PASS_CUDA_ZERO_UPDATES' or result['optimizer_updates']!=0:raise ValueError('incomplete verification')
        records=comparison['records'];groups={}
        for r in records:groups.setdefault((r['input_path'],r['norm_scale']),[]).append(r)
        for scale in (.5,1.,1.5):
            for scope,limit in (('prefix32',32),('prefix128',128),('full_selected_row',None)):
                chosen=[]
                for (path,s),rows in groups.items():
                    if s!=scale:continue
                    n=max(r['length'] for r in rows);length=n if limit is None else min(n,limit)
                    chosen.append(next(r for r in rows if r['length']==length))
                for stage,name in enumerate(STAGES):
                    rr=[r for r in chosen if r['stage']==stage];errors=[r['metrics']['relative_output_l2'] for r in rr]
                    summary.append(dict(arm=arm,stage=name,norm_scale=scale,scope=scope,cases=len(rr),lengths=[r['length'] for r in rr],
                        median_relative_l2=float(np.median(errors)),mean_relative_l2=float(np.mean(errors)),max_relative_l2=max(errors),review_cases=sum(x>.20 for x in errors),
                        median_total_variation=float(np.median([r['metrics']['attention_total_variation']['median'] for r in rr])),
                        median_feature_ess=float(np.median([r['metrics']['pair_feature_ess']['median'] for r in rr])),
                        median_query_p95=float(np.median([r['metrics']['query_norm']['p95'] for r in rr])),median_key_p95=float(np.median([r['metrics']['key_norm']['p95'] for r in rr]))))
        allocator=result['resources']['own_allocator']
        telemetry=[json.loads(p.read_text()) for p in (RUN/arm).glob('resources_*.json')]
        arms[arm]=dict(kernel_cases=len(kernels),verification_seconds=result['seconds'],peak_allocated_mib=allocator['peak_allocated_bytes']/2**20,peak_reserved_mib=allocator['peak_reserved_bytes']/2**20,
            max_kernel_cpu_cuda_abs=max(r['output_max_abs'] for r in kernels),max_kernel_cpu_cuda_gradient_abs=max(max(r['gradient_max_abs']) for r in kernels),
            full_model_cpu_cuda_abs=full['cpu_cuda_logit_max_abs'],full_model_cpu_cuda_gradient_abs=full['parameter_gradient_max_abs'],
            max_real_feature_reference_abs=max(r['metrics']['production_vs_feature_max_abs'] for r in records),
            max_profile_forward_backward_seconds=max(r['forward_backward_seconds'] for r in result['profile']),
            profile=result['profile'],peak_host_rss_mib=max(r['host']['peak_process_rss_bytes'] for r in telemetry)/2**20,
            min_sampled_free_gib=min(r['observation']['device']['free_bytes'] for r in telemetry)/2**30,
            exact_fresh_recovery=json.loads((RUN/f'{arm}_recovery/recovery_result.json').read_text())['status'],optimizer_updates=0)
    output=dict(status='ALL_FOUR_FP64_CUDA_PASS_ZERO_UPDATES',review_threshold=.20,arms=arms,approximation=summary,
        scope='same14TRAIN initial representations/fourstages; same commonpool seed42; correlated prefixes/scales, not independent replicates or learned validation results',
        native_fp32_status='ORF512 strict stress failed; antithetic FP32 unrun; failure preserved separately',adam_step_measured=False)
    (OUT/'COMPARISON_SUMMARY.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    lines=['| Stage | ORF128 | ORF512 | Antithetic128 | Antithetic512 |','|---|---:|---:|---:|---:|']
    for stage in STAGES:
        cells=[]
        for arm in ARMS:
            r=next(x for x in summary if x['arm']==arm and x['stage']==stage and x['norm_scale']==1 and x['scope']=='prefix128')
            cells.append(f"{r['median_relative_l2']:.6f} ({r['review_cases']}/{r['cases']} REVIEW)")
        lines.append('| '+stage+' | '+' | '.join(cells)+' |')
    (OUT/'APPROXIMATION_TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
