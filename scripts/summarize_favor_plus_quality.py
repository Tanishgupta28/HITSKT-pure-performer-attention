#!/usr/bin/env python3
"""Apply the frozen feature-count decision rule; no model selection by AUC."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from ktbench.favor_plus_pipeline import sha256,write_json
from ktbench.models.favor_plus import orthogonal_gaussian_projection


def kernel_moments():
    """Numerical 1D Gaussian integral and independent NumPy iid MC control.

    Lognormal variance is a mathematical iid reference; ORF rows are dependent.
    Empirical means here are finite samples, not proof of bias/unbiasedness.
    """
    rows=np.concatenate([orthogonal_gaussian_projection(2,512,64,s).reshape(-1,64).numpy()
                         for s in range(32)])
    iid=np.random.default_rng(824).standard_normal(rows.shape)
    results=[]
    for radius in (.25,.5,1.,3.):
        # parallel x=y=(radius,0,...), raw q=x*d**.25
        truth=np.exp(radius**2)
        integrand=lambda t:np.exp(2*radius*t-radius**2-t*t/2)/np.sqrt(2*np.pi)
        integrated,_=quad(integrand,-32.,32.,epsabs=1e-10,epsrel=1e-12)
        estimated=np.exp(2*radius*rows[:,0]-radius**2)
        iid_est=np.exp(2*radius*iid[:,0]-radius**2)
        results.append(dict(radius=radius,exact_kernel=float(truth),gaussian_quadrature=float(integrated),
            quadrature_relative_error=float(abs(integrated/truth-1)),
            orf_empirical_mean_to_truth=float(estimated.mean()/truth),iid_empirical_mean_to_truth=float(iid_est.mean()/truth),
            iid_relative_variance_per_feature=float(np.expm1(4*radius**2)),
            orf_top_one_percent_mass_fraction=float(np.sort(estimated)[-max(1,len(estimated)//100):].sum()/estimated.sum()),
            number_of_rows=len(rows)))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root
    design=json.loads((root/'PREDECLARED_DESIGN.json').read_text())
    synth=json.loads((root/'synthetic_quality.json').read_text())
    samples=json.loads((root/'training_sample_summary.json').read_text())
    precision=json.loads((root/'precision_controls.json').read_text())
    contexts=[];datasets={};all_contexts=[]
    for dataset in samples['datasets']:
        manifest=json.loads(Path(dataset['manifest']).read_text())
        initial=manifest['sources']['untrained_favor_seed42_m128']['contexts']
        contexts.extend(initial.values())
        all_contexts.extend(c for s in manifest['sources'].values() for c in s['contexts'].values())
        datasets[dataset['dataset']]=dict(selected=manifest['selected'],sources={})
        for source,source_data in manifest['sources'].items():
            ds={}
            for m in design['features']:
                rs=[r for c in source_data['contexts'].values() for r in c['results'] if r['features']==m]
                ds[m]=dict(mean_mse=float(np.mean([r['feature_vs_exact_softmax']['mse'] for r in rs])),
                    mean_relative_l2=float(np.mean([r['feature_vs_exact_softmax']['relative_l2'] for r in rs])))
            datasets[dataset['dataset']]['sources'][source]=dict(norms=source_data['norms'],feature_summary=ds,
                checkpoint_sha256=source_data['checkpoint_sha256'],note=source_data['weight_note'])
    numeric_pass=synth['status']=='PASS' and samples['status']=='PASS' and precision['status']=='PASS'
    comparison={};baseline_mse=None
    for m in design['features']:
        bycontext=[np.mean([r['feature_vs_exact_softmax']['relative_l2'] for r in c['results'] if r['features']==m]) for c in contexts]
        rs=[r for c in contexts for r in c['results'] if r['features']==m]
        mean_mse=float(np.mean([r['feature_vs_exact_softmax']['mse'] for r in rs]))
        if m==128:baseline_mse=mean_mse
        benchmarks=[r for r in synth['resource_benchmarks'] if r['features']==m]
        ratios=[]
        for b in benchmarks:
            ref=next(r for r in synth['resource_benchmarks'] if r['features']==128 and r['length']==b['length'])
            ratios.append((b['forward_seconds']+b['backward_seconds'])/(ref['forward_seconds']+ref['backward_seconds']))
        peak=max(b['peak_rss_bytes'] for b in benchmarks)
        resource_ok=peak<=2*1024**3 and max(ratios)<=4
        med=float(np.median(bycontext));p90=float(np.quantile(bycontext,.9))
        comparison[m]=dict(context_count=len(contexts),mean_mse=mean_mse,
            median_context_mean_seed_relative_l2=med,p90_context_mean_seed_relative_l2=p90,
            mse_reduction_fraction_vs128=1-mean_mse/baseline_mse,
            peak_rss_bytes=peak,worst_cpu_time_ratio_vs128=max(ratios),resource_status='PASS' if resource_ok else 'FAIL',
            accuracy_target_status='PASS' if med<=.10 and p90<=.20 else 'FAIL',
            eligible=numeric_pass and resource_ok and med<=.10 and p90<=.20)
    eligible=[m for m in design['features'] if comparison[m]['eligible']]
    if eligible:chosen=min(eligible);reason='smallest count meeting frozen initial-context accuracy/numerical/resource targets'
    else:
        fallback=[m for m in design['features'] if comparison[m]['resource_status']=='PASS' and comparison[m]['mse_reduction_fraction_vs128']>=.20]
        chosen=max(fallback) if fallback else 128
        reason='fallback for diagnostic pilot only; no tested count met initial-context targets'
    grid=[]
    for scale in design['synthetic']['scales']:
        for m in design['features']:
            rs=[r for r in synth['synthetic'] if r['scale']==scale and r['features']==m]
            grid.append(dict(scale=scale,features=m,cases=len(rs),
                mean_mse=float(np.mean([r['feature_vs_exact_softmax']['mse'] for r in rs])),
                mean_relative_l2=float(np.mean([r['feature_vs_exact_softmax']['relative_l2'] for r in rs])),
                mean_pair_median_log10_iid_cv2=float(np.mean([r['iid_log10_cv_squared']['median'] for r in rs])),
                mean_pair_median_feature_ess=float(np.mean([r['feature_ess']['median'] for r in rs]))))
    gradients=[r for r in synth['synthetic'] if 'gradient_max_abs' in r]
    gradients += [r for c in all_contexts for r in c['results'] if 'gradient_max_abs' in r]
    result=dict(status='PASS' if numeric_pass else 'FAIL',recommendation_features=chosen,recommendation_scope='bounded pilot only',
        recommendation_reason=reason,default_changed=False,full_training_approved=False,test_auc_used=False,
        feature_comparison=comparison,datasets=datasets,synthetic_grid=grid,gaussian_kernel_moment_controls=kernel_moments(),
        synthetic_comparisons=len(synth['synthetic']),sample_contexts=len(all_contexts),
        sample_comparisons=sum(len(c['results']) for c in all_contexts),
        implementation_max_abs_output=max([r['implementation_vs_feature_reference']['max_abs'] for r in synth['synthetic']]+
                                         [r['implementation_vs_feature_reference']['max_abs'] for c in all_contexts for r in c['results']]),
        double_gradient_comparison_cases=len(gradients),
        implementation_max_abs_double_gradient=max(v for r in gradients for v in r['gradient_max_abs'].values()),
        precision_controls=precision['checks'],gpu_status='PENDING: no independent exclusive allocation established',
        source_sha256=sha256(Path(__file__)),design_sha256=sha256(root/'PREDECLARED_DESIGN.json'))
    write_json(root/'quality_decision.json',result)
    print('Recommendation',chosen,'for bounded pilot only;',reason)


if __name__=='__main__':main()
