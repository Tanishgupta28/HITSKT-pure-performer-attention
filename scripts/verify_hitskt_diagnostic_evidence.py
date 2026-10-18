#!/usr/bin/env python3
"""Check report rendering, evidence hashes, unchanged gates and scoped counts."""
import os
from pathlib import Path
import re
from scripts.diagnose_hitskt_common import OUT,DATASETS,read,sha,write,now,old_evaluation


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert read(OUT/'preservation_initial.json')['status']==read(OUT/'preservation_final.json')['status']=='PASS'
    assert read(OUT/'preservation_final.json')['original_files']==375
    assert read(OUT/'preservation_final.json')['previous_audit_files']==106
    source_hashes={sha(p):str(p) for p in Path('scripts').glob('diagnose_hitskt_*.py')}
    source_hashes[sha('scripts/generate_hitskt_diagnostic_reports.py')]='scripts/generate_hitskt_diagnostic_reports.py'
    bindings=[]
    for p in sorted(OUT.glob('*.json')):
        x=read(p)
        if 'implementation_sha256' in x:
            assert x['implementation_sha256'] in source_hashes,p
            bindings.append(dict(evidence=str(p),implementation=source_hashes[x['implementation_sha256']]))
    cache_files=0
    for d in DATASETS:
        old=old_evaluation(d)
        for name,value in old['prediction_sha256'].items():
            assert sha(Path(old['prediction_artifacts_local'])/name)==value;cache_files+=1
        x=read(OUT/f'{d}_fp64_{"full" if d=="assist2017" else "sample"}.json')
        assert x['status']=='COMPLETE' and not x['skipped_batches']
        for name,value in x['prediction_sha256'].items():
            assert sha(Path(x['prediction_artifacts_local'])/name)==value;cache_files+=1
        m=read(OUT/f'{d}_diagnostic_manifest.json')
        assert m['cpu_verification'].startswith('FAIL') and m['cuda_verification'].startswith('NOT VERIFIED')
        assert m['stored_metrics']==old['stored_metrics'] and m['cpu_metrics']==old['recalculated_metrics']
        assert x['unchanged_absolute_metric_tolerance']==1e-6
        for p,value in m['input_sha256'].items():assert sha(p)==value
        assert m['comparison']['roc_auc']['within_tolerance']
        assert not all(v['within_tolerance'] for v in m['comparison'].values())
    # Re-render without writing: final prose/tables must match the generator.
    import scripts.generate_hitskt_diagnostic_reports as generator
    rendered=[]
    def check(name,body):
        assert (Path('reports')/name).read_text()==body.rstrip()+'\n',name
        rendered.append(name)
    generator.save=check;generator.write=lambda *args,**kwargs:None;generator.main()
    documents=[Path('reports')/p for p in rendered]+[
        Path('reports/HITSKT_AVAILABILITY_PROTOCOL_V2_PROPOSAL.md'),Path('reports/HITSKT_FAVOR_PLUS_IMPLEMENTATION_PROPOSAL.md')]
    for p in documents:
        content=p.read_text();assert not any(ord(c)<32 and c!='\n' for c in content),p
        for target in re.findall(r'\]\(([^)]+)\)',content):
            if target.startswith(('http://','https://')):continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
    testlog=(OUT/'verification_tests.log').read_text();assert '59 passed' in testlog and 'FAILED' not in testlog
    resource=read(OUT/'resource_final.json')
    assert resource['cuda_allocations_by_this_diagnosis']==0 and resource['oom_control']['oom_kill']=='2'
    assert resource['cgroup_limit_bytes']==90*1024**3
    write(OUT/'verification_summary.json',dict(utc=now(),status='PASS',
        implementation_sha256=sha(__file__),implementation_bindings=bindings,verified_prediction_arrays=cache_files,
        preserved_originals=375,preserved_prior_audit_files=106,relevant_tests_passed=59,
        report_rendering_and_local_links='PASS',reports_sha256={str(p):sha(p) for p in documents},
        limitations=['No CUDA allocation/replay','No historical CUDA vectors','Junyi/EdNet FP64 inference sampled only',
            'Actual correctness-release timing NOT VERIFIED','Baselines not freshly replayed']))
    print('PASS: report rendering, local links, source/input/prediction hashes, frozen gates, preservation and59 tests')


if __name__=='__main__':main()
