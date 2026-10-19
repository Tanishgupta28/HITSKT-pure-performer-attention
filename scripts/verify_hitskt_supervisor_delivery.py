#!/usr/bin/env python3
"""Verify supervisor tables against source facts and independently parse PDF."""
from pathlib import Path
import re
import unicodedata
import markdown
from bs4 import BeautifulSoup
import pymupdf
from pypdf import PdfReader
from scripts.diagnose_hitskt_common import read,write,sha,now,old_evaluation

OUT=Path('reports/supervisor_verification')
DATASETS=['assist2017','junyi','ednet_kt1']
NAMES=['ASSIST2017','Junyi','EdNet-KT1']


def main():
    facts=read(OUT/'supervisor_facts.json');brief=Path('reports/HITSKT_SUPERVISOR_BRIEF.md')
    pdf=Path('reports/HITSKT_SUPERVISOR_BRIEF.pdf');html=Path('reports/HITSKT_SUPERVISOR_BRIEF.html')
    source=brief.read_text();soup=BeautifulSoup(markdown.markdown(source,extensions=['tables']),'html.parser')
    tables=soup.find_all('table');assert len(tables)==4
    for i,d in enumerate(DATASETS):
        v=facts['datasets'][d];cells=[c.get_text(' ',strip=True) for c in tables[0].find_all('tr')[i+1].find_all('td')]
        assert cells[0]==NAMES[i] and float(cells[1])==v['historical_cuda_metrics']['roc_auc']
        assert float(cells[2])==v['independent_cpu_metrics']['roc_auc']
        assert cells[3]=='PASS / PASS' and cells[4]=='NOT VERIFIED'
        assert 'no retraining yet' in cells[5]
        old=old_evaluation(d);assert v['historical_cuda_metrics']==old['stored_metrics']
        assert v['independent_cpu_metrics']==old['recalculated_metrics']
        assert v['absolute_tolerance']==1e-6 and v['historical_to_cpu_metric_gate']=='FAIL'
        assert old['comparison']['roc_auc']['within_tolerance']
        assert not all(r['within_tolerance'] for r in old['comparison'].values())
        cells=[c.get_text(' ',strip=True) for c in tables[1].find_all('tr')[i+1].find_all('td')]
        expected=[v['source_rows'],v['retained_students'],v['retained_interactions'],v['sessions']]
        assert [int(c.replace(',','')) for c in cells[1:]]==expected
        cells=[c.get_text(' ',strip=True) for c in tables[3].find_all('tr')[i+1].find_all('td')]
        for j,model in enumerate(['dkt','dkvmn','sakt','rkt']):
            baseline=next(b for b in facts['baselines'] if b['model']==model and b['dataset']==d)
            assert cells[j+1]==(format(baseline['stored_metrics']['roc_auc'],'.9f') if baseline['complete'] else 'MISSING')
    for path,value in facts['input_sha256'].items():assert sha(path)==value,path
    assert sum(d['test_targets'] for d in facts['datasets'].values())==15998524
    assert len([b for b in facts['baselines'] if b['complete']])==11
    assert {b['history_length'] for b in facts['baselines'] if b['model']=='dkvmn'}=={199}
    assert {b['history_length'] for b in facts['baselines'] if b['model']=='rkt'}=={49}
    render=read(OUT/'pdf_render.json');visual=read(OUT/'pdf_visual_review.json')
    assert render['pdf_sha256']==visual['pdf_sha256']==sha(pdf)
    assert render['source_sha256']==sha(brief) and render['html_sha256']==sha(html)
    assert render['style_sha256']==sha(OUT/'report_style.css')
    assert render['renderer_sha256']==sha('scripts/render_hitskt_supervisor_pdf.py')
    assert visual['status']=='PASS' and all(p['scale']==1 for p in render['pages'])
    reader=PdfReader(pdf);assert len(reader.pages)==2
    text=' '.join(unicodedata.normalize('NFKC','\n'.join(p.extract_text() for p in reader.pages)).split())
    for table in [tables[0],tables[1],tables[3]]:
        for c in table.find_all('td'):
            value=c.get_text(' ',strip=True)
            if re.fullmatch(r'[0-9.,]+',value):assert value in text,value
    for required in ['NOT VERIFIED','FAIL','ELU+1','FAVOR+','teacher-forced','not student-disjoint','21,749','59','0.006571357','0.006356979']:
        assert required in text,required
    doc=pymupdf.open(pdf)
    for p in doc:
        assert p.rect.width>590 and p.rect.height>840
        for x0,y0,x1,y1,*_ in p.get_text('words'):
            assert x0>=32 and x1<=p.rect.width-30 and y0>=20 and y1<=p.rect.height-12,(x0,y0,x1,y1)
    assert len(doc[1].get_links())>=4
    documents=[brief,Path('reports/HITSKT_VERIFICATION_STATUS.md'),Path('reports/HITSKT_CHECKPOINT_REPLAY_PLAN.md')]
    for p in documents:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https://','http://')):continue
            local=p.parent/target.split('#')[0]
            assert local.exists() or local.resolve()==(OUT/'delivery_verification.json').resolve(),(p,target)
    initial=read(OUT/'preservation_initial.json');final=read(OUT/'preservation_final.json')
    assert initial['status']==final['status']=='PASS'
    assert final['original_count']==375 and final['prior_verification_count']==156
    assert initial['unrelated_files']==final['unrelated_files']
    assert '59 passed' in (OUT/'causal_control_recheck.log').read_text()
    resource=read(OUT/'resource_observation.json');assert resource['cuda_allocations']==0
    preflight=read(OUT/'replay_resource_preflight.json')
    assert preflight['cuda_allocations']==preflight['model_forwards']==0
    assert preflight['hitskt']['ednet_kt1']['maximum_forward_estimate_gib']>16
    assert all(p['all_original_batch_order_hash_matches'] for p in preflight['hitskt'].values())
    bindings={}
    for name,script in [('supervisor_facts.json','prepare_hitskt_supervisor_facts.py'),
        ('baseline_context_correction.json','prepare_hitskt_supervisor_facts.py'),
        ('replay_resource_preflight.json','preflight_hitskt_checkpoint_replays.py'),
        ('preservation_initial.json','audit_supervisor_evidence.py'),('preservation_final.json','audit_supervisor_evidence.py')]:
        assert read(OUT/name)['implementation_sha256']==sha(Path('scripts')/script)
        bindings[str(OUT/name)]=str(Path('scripts')/script)
    outputs=documents+[pdf,html,OUT/'report_style.css']
    # The only permitted pending link is the manifest being created here.
    write(OUT/'delivery_verification.json',dict(utc=now(),status='PASS',report_tables='PASS',
        pdf_independent_parser='PASS',pdf_layout_geometry='PASS',pdf_visual_review='PASS',pages=2,
        source_link_checks='PASS',relevant_controls_passed=59,cached_cpu_targets_recalculated=15998524,
        original_research_files_preserved=375,prior_verification_files_preserved=156,
        full_cuda_replays=0,full_baseline_replays=0,full_training_runs=0,
        baseline_context_erratum='VERIFIED: DKVMN199, RKT49',
        resource_status='NOT VERIFIED: no idle independent GPU; preflight estimates only',
        output_sha256={str(p):sha(p) for p in outputs},implementation_bindings=bindings,implementation_sha256=sha(__file__),
        publication_readiness='NOT VERIFIED',next_step='Independent-resource CUDA replay under unchanged1e-6 gate; no retraining yet'))
    print('PASS: evidence-derived tables, independent PDF parse, two-page geometry/visual review, links,59 controls,375+156 preservation')


if __name__=='__main__':main()
