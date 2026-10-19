#!/usr/bin/env python3
"""Render the editable supervisor Markdown to a bounded two-page PDF and HTML."""
import argparse
import html
import json
from pathlib import Path
import re
import markdown
import pymupdf
from bs4 import BeautifulSoup
from scripts.diagnose_hitskt_common import sha

ROOT='https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/blob/main/reports/'


def build(source,style,pdf_path,html_path,record_path):
    parts=source.read_text().split('<!-- PAGEBREAK -->');assert len(parts)==2
    css=style.read_text();rendered=[]
    for page_number,part in enumerate(parts):
        soup=BeautifulSoup(markdown.markdown(part,extensions=['tables']),'html.parser')
        for a in soup.find_all('a',href=True):
            if not a['href'].startswith(('http://','https://','#')):a['href']=ROOT+a['href']
        tables=soup.find_all('table')
        widths=([['11%','21%','21%','13%','13%','21%'],['22%','22%','18%','20%','18%']]
            if page_number==0 else [['15%','44%','41%'],['24%','19%','19%','19%','19%']])
        assert len(tables)==len(widths)
        for table,column_widths in zip(tables,widths):
            for th,width in zip(table.find_all('th'),column_widths):th['style']='width:'+width
        for cell in soup.find_all(['td','p']):
            if cell.get_text(strip=True) in ['PASS','FAIL','NOT VERIFIED','PASS / PASS']:
                cell['class']={'PASS':'pass','FAIL':'fail','NOT VERIFIED':'pending','PASS / PASS':'pass'}[cell.get_text(strip=True)]
        rendered.append(str(soup))
    doc=pymupdf.open();pages=[];width,height=pymupdf.paper_size('a4')
    for i,body in enumerate(rendered):
        page=doc.new_page(width=width,height=height)
        area=pymupdf.Rect(38,35,width-38,height-44)
        spare,scale=page.insert_htmlbox(area,body,css=css,scale_low=1)
        if spare<0:raise ValueError(f'Page{i+1} exceeds fixed area; edit source/style, do not silently shrink text')
        assert scale==1
        page.draw_line((38,height-32),(width-38,height-32),color=(.70,.77,.81),width=.6)
        page.insert_text((38,height-20),'HiTSKT | Supervisor verification | Evidence e1065c9 | 19 October 2026',fontsize=7,color=(.3,.4,.48))
        page.insert_text((width-85,height-20),f'{i+1} / 2',fontsize=7,color=(.3,.4,.48))
        pages.append(dict(page=i+1,spare_height_points=spare,scale=scale,words=len(page.get_text('words'))))
    doc.set_metadata(dict(title='Modified HiTSKT: supervisor verification brief',author='HiTSKT project research verification',
        subject='Historical CUDA and independent CPU results; verified, failed and unverified evidence',
        keywords='HiTSKT, ELU+1, reproducibility, retrospective evaluation',
        creationDate='D:20261019000000Z',modDate='D:20261019000000Z'))
    assert len(doc)==2
    doc.save(pdf_path,deflate=True,garbage=4);doc.close()
    full='<!doctype html>\n<html><head><meta charset="utf-8"><title>HiTSKT supervisor brief</title><style>'+css+'\n@media print { .page + .page { page-break-before: always; } }\n</style></head><body>'
    full+='\n'.join('<section class="page">'+p+'</section>' for p in rendered)+'</body></html>\n'
    html_path.write_text(full)
    record_path.write_text(json.dumps(dict(status='RENDERED, AWAITING VISUAL REVIEW',pages=pages,
        source_sha256=sha(source),style_sha256=sha(style),pdf_sha256=sha(pdf_path),html_sha256=sha(html_path),
        renderer_sha256=sha(__file__),pymupdf_version=pymupdf.__version__,markdown_version=markdown.__version__,
        page_size='A4',font_size_points=10,scale_lower_bound=1,editable_source=str(source)),indent=2)+'\n')
    print(json.dumps(pages))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('reports/HITSKT_SUPERVISOR_BRIEF.md'))
    parser.add_argument('--pdf',type=Path,default=Path('reports/HITSKT_SUPERVISOR_BRIEF.pdf'))
    parser.add_argument('--html',type=Path,default=Path('reports/HITSKT_SUPERVISOR_BRIEF.html'))
    parser.add_argument('--record',type=Path,default=Path('reports/supervisor_verification/pdf_render.json'))
    args=parser.parse_args()
    for path in [args.pdf,args.html,args.record]:
        if path.exists():raise FileExistsError(path)
    build(args.source,Path('reports/supervisor_verification/report_style.css'),args.pdf,args.html,args.record)
