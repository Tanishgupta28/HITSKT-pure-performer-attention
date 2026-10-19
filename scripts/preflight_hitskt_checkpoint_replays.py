#!/usr/bin/env python3
"""Metadata-only batch/Phi RAM estimates; never initializes CUDA or a model."""
from pathlib import Path
import hashlib
import os
import zipfile
import numpy as np
from ktbench.batching import collate_hitskt
from ktbench.data.session_store import SessionStore,RollingSessionDataset,make_session_dataloader
from scripts.diagnose_hitskt_common import DATASETS,read,write,sha,now

OUT=Path('reports/supervisor_verification')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    order=read('reports/reproducibility_diagnostics/evaluation_order.json')['records'];hitskt={};phi={}
    for dataset in DATASETS:
        cfg=read(Path('experiments/hitskt')/dataset/'full/config.json')
        store=SessionStore(Path('data/processed')/dataset/'full/session_store')
        ds=RollingSessionDataset(store,'test',history_sessions=cfg['history_sessions'])
        _,sampler=make_session_dataloader(ds,token_budget=cfg['token_budget'],maximum_batch_size=cfg['maximum_batch_size'],
            bucket_size=cfg['bucket_size'],shuffle=False,seed=42,workers=0,pin_memory=False);sampler.set_epoch(0)
        max_estimate=0;largest=None;targets=batches=0;digest=hashlib.sha256()
        for bi,indices in enumerate(sampler):
            shapes=[ds.shape(j) for j in indices]
            history_slots=sum(len(s.history_lengths) for s in shapes)*(1+max(max(s.history_lengths) for s in shapes))
            target_slots=len(shapes)*(1+max(s.target_length for s in shapes));count=sum(s.target_length for s in shapes)
            estimate=3*max(history_slots,target_slots)*cfg['heads']*(cfg['width']//cfg['heads'])**2*4+1024**3
            if estimate>max_estimate:
                max_estimate=estimate;largest=dict(batch_index=bi,examples=len(shapes),history_slots=history_slots,target_slots=target_slots)
            if bi<3:
                batch=collate_hitskt([ds[j] for j in indices],num_questions=store.metadata['num_questions'],num_skills=store.metadata['num_skills'])
                assert history_slots==batch.history_question.numel() and target_slots==batch.target_question.numel()
            digest.update(np.asarray([count,*map(int,indices)],dtype='<i8').tobytes());targets+=count;batches+=1
        assert targets==order[dataset]['targets'] and batches==order[dataset]['batches']
        assert digest.hexdigest()==order[dataset]['batch_membership_count_sha256']
        hitskt[dataset]=dict(original_batches=batches,targets=targets,maximum_forward_estimate_bytes=max_estimate,
            maximum_forward_estimate_gib=max_estimate/1024**3,largest_original_batch=largest,
            all_original_batch_order_hash_matches=True,first_three_shape_estimates_match_actual_collator=True,
            under_16gib_estimate_cap=max_estimate<=16*1024**3)
        root=Path('data/processed')/dataset/'full/rkt_phi_full';files=[]
        for name in ['phi_all_train.npz',*[f'phi_excluding_fold_{i}.npz' for i in range(5)]]:
            path=root/name;members=[]
            with zipfile.ZipFile(path) as z:
                for item in z.infolist():
                    with z.open(item) as f:
                        version=np.lib.format.read_magic(f)
                        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                        else:raise ValueError(('Unsupported NPY version',version))
                    assert not dtype.hasobject
                    members.append(dict(name=item.filename,shape=list(shape),dtype=str(dtype),
                        array_payload_bytes=int(np.prod(shape))*dtype.itemsize,uncompressed_member_bytes=item.file_size))
            files.append(dict(path=str(path),compressed_file_bytes=path.stat().st_size,members=members))
        total=sum(m['array_payload_bytes'] for p in files for m in p['members'])
        phi[dataset]=dict(all_six_repository_payload_bytes=total,all_six_repository_payload_gib=total/1024**3,files=files,
            limitation='Header/payload estimate only, not repository allocation, peak RSS or verification of raw relation counts')
        print(dataset,'HiTSKT max estimate GiB',max_estimate/1024**3,'RKT six-matrix payload GiB',total/1024**3,flush=True)
    write(OUT/'replay_resource_preflight.json',dict(utc=now(),status='PASS: metadata-only preflight completed',
        hitskt=hitskt,rkt_phi=phi,implementation_sha256=sha(__file__),cuda_allocations=0,model_forwards=0,
        estimate='3 copies of largest original causal outer-product prefix +1GiB overhead; FP32; not measured peak CUDA/RSS',
        gpu_admission='NOT VERIFIED: resources remain occupied; estimates never establish an independent lease'))


if __name__=='__main__':main()
