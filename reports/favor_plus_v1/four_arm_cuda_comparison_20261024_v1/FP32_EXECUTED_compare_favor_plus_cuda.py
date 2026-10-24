#!/usr/bin/env python3
"""Four-arm shared-CUDA verification/profile only. This CLI cannot optimize."""
import argparse,copy,datetime as dt,gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from ktbench.favor_plus_matched_comparison import (ARMS,VERIFY_ACTION,LIMITS,MatchedArmConfig,
    MatchedComparisonModel,SharedStudyGuard,fresh_model,learned_digest,initialize_cuda,arm_projection)
from ktbench.favor_plus_pipeline import ROOT,fresh_run,sha256,write_json,runtime
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree
from ktbench.favor_plus_bounded_assist import select_samples,collate_selected,descriptor
from ktbench.data.session_store import SessionStore
from ktbench.batching import Session,SessionExample,collate_hitskt
from ktbench.metrics import masked_bce_loss
from ktbench.training import capture_rng_state,restore_rng_state
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.favor_plus import causal_favor_attention
from ktbench.favor_plus_post50_diagnostics import attention_diagnostics
from scripts.favor_plus_quality_common import torch_feature_reference

HERE=ROOT/'reports/favor_plus_v1/four_arm_cuda_comparison_20261024_v1'
DESIGN=HERE/'PREDECLARED_DESIGN.json'
PARENT=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1'
FROZEN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/postepoch8_frozen_20261024_v1'
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/four_arm_cuda_readiness_20261024_v1'
SOURCES=('ktbench/favor_plus_matched_comparison.py','scripts/compare_favor_plus_cuda.py','ktbench/models/favor_plus_antithetic_experimental.py','ktbench/favor_plus_post50_diagnostics.py')


def setup_cpu():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)


def synthetic_batch(config):
    examples=[SessionExample([Session([1,2],[1,2],[0,1])],Session([3,4,5,6],[1]*4,[1,0,1,0]),0),
              SessionExample([Session([1],[1],[0]),Session([2,3,4],[1]*3,[1,0,1])],Session([2,3],[1,1],[1,0]),0)]
    return examples,collate_hitskt(examples,num_questions=config.num_questions,num_skills=config.num_skills)


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU plan requires CUDA hidden')
    setup_cpu();parent=json.loads((PARENT/'prepared_plan.json').read_text());c=parent['contract']
    for name,expected in c['source_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('historical source changed:'+name)
    for name,expected in c['dataset_identity']['files'].items():
        if sha256(Path(parent['store_path'])/name)!=expected:raise ValueError('processed dataset changed')
    train,_,samples=select_samples(SessionStore(parent['store_path']))
    training=samples['training'][:20];ordered=sorted(training,key=lambda x:x['padded_tokens'])
    profile=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    largest=max((descriptor(train,i) for i in range(len(train)) if
        sum(train.shape(i).history_lengths)+train.shape(i).target_length>0),key=lambda x:x['padded_tokens'] if x['padded_tokens']<=2048 else -1)
    if largest['target_session'] not in {x['target_session'] for x in profile}:profile.append(largest)
    config=HiTSKTConfig(**c['model_config']);identities=[]
    for arm in ARMS:
        model=fresh_model(config,arm);identities.append(dict(arm=arm,learned_sha256=learned_digest(model),cpu_rng_after_factory_sha256=digest(torch.random.get_rng_state()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),projection_sha256={n:digest(v) for n,v in model.named_buffers() if n.endswith('.projection')}));del model
    if len({x['learned_sha256'] for x in identities})!=1 or len({x['cpu_rng_after_factory_sha256'] for x in identities})!=1:raise RuntimeError('unmatched initial learned parameters/RNG')
    full_manifest=json.loads((ROOT/'reports/favor_plus_v1/postepoch8_approximation_20261024_v1/RAW_ARRAY_MANIFEST.json').read_text())
    initial=[x for x in full_manifest['files'] if Path(x['path']).name.startswith('initial_')]
    for row in initial:
        if sha256(ROOT/row['path'])!=row['sha256']:raise ValueError('frozen INITIAL QKV source changed')
    plan=dict(kind='four_arm_fresh_cuda_readiness_v1',model_config=c['model_config'],dataset_identity=c['dataset_identity'],store_path=parent['store_path'],design_sha256=sha256(DESIGN),sources={name:sha256(ROOT/name) for name in SOURCES},historical_sources=c['source_sha256'],fresh_identities=identities,
        training=training,training_order_sha256=digest(training),validation=samples['validation'],validation_coverage=samples['validation_coverage'],diagnostics=samples['diagnostics'],profile=profile,initial_arrays=initial,
        no_trained_weights_loaded=True,optimizer_updates_authorized=0,training_approval='PENDING EXPLICIT USER RESPONSE',review_threshold=.20,source_initial_qkv_scope='fresh original128 initialization, one shared input per stage; identical across arms, not trained7/8 representations',skipped_oversize_for_pilot_only=samples['skipped_oversize_training_singletons'])
    fresh_run('assist2017',RUN.name);write_json(RUN/'prepared_plan.json',plan)
    print(json.dumps(dict(status='PREPARED_ZERO_UPDATES',initial_parameters_equal=True,training_samples=len(training),validation=samples['validation_coverage'],profile_costs=[x['padded_tokens'] for x in profile]),indent=2))


def load_plan():
    path=RUN/'prepared_plan.json';p=json.loads(path.read_text())
    if p['design_sha256']!=sha256(DESIGN):raise ValueError('design drift')
    for name,expected in {**p['historical_sources'],**p['sources']}.items():
        if sha256(ROOT/name)!=expected:raise ValueError('source drift:'+name)
    return p


def record():
    p=load_plan();now=dt.datetime.now(dt.timezone.utc)
    value=dict(action=VERIFY_ACTION,resource_uuid='MIG-d2fcfd79-720e-5218-82c9-0a132be68920',prepared_plan_sha256=sha256(RUN/'prepared_plan.json'),shared_permission_confirmed=True,
        known_explicit_restriction=False,known_incompatible_reservation=False,limits=LIMITS,starts_utc=now.isoformat(),expires_utc=(now+dt.timedelta(hours=2)).isoformat(),campaign_started_unix=now.timestamp(),cuda_verification_approved=True,max_optimizer_updates=0,
        training_approved=False,actual_training_approval_evidence='',scope_evidence=str(HERE/'USER_EXECUTION_SCOPE.md'))
    path=HERE/'CUDA_VERIFICATION_RECORD.json'
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def failure(output,phase,exc,guard):
    value=dict(status='FAIL_STOP_NO_RETRY',phase=phase,type=type(exc).__name__,message=str(exc),last_cached_resources=guard.last,optimizer_updates=0,no_cuda_calls_in_failure_handler=True)
    write_json(output/'failure.json',value)
    if not (RUN/'study_failure.json').exists():write_json(RUN/'study_failure.json',value)


def worker(arm):
    plan=load_plan();record=json.loads((HERE/'CUDA_VERIFICATION_RECORD.json').read_text());out=RUN/arm
    if (RUN/'study_failure.json').exists():raise RuntimeError('failed study cannot retry')
    out.mkdir(exist_ok=False);guard=SharedStudyGuard(record,RUN/'prepared_plan.json',out);phase='admission';rows=[];started=time.time()
    try:
        initialize_cuda(guard);cfg=MatchedArmConfig(arm);config=HiTSKTConfig(**plan['model_config'])
        for length in (17,65,257,939):
            for chunk in (16,32):
                for norm in (.35,1.,3.):
                    phase=f'kernel_{length}_{chunk}_{norm}';guard.check(phase)
                    gen=torch.Generator().manual_seed(42+length);cpu=[(torch.randn(1,2,length,64,generator=gen)*norm).requires_grad_() for _ in range(3)]
                    w=arm_projection(2,64,cfg,0);mask=torch.ones(1,length,dtype=torch.bool);mask[:,1::7]=False
                    expected=causal_favor_attention(*cpu,w,query_mask=mask,key_mask=mask,chunk_size=chunk);cg=torch.autograd.grad(expected.square().mean(),cpu)
                    gpu=[x.detach().cuda().requires_grad_() for x in cpu];wm=w.cuda();gm=mask.cuda();torch.cuda.synchronize();stamp=time.perf_counter()
                    actual=causal_favor_attention(*gpu,wm,query_mask=gm,key_mask=gm,chunk_size=chunk);grads=torch.autograd.grad(actual.square().mean(),gpu);torch.cuda.synchronize();seconds=time.perf_counter()-stamp
                    delta=float((actual.cpu()-expected).abs().max());gd=[float((a.cpu()-b).abs().max()) for a,b in zip(grads,cg)]
                    if delta>5e-5 or not all(torch.allclose(a.cpu(),b,atol=5e-5,rtol=1e-3) for a,b in zip(grads,cg)):raise AssertionError('CPU/CUDA output or gradient rule failed')
                    if not torch.isfinite(actual).all() or not all(torch.isfinite(x).all() for x in grads):raise FloatingPointError('nonfinite kernel')
                    oracle_gradient=None
                    if length<=65:
                        ex=[x.detach().double().requires_grad_() for x in cpu];ref=torch_feature_reference(*ex,w.double(),mask,mask);rg=torch.autograd.grad(ref.square().mean(),ex)
                        if (actual.cpu().double()-ref).abs().max()>5e-5 or not all(torch.allclose(a.cpu().double(),b,atol=5e-5,rtol=1e-3) for a,b in zip(grads,rg)):raise AssertionError('independent feature reference rule failed')
                        oracle_gradient=max(float((a.cpu().double()-b).abs().max()) for a,b in zip(grads,rg))
                        probe=causal_favor_attention(*gpu,wm,query_mask=gm,key_mask=gm,chunk_size=chunk);future=torch.autograd.grad(probe[:,:,:length//2].sum(),gpu)
                        if not all(x[:,:,length//2:].eq(0).all() for x in future):raise AssertionError('future gradient leak')
                    changed=[x.detach().clone() for x in gpu]
                    for x in changed:x[:,:,length//2:]+=100
                    with torch.no_grad():other=causal_favor_attention(*changed,wm,query_mask=gm,key_mask=gm,chunk_size=chunk)
                    if not torch.equal(actual[:,:,:length//2],other[:,:,:length//2]):raise AssertionError('future output leak')
                    if not actual.masked_select(~gm[:,None,:,None].expand_as(actual)).eq(0).all():raise AssertionError('padding output nonzero')
                    if seconds>30:raise TimeoutError('30s kernel step ceiling')
                    rows.append(dict(phase=phase,status='PASS',output_max_abs=delta,gradient_max_abs=gd,independent_gradient_max_abs=oracle_gradient,seconds=seconds,resources=guard.check('after_'+phase)))
                    del cpu,gpu,actual,expected,cg,grads,changed,other
        write_json(out/'kernel_results.json',dict(status='PASS',checks=rows,optimizer_updates=0))
        phase='full_model';guard.check(phase);cpu=fresh_model(config,arm).eval();before=learned_digest(cpu)
        identity=next(x for x in plan['fresh_identities'] if x['arm']==arm)
        if before!=identity['learned_sha256'] or digest(torch.random.get_rng_state())!=identity['cpu_rng_after_factory_sha256']:raise AssertionError('fresh parameter/RNG identity differs from plan')
        net=copy.deepcopy(cpu).cuda().eval();examples,b=synthetic_batch(config);gb=b.to('cuda:0')
        clogits=cpu(b);masked_bce_loss(clogits,b.target_labels,b.target_metric_mask).backward();logits=net(gb);masked_bce_loss(logits,gb.target_labels,gb.target_metric_mask).backward()
        logit_delta=float((logits.cpu()-clogits).abs().max());param_grad_delta=max(float((a.grad.cpu()-c.grad).abs().max()) for a,c in zip(net.parameters(),cpu.parameters()))
        if logit_delta>5e-5 or not all(torch.allclose(a.grad.cpu(),c.grad,atol=5e-5,rtol=1e-3) for a,c in zip(net.parameters(),cpu.parameters())):raise AssertionError('full-model CPU/CUDA rule failed')
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()):raise FloatingPointError('nonfinite full-model gradient')
        for pos in range(4):
            changed=copy.deepcopy(examples);changed[0].target.correct[pos]=1-changed[0].target.correct[pos]
            with torch.no_grad():z=net(collate_hitskt(changed,num_questions=config.num_questions,num_skills=config.num_skills).to('cuda:0'))
            if not torch.equal(logits[0,:pos+1],z[0,:pos+1]) or not torch.equal(logits[1],z[1]):raise AssertionError('full-model own/future label leak')
            retained=[];h=net.core.correct_padding_encoder.correct.register_forward_hook(lambda m,a,o:retained.append(o))
            try:
                y=net(gb);response_grad=torch.autograd.grad(y[0,pos],retained[0])[0]
                if not response_grad[0,pos+1:].eq(0).all() or not response_grad[1].eq(0).all():raise AssertionError('correctness shift gradient leak')
            finally:h.remove()
        padded=copy.deepcopy(gb);padded.target_correct_input[~padded.target_attention_mask]=1;padded.history_correct[~padded.history_mask]=1
        padded.target_question[~padded.target_attention_mask]=1;padded.target_skill[~padded.target_attention_mask]=1
        padded.history_question[~padded.history_mask]=1;padded.history_skill[~padded.history_mask]=1;padded.target_labels.fill_(99)
        with torch.no_grad():z=net(padded)
        if not torch.equal(z[gb.target_metric_mask],logits[gb.target_metric_mask]) or int(b.target_metric_mask.sum())!=6 or not b.target_correct_input[:,0].eq(4).all():raise AssertionError('BOS/EOS/padding failed')
        write_json(out/'full_model_results.json',dict(status='PASS',cpu_cuda_logit_max_abs=logit_delta,parameter_gradient_max_abs=param_grad_delta,own_future_labels_gradients_padding='PASS',resources=guard.check('after_full_model'),optimizer_updates=0));del cpu;net.zero_grad(set_to_none=True)
        phase='real_data_zero_update_profile';store=SessionStore(plan['store_path']);from ktbench.data.session_store import RollingSessionDataset
        train=RollingSessionDataset(store,'train',history_sessions=15);profile=[];reserve=[torch.zeros_like(p) for p in net.parameters() for _ in range(2)];net.train()
        for item in plan['profile']:
            guard.check('before_real_profile');rb=collate_selected(train,item).to('cuda:0');net.zero_grad(set_to_none=True);torch.cuda.manual_seed(42);torch.cuda.synchronize();stamp=time.perf_counter()
            loss=masked_bce_loss(net(rb),rb.target_labels,rb.target_metric_mask);loss.backward();torch.cuda.synchronize();seconds=time.perf_counter()-stamp
            if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()):raise FloatingPointError('profile nonfinite')
            norm=sum(float(p.grad.double().square().sum()) for p in net.parameters())**.5
            if float(loss)>20 or norm>100 or seconds>30:raise RuntimeError('predeclared profile loss/gradient/time severe gate')
            profile.append(dict(sample=item,loss=float(loss),gradient_l2=norm,forward_backward_seconds=seconds,resources=guard.check('after_real_profile'),optimizer_updates=0));del rb,loss;time.sleep(.1)
        if learned_digest(net)!=before:raise RuntimeError('zero-update verification changed learned parameters')
        del reserve;net.zero_grad(set_to_none=True);net.eval()
        with torch.no_grad():proof=net(gb).cpu()
        payload=cpu_tree(net.checkpoint());payload.update(reference_logits=proof,reference_batch_kind='synthetic2xvariable',cpu_rng_state=torch.random.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),learned_initial_sha256=before)
        tmp=out/'initial_model.partial.pt';target=out/'initial_model.pt'
        with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        if not equal_tree(payload,torch.load(tmp,map_location='cpu',weights_only=True)):raise RuntimeError('serialization roundtrip mismatch')
        os.link(tmp,target);tmp.unlink()
        result=dict(status='PASS_CUDA_ZERO_UPDATES',arm=arm,kernel_checks=len(rows),profile=profile,learned_initial_sha256=before,learned_parameters_unchanged=True,checkpoint_sha256=sha256(target),runtime=runtime(torch.device('cuda:0')),seconds=time.time()-started,resources=guard.check('arm_complete'),optimizer_updates=0,adam_step_measured=False)
        write_json(out/'cuda_result.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:failure(out,phase,exc,guard);raise


def verify_reload(arm):
    plan=load_plan();record=json.loads((HERE/'CUDA_VERIFICATION_RECORD.json').read_text());out=RUN/f'{arm}_recovery'
    if (RUN/'study_failure.json').exists():raise RuntimeError('failed study cannot continue')
    out.mkdir(exist_ok=False);g=SharedStudyGuard(record,RUN/'prepared_plan.json',out)
    try:
        initialize_cuda(g);path=RUN/arm/'initial_model.pt';payload=torch.load(path,map_location='cpu',weights_only=True);net=MatchedComparisonModel.from_checkpoint(path).cuda().eval();_,b=synthetic_batch(net.config)
        with torch.no_grad():logits=net(b.to('cuda:0')).cpu()
        if not torch.equal(logits,payload['reference_logits']) or learned_digest(net)!=payload['learned_initial_sha256']:raise AssertionError('fresh checkpoint output/state not exact')
        torch.random.set_rng_state(payload['cpu_rng_state']);torch.cuda.set_rng_state(payload['cuda_rng_state'])
        if not torch.equal(torch.random.get_rng_state(),payload['cpu_rng_state']) or not torch.equal(torch.cuda.get_rng_state(),payload['cuda_rng_state']):raise AssertionError('RNG restoration mismatch')
        result=dict(status='PASS_EXACT_FRESH_PROCESS',arm=arm,checkpoint_sha256=sha256(path),max_abs_output=0.,rng_restored_exact=True,optimizer_updates=0,resources=g.check('recovery_complete'));write_json(out/'recovery_result.json',result)
    except BaseException as exc:failure(out,'fresh_recovery',exc,g);raise


def real_attention_comparison(arm):
    plan=load_plan();record=json.loads((HERE/'CUDA_VERIFICATION_RECORD.json').read_text());out=RUN/f'{arm}_attention'
    if (RUN/'study_failure.json').exists():raise RuntimeError('failed study cannot continue')
    out.mkdir(exist_ok=False);g=SharedStudyGuard(record,RUN/'prepared_plan.json',out);rows=[];cfg=MatchedArmConfig(arm)
    try:
        initialize_cuda(g)
        for index,item in enumerate(plan['initial_arrays']):
            g.check('before_frozen_input')
            path=ROOT/item['path']
            if sha256(path)!=item['sha256']:raise ValueError('initial input identity mismatch')
            with np.load(path) as z:a={k:z[k] for k in z.files}
            stage=next(i for i,name in enumerate(('action_encoder','session_encoder','correct_padding_encoder','decoder')) if name in path.name)
            w=arm_projection(4,64,cfg,stage);n=a['q'].shape[1]
            for length in sorted({x for x in (8,32,128,256,512) if x<=n}|{n}):
                for scale in (.5,1.,1.5):
                    q,k,v=[torch.from_numpy(a[x][:,:length])[None] for x in ('q','k','v')];q=q*scale;k=k*scale;qm=torch.from_numpy(a['query_mask'][:length])[None];km=torch.from_numpy(a['key_mask'][:length])[None]
                    with torch.no_grad():actual=causal_favor_attention(q.cuda(),k.cuda(),v.cuda(),w.cuda(),query_mask=qm.cuda(),key_mask=km.cuda(),chunk_size=32).cpu().numpy()[0]
                    stats=attention_diagnostics(q.numpy()[0],k.numpy()[0],v.numpy()[0],w.numpy(),qm.numpy()[0],km.numpy()[0],actual)
                    if stats['feature_kernel_agreement']!='PASS':raise AssertionError('real-input independent kernel disagreement')
                    rows.append(dict(input_path=item['path'],stage=stage,length=length,norm_scale=scale,metrics=stats))
            write_json(out/f'input_{index:02d}.json',dict(status='PASS',records=rows[-len(sorted({x for x in (8,32,128,256,512) if x<=n}|{n}))*3:],resources=g.check('after_frozen_input')))
        write_json(out/'comparison_result.json',dict(status='PASS_SAME_INITIAL_TRAIN_QKV',arm=arm,records=rows,review_threshold=.20,optimizer_updates=0,trained_representations_used=False,resources=g.check('comparison_complete')))
    except BaseException as exc:failure(out,'real_attention',exc,g);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','record','verify','reload','compare'));p.add_argument('--arm',choices=ARMS);p.add_argument('--execute-cuda',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='record':record()
    else:
        if not a.execute_cuda or not a.arm:p.error('explicit CUDA command and arm required')
        {'verify':worker,'reload':verify_reload,'compare':real_attention_comparison}[a.action](a.arm)


if __name__=='__main__':main()
