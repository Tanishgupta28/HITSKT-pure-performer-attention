"""Fresh matched20 pilot, explicitly unreachable without new training approval."""
from __future__ import annotations
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
from ktbench.favor_plus_matched_comparison import (ARMS,TRAIN_ACTION,VERIFY_ACTION,LIMITS,
    MatchedComparisonModel,MatchedArmConfig,arm_projection,SharedStudyGuard,fresh_model,learned_digest,initialize_cuda,require_scope)
from ktbench.favor_plus_pipeline import ROOT,sha256,write_json
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree,fixed_example_diagnostics
from ktbench.favor_plus_assist_epochs import fixed_logits,probe
from ktbench.favor_plus_bounded_assist import collate_selected,severe_check
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.metrics import masked_bce_loss,binary_metrics
from ktbench.training import capture_rng_state,restore_rng_state
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.favor_plus_post50_diagnostics import attention_diagnostics

READY=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/four_arm_cuda_readiness_fp64_20261024_v2'
PILOT=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/matched_four_arm20_fp64_20261024_v1'
HERE=ROOT/'reports/favor_plus_v1/four_arm_cuda_comparison_20261024_v1'
SOURCES=('ktbench/favor_plus_matched_pilot.py','scripts/pilot_favor_plus_matched.py')


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU preparation only')
    ready=json.loads((READY/'prepared_plan.json').read_text());receipts={}
    for arm in ARMS:
        for name in (f'{arm}/cuda_result.json',f'{arm}_recovery/recovery_result.json',f'{arm}_attention/comparison_result.json'):
            p=READY/name;value=json.loads(p.read_text())
            if not value['status'].startswith('PASS') or value['optimizer_updates']!=0:raise ValueError('all four CUDA/profile/recovery/comparison prerequisites required')
            receipts[name]=sha256(p)
    parent=json.loads((ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1/prepared_plan.json').read_text())
    plan=dict(kind='fresh_matched_four_arm20_fp64_proposal_v1',arms=list(ARMS),maximum_updates_per_arm=20,total_optimizer_updates_maximum=80,
        training_approved=False,attention_accumulation_dtype='float64',seed=42,max_campaign_seconds=900,batch_size=1,padded_cost_maximum=2048,
        optimizer=dict(lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=0),milestones=[0,5,20],resumable_checkpoint_every=5,audit_projection_seeds=[17,101],
        readiness_plan_sha256=sha256(READY/'prepared_plan.json'),readiness_receipts=receipts,model_config=ready['model_config'],dataset_identity=ready['dataset_identity'],store_path=ready['store_path'],
        training=ready['training'],training_order_sha256=ready['training_order_sha256'],validation=ready['validation'],validation_coverage=ready['validation_coverage'],diagnostics=ready['diagnostics'],fresh_identities=ready['fresh_identities'],
        sources={**ready['sources'],**{name:sha256(ROOT/name) for name in SOURCES}},historical_sources=ready['historical_sources'],severe_stops=parent['contract']['review_and_severe_stops']['severe_stops'],review_threshold=.20,
        notice='PROPOSAL ONLY. No historical trained weights or readiness dropout RNG warm-start. All arms fresh42. Initial/5/20 sampled validation, no test; no full epoch training.')
    PILOT.mkdir(exist_ok=False);write_json(PILOT/'prepared_plan.json',plan)
    template=dict(action=TRAIN_ACTION,resource_uuid='MIG-d2fcfd79-720e-5218-82c9-0a132be68920',prepared_plan_sha256=sha256(PILOT/'prepared_plan.json'),shared_permission_confirmed=True,known_explicit_restriction=False,known_incompatible_reservation=False,
        limits=LIMITS,training_approved=False,approval_is_proposal=True,actual_training_approval_evidence='',maximum_updates_per_arm=20,arms=list(ARMS),starts_utc='SET_ONLY_AFTER_EXPLICIT_APPROVAL',expires_utc='SET_ONLY_AFTER_EXPLICIT_APPROVAL',campaign_started_unix=None)
    write_json(HERE/'MATCHED_TRAINING_RECORD_TEMPLATE.json',template)
    print(json.dumps(dict(status='PROPOSAL_PREPARED_NOT_AUTHORIZED',plan_sha256=sha256(PILOT/'prepared_plan.json'),maximum_steps_per_arm=20,arms=list(ARMS),total_steps_maximum=80,training_approved=False)))


def load_plan():
    p=json.loads((PILOT/'prepared_plan.json').read_text())
    for name,expected in {**p['historical_sources'],**p['sources']}.items():
        if sha256(ROOT/name)!=expected:raise ValueError('pilot source identity changed:'+name)
    if p['readiness_plan_sha256']!=sha256(READY/'prepared_plan.json'):raise ValueError('readiness plan changed')
    for name,expected in p['readiness_receipts'].items():
        if sha256(READY/name)!=expected:raise ValueError('CUDA prerequisite changed')
    if p['training_order_sha256']!=digest(p['training']) or len(p['training'])!=20:raise ValueError('exact20 sample order required')
    for name,expected in p['dataset_identity']['files'].items():
        if sha256(Path(p['store_path'])/name)!=expected:raise ValueError('processed data changed')
    return p


def budget_check(g,phase):
    if time.time()-g.record['campaign_started_unix']>900:raise TimeoutError('proposed15min total matched-pilot ceiling')
    return g.check(phase)


def checkpoint(net,opt,p,step,proof,path):
    payload=cpu_tree(net.checkpoint());payload.update(kind='four_arm20_training_checkpoint_v1',contract=p,progress=dict(update=step,next_training_index=step,complete=step==20),optimizer_state_dict=cpu_tree(opt.state_dict()),reference_logits=fixed_logits(net,proof),reference_probe=probe(net,opt,proof),rng_state=cpu_tree(capture_rng_state()))
    keys=('state_dict','optimizer_state_dict','rng_state','progress','contract','reference_logits','reference_probe');payload['component_sha256']={k:digest(payload[k]) for k in keys}
    if not finite_tree(payload):raise FloatingPointError('checkpoint nonfinite')
    state=payload['optimizer_state_dict']['state']
    if len(state)!=(sum(1 for _ in net.parameters()) if step else 0):raise ValueError('missing/unexpected Adam state')
    if step and any(float(s['step'])!=step or not (s['exp_avg_sq']>=0).all() for s in state.values()):raise ValueError('Adam counter/moment mismatch')
    projections={n:digest(v) for n,v in net.named_buffers() if n.endswith('.projection')}
    if projections!=next(x for x in p['fresh_identities'] if x['arm']==net.comparison.arm)['projection_sha256']:raise ValueError('fixed projection changed')
    tmp=path.with_suffix('.partial.pt')
    with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    restored=torch.load(tmp,map_location='cpu',weights_only=True)
    if not equal_tree(restored,payload):raise RuntimeError('checkpoint serialization mismatch')
    os.link(tmp,path);tmp.unlink();return dict(path=str(path),sha256=sha256(path),update=step)


def evaluate(net,ds,samples,g):
    was=net.training;net.eval();prob=[];labels=[];loss_sum=0.;count=0
    try:
        for item in samples:
            budget_check(g,'sampled_validation');b=collate_selected(ds,item).to('cuda:0')
            with torch.no_grad():out=net(b);loss=masked_bce_loss(out,b.target_labels,b.target_metric_mask)
            if not finite_tree(out) or not torch.isfinite(loss):raise FloatingPointError('validation nonfinite')
            mask=b.target_metric_mask;n=int(mask.sum());count+=n;loss_sum+=float(loss)*n;prob.append(out[mask].sigmoid().cpu());labels.append(b.target_labels[mask].cpu())
        prob=torch.cat(prob);labels=torch.cat(labels);metrics=binary_metrics(prob,labels,loss=loss_sum/count);metrics['bce']=metrics['loss']
        return metrics,prob,labels
    finally:net.train(was)


def diagnose(net,train,p,g,out,step,baseline):
    rows=[];audits=[];stages=('action_encoder','session_encoder','correct_padding_encoder','decoder')
    for index,item in enumerate(p['diagnostics']):
        budget_check(g,'fixed_training_diagnostic');b=collate_selected(train,item).to('cuda:0')
        def persist(name,arrays):
            with (out/f's{step:03d}_case{index}_{name.replace(".","_")}.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
            stage=next(i for i,s in enumerate(stages) if name.startswith(s+'.'))
            for seed in p['audit_projection_seeds']:
                w=arm_projection(arrays['q'].shape[1],arrays['q'].shape[-1],MatchedArmConfig(net.comparison.arm,projection_seed=seed,accumulation_dtype='float64'),stage).numpy()
                metrics=attention_diagnostics(*(arrays[x][0] for x in ('q','k','v')),w,arrays['query_mask'][0],arrays['key_mask'][0])
                audits.append(dict(case=index,stage=name,audit_projection_seed=seed,metrics=metrics,diagnostic_only_never_used_by_model=True))
        rows.append(fixed_example_diagnostics(net.core,b,baseline=None if baseline is None else baseline[index],save_prefix=persist))
    decision=severe_check(rows,{'prospective_severe_stops':p['severe_stops']})
    write_json(out/f'attention_{step:03d}.json',dict(cases=rows,independent_audit_projections=audits,decision=decision,review_threshold=.20))
    if decision['status']=='STOP':raise FloatingPointError('unchanged severe attention criterion')
    return rows,decision


def execute(arm,record_path):
    record=json.loads(record_path.read_text())
    require_scope(record,PILOT/'prepared_plan.json',TRAIN_ACTION)  # BEFORE any CUDA or optimizer call
    p=load_plan()
    if (PILOT/'study_failure.json').exists():raise RuntimeError('failed pilot cannot retry')
    previous=list(ARMS).index(arm)-1
    if previous>=0 and not (PILOT/ARMS[previous]/'pilot_result.json').exists():raise ValueError('fixed sequential arm order required')
    out=PILOT/arm;out.mkdir(exist_ok=False);g=SharedStudyGuard(record,PILOT/'prepared_plan.json',out,TRAIN_ACTION);updates=[];phase='admission';milestones=[];step=0;optimizer_step_calls=0;returned_optimizer_steps=0
    try:
        initialize_cuda(g);random.seed(42);np.random.seed(42);torch.cuda.manual_seed(42)
        cfg=HiTSKTConfig(**p['model_config']);net=fresh_model(cfg,arm,accumulation_dtype='float64');identity=next(x for x in p['fresh_identities'] if x['arm']==arm)
        if learned_digest(net)!=identity['learned_sha256'] or digest(torch.random.get_rng_state())!=identity['cpu_rng_after_factory_sha256']:raise AssertionError('fresh initialization/RNG mismatch')
        net=net.cuda();opt=torch.optim.Adam(net.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        store=SessionStore(p['store_path']);train=RollingSessionDataset(store,'train',history_sessions=15);val=RollingSessionDataset(store,'validation',history_sessions=15);proof=collate_selected(train,p['training'][0]).to('cuda:0');baseline=None
        for step in range(21):
            if step:
                phase=f'update_{step}';budget_check(g,phase);b=collate_selected(train,p['training'][step-1]).to('cuda:0')
                if b.padded_tokens>2048 or b.target_question.shape[0]!=1:raise ValueError('pilot whole-singleton limit')
                net.train();opt.zero_grad(set_to_none=True);rng=digest(cpu_tree(capture_rng_state()));before=learned_digest(net);torch.cuda.synchronize();stamp=time.perf_counter()
                logits=net(b);loss=masked_bce_loss(logits,b.target_labels,b.target_metric_mask)
                if not torch.isfinite(loss) or float(loss)>20:raise FloatingPointError('nonfinite/excessive training loss')
                loss.backward()
                if any(x.grad is None or not torch.isfinite(x.grad).all() for x in net.parameters()):raise FloatingPointError('nonfinite/missing gradient')
                grad=sum(float(x.grad.double().square().sum()) for x in net.parameters())**.5
                if grad>100:raise FloatingPointError('gradient severe gate')
                budget_check(g,'pre_Adam')
                if time.perf_counter()-stamp>30:raise TimeoutError('preAdam30s gate')
                optimizer_step_calls+=1
                opt.step()  # reachable only after NEW explicit matched-training approval
                returned_optimizer_steps+=1
                torch.cuda.synchronize();seconds=time.perf_counter()-stamp
                if seconds>30 or not finite_tree(net.state_dict()) or not finite_tree(opt.state_dict()):raise FloatingPointError('time/model/Adam severe gate')
                if before==learned_digest(net):raise RuntimeError('parameters did not update')
                row=dict(update=step,training_bce=float(loss),gradient_l2=grad,gradients_and_model_adam_finite=True,pre_update_rng_sha256=rng,seconds=seconds,resources=budget_check(g,'after_Adam'));updates.append(row);write_json(out/f'update_{step:03d}.json',row);time.sleep(.1)
            if step in (0,5,20):
                phase=f'milestone_{step}';metrics,prob,y=evaluate(net,val,p['validation'],g)
                if len(y)!=p['validation_coverage']['targets'] or int(y.sum())!=p['validation_coverage']['positive']:raise ValueError('frozen validation labels changed')
                with (out/f'validation_{step:03d}.npz').open('xb') as f:np.savez_compressed(f,probability=prob.numpy(),label=y.numpy())
                diag,decision=diagnose(net,train,p,g,out,step,baseline)
                if baseline is None:baseline=diag
                milestones.append(dict(update=step,preliminary_sampled_validation=metrics,attention_status=decision['status'],not_full_dataset_performance=True));write_json(out/f'milestone_{step:03d}.json',milestones[-1])
            if step%5==0:
                budget_check(g,'checkpoint');receipt=checkpoint(net,opt,p,step,proof,out/f'checkpoint_{step:03d}.pt');write_json(out/f'checkpoint_{step:03d}.json',receipt)
        write_json(out/'pilot_result.json',dict(status='PASS20_UPDATES',arm=arm,optimizer_updates=20,milestones=milestones,updates=updates,test_evaluations=0,resources=budget_check(g,'arm_complete')))
    except BaseException as exc:
        value=dict(status='STOP_NO_RETRY',arm=arm,phase=phase,last_loop_step=step,fully_verified_optimizer_updates=len(updates),optimizer_step_calls=optimizer_step_calls,returned_optimizer_steps=returned_optimizer_steps,type=type(exc).__name__,message=str(exc),last_cached_resources=g.last,valid_checkpoints=[str(x) for x in sorted(out.glob('checkpoint_*.pt'))],no_cuda_calls_in_handler=True)
        write_json(out/'failure.json',value);write_json(PILOT/'study_failure.json',value);raise


def recover(arm,record_path):
    record=json.loads(record_path.read_text());require_scope(record,PILOT/'prepared_plan.json',TRAIN_ACTION);p=load_plan()
    if (PILOT/'study_failure.json').exists():raise RuntimeError('stop after failed study')
    path=PILOT/arm/'checkpoint_020.pt';v=torch.load(path,map_location='cpu',weights_only=True)
    receipt=json.loads((path.with_suffix('.json')).read_text())
    if receipt['sha256']!=sha256(path):raise ValueError('checkpoint receipt mismatch')
    if v['contract']!=p or v['progress']['update']!=20:raise ValueError('checkpoint contract/progress mismatch')
    for key,value in v['component_sha256'].items():
        if digest(v[key])!=value:raise ValueError('checkpoint component mismatch:'+key)
    out=PILOT/f'{arm}_recovery';out.mkdir(exist_ok=False);g=SharedStudyGuard(record,PILOT/'prepared_plan.json',out,TRAIN_ACTION);initialize_cuda(g)
    net=MatchedComparisonModel.from_checkpoint(path).cuda();opt=torch.optim.Adam(net.parameters(),lr=5e-5);opt.load_state_dict(v['optimizer_state_dict']);restore_rng_state(v['rng_state'])
    train=RollingSessionDataset(SessionStore(p['store_path']),'train',history_sessions=15);b=collate_selected(train,p['training'][0]).to('cuda:0');before=digest(cpu_tree(capture_rng_state()))
    if not torch.equal(fixed_logits(net,b),v['reference_logits']) or probe(net,opt,b)!=v['reference_probe']:raise AssertionError('fresh next-computation recovery mismatch')
    if digest(cpu_tree(capture_rng_state()))!=before or not equal_tree(cpu_tree(opt.state_dict()),v['optimizer_state_dict']):raise AssertionError('recovery changed RNG/Adam')
    if any(float(s['step'])!=20 for s in opt.state_dict()['state'].values()):raise ValueError('Adam counters not20')
    write_json(out/'recovery_result.json',dict(status='PASS_EXACT_FRESH_PROCESS',checkpoint_sha256=sha256(path),optimizer_updates=0,predictions_loss_gradients_and_rng='EXACT',resources=budget_check(g,'recovery_complete')))
